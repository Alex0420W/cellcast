"""Shared utilities for the M4A diagnostic probes (P1–P5).

Loads the M3 best checkpoint into a CellCastModule, exposes a forward that
returns the encoder's last hidden state alongside predictions, and identifies
SMILES / MASK / dose / gene token spans for downstream analysis.

Inference + analysis only. No training. No checkpoint writes.
"""
from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(os.path.expanduser("~/cellcast"))
sys.path.insert(0, str(ROOT))

from scripts.train import CellCastModule  # noqa: E402
from src.tasks.drug_response_vector import (  # noqa: E402
    DOSE_TOKENS,
    SLICED_PRED_KEY,
    build_sample_dict,
    configure_frozen_backbone_with_trainable_dose_rows,
    load_or_expand_tokenizer,
    process_model_output,
)

CKPT = ROOT / "runs/cellcast_v0/checkpoints/best-7-816-pcorr=0.1282.ckpt"
CKPT_SHA256 = "2ccb407a889293d133364d52d79293b97e842d139d12c210ea8943fd554a34cd"
PARQUET = ROOT / "data/sciplex/processed/cellcast_v0.parquet"
PARQUET_SHA256 = "404d35e07a0baaa1d1b730a413a81846b845d717c064246aa7ce0e09f48a5b46"
SPLITS = ROOT / "data/sciplex/processed/splits.json"
HVG_PATH = ROOT / "data/sciplex/processed/hvg_genes.txt"
BASELINE = ROOT / "results/baseline_predictions.npz"
PRED_NPZ = ROOT / "results/cellcast_v0_predictions.npz"
OUT_ROOT = ROOT / "results/4a_diagnostics"


# ---- Loaders ---------------------------------------------------------------- #
@dataclass
class LoadedModel:
    lm: torch.nn.Module
    tokenizer_op: object
    device: torch.device
    head: torch.nn.Module
    special_token_ids: dict[str, int]


def load_model() -> LoadedModel:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer_op, _ = load_or_expand_tokenizer()
    ckpt = torch.load(str(CKPT), map_location="cpu", weights_only=False)
    hparams = dict(ckpt.get("hyper_parameters", {}))
    hparams.pop("tokenizer_op", None)
    lm = CellCastModule(**hparams, tokenizer_op=tokenizer_op)
    lm.load_state_dict(ckpt["state_dict"], strict=True)
    configure_frozen_backbone_with_trainable_dose_rows(lm.model, tokenizer_op)
    lm.eval().to(device)

    special_names = (
        "<MASK>", "<MOLECULAR_ENTITY>",
        "<MOLECULAR_ENTITY_SMALL_MOLECULE>", "<SMILES_SEQUENCE>",
        "<MOLECULAR_ENTITY_CELL_GENE_EXPRESSION_RANKED>",
        "<EOS>", "<PAD>",
    ) + DOSE_TOKENS
    special_ids = {}
    for n in special_names:
        try:
            special_ids[n] = tokenizer_op.get_token_id(n)
        except Exception:
            pass
    return LoadedModel(lm=lm, tokenizer_op=tokenizer_op, device=device,
                       head=lm.model.scalars_prediction_head,
                       special_token_ids=special_ids)


# ---- Data ------------------------------------------------------------------- #
def load_test_df() -> pd.DataFrame:
    df = pd.read_parquet(PARQUET)
    splits = json.loads(SPLITS.read_text())
    test = df[df["drug_name"].isin(splits["test_drugs"])].reset_index(drop=True)
    return test


def load_full_df() -> pd.DataFrame:
    return pd.read_parquet(PARQUET)


def load_hvg_count() -> int:
    return len([ln for ln in HVG_PATH.read_text().splitlines() if ln.strip()])


# ---- Forward ---------------------------------------------------------------- #
def build_one_sample(*, smiles: str, dose_bin: str, ranked_genes: list[str],
                     lfc_vector: np.ndarray, tokenizer_op) -> dict:
    return build_sample_dict(
        smiles=smiles, dose_bin=dose_bin, ranked_genes=list(ranked_genes),
        lfc_vector=np.asarray(lfc_vector, dtype=np.float32),
        tokenizer_op=tokenizer_op,
    )


def collate(samples: list[dict]) -> dict:
    from fuse.data.utils.collates import CollateDefault
    return CollateDefault()(samples)


def to_device(batch: dict, device) -> dict:
    for k in ("data.encoder_input_token_ids", "data.encoder_input_attention_mask",
              "data.labels.scalars.values", "data.labels.scalars.valid_mask"):
        if k in batch and torch.is_tensor(batch[k]):
            batch[k] = batch[k].to(device)
    return batch


@dataclass
class ForwardOut:
    last_hidden: torch.Tensor   # [B, S, D]
    pred: torch.Tensor          # [B, G] sliced at MASK
    token_ids: torch.Tensor     # [B, S]
    attention_mask: torch.Tensor  # [B, S]


def forward_with_hidden(loaded: LoadedModel, samples: list[dict],
                        autocast: bool = True) -> ForwardOut:
    """Run forward_encoder_only and return last_hidden + pred + token_ids."""
    batch = collate(samples)
    batch = to_device(batch, loaded.device)
    with torch.inference_mode():
        if autocast and loaded.device.type == "cuda":
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                out = loaded.lm.model.forward_encoder_only(batch)
        else:
            out = loaded.lm.model.forward_encoder_only(batch)
        out = process_model_output(out)
    from mammal.keys import (
        ENCODER_LAST_HIDDEN_STATE,
        ENCODER_INPUTS_TOKENS,
        ENCODER_INPUTS_ATTENTION_MASK,
    )
    return ForwardOut(
        last_hidden=out[ENCODER_LAST_HIDDEN_STATE].float(),
        pred=out[SLICED_PRED_KEY].float(),
        token_ids=out[ENCODER_INPUTS_TOKENS].detach(),
        attention_mask=out[ENCODER_INPUTS_ATTENTION_MASK].detach(),
    )


# ---- Span identification ---------------------------------------------------- #
@dataclass
class Spans:
    """All positions are 0-indexed into the token sequence."""
    mask: int                        # MASK position (always 0 by template)
    smiles_start: int                # first SMILES content token (after SMILES_SEQUENCE marker)
    smiles_end: int                  # exclusive; one past last SMILES content token
    dose_pos: int                    # position of <DOSE_*nM>
    gene_start: int                  # first gene token
    gene_end: int                    # exclusive
    eos: int                         # position of <EOS>
    valid_len: int                   # total non-padding length


def find_spans(token_ids_1d: torch.Tensor, special_ids: dict[str, int],
               attention_mask_1d: torch.Tensor | None = None) -> Spans:
    ids = token_ids_1d.tolist()
    if attention_mask_1d is not None:
        valid_len = int(attention_mask_1d.sum().item())
    else:
        valid_len = len(ids)

    mask_id = special_ids["<MASK>"]
    smiles_seq_id = special_ids["<SMILES_SEQUENCE>"]
    gene_marker_id = special_ids["<MOLECULAR_ENTITY_CELL_GENE_EXPRESSION_RANKED>"]
    eos_id = special_ids["<EOS>"]
    dose_ids = {special_ids[t] for t in DOSE_TOKENS}

    mask_pos = ids.index(mask_id)
    smiles_seq_pos = ids.index(smiles_seq_id)
    smiles_start = smiles_seq_pos + 1

    # dose token: scan from smiles_start forward until we hit a dose id
    dose_pos = next(i for i in range(smiles_start, valid_len) if ids[i] in dose_ids)
    smiles_end = dose_pos  # SMILES content runs from smiles_start to dose_pos (exclusive)

    # gene marker is right after dose token (within ~3 specials); gene content starts after
    gene_marker_pos = ids.index(gene_marker_id, dose_pos)
    gene_start = gene_marker_pos + 1

    # EOS at the end of valid content
    eos_pos = ids.index(eos_id, gene_start) if eos_id in ids[gene_start:valid_len] else valid_len - 1
    gene_end = eos_pos

    return Spans(
        mask=mask_pos,
        smiles_start=smiles_start, smiles_end=smiles_end,
        dose_pos=dose_pos,
        gene_start=gene_start, gene_end=gene_end,
        eos=eos_pos, valid_len=valid_len,
    )


def span_meanpool(hidden: torch.Tensor, start: int, end: int) -> torch.Tensor:
    """Mean-pool hidden[start:end, :] -> [D]."""
    return hidden[start:end, :].mean(dim=0)


# ---- Distance + plotting ---------------------------------------------------- #
def cosine_distance_matrix(vecs: np.ndarray) -> np.ndarray:
    """vecs: [N, D]. Returns [N, N] cosine distance (1 - cos sim)."""
    v = vecs / (np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-12)
    sim = v @ v.T
    return 1.0 - sim


def heatmap(matrix: np.ndarray, labels: list[str], title: str, out_path: Path,
            vmin: float | None = None, vmax: float | None = None,
            cmap: str = "viridis") -> None:
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(max(4, 0.7 * len(labels) + 2),
                                    max(3.5, 0.7 * len(labels) + 1.5)),
                           dpi=150)
    im = ax.imshow(matrix, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(labels, fontsize=8)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            v = matrix[i, j]
            color = "white" if (vmax is None or v > (vmax + (vmin or 0)) / 2) else "black"
            ax.text(j, i, f"{v:.3f}", ha="center", va="center",
                    color=color, fontsize=7)
    ax.set_title(title, fontsize=10)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


# ---- Drug picks for P1 ------------------------------------------------------ #
P1_DRUGS = (
    "Lomustine ",                                  # DNA damage; smallest SMI; alkylating agent
    "Quercetin",                                   # Antioxidant; flavonoid polyphenol
    "Dasatinib",                                   # Tyr kinase; classic kinase scaffold
    "Bisindolylmaleimide IX (Ro 31-8220 Mesylate)",# PKC; bisindolylmaleimide ring system
    "2-Methoxyestradiol (2-MeOE2)",                # HIF; steroid backbone
)
