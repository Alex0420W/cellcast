"""DrugResponseVectorTask — CellCast's forked MAMMAL task.

Forked from `mammal/examples/cell_line_drug_response/task.py` with the
modifications agreed in `docs/DECISIONS.md` dated 2026-05-14:

  - **Output:** per-HVG LFC vector instead of scalar IC50. Head widens to
    `num_classes = num_HVGs`; mask slice becomes `scalars_preds[:, 0, :]`.
  - **Labels:** tensor bypass at `LABELS_SCALARS_VALUES` (3A-confirmed).
    No `LABELS_STR` written.
  - **Dose:** discrete `<DOSE_*nM>` token inserted between the SMILES and
    GENE blocks.
  - **Device:** preprocessing stays CPU-side; collate handles .to(device).

EXISTING prompt template (cell_line_drug_response/task.py:146-152) for cross-ref::

    <@TOKENIZER-TYPE=SMILES><MASK>
    <@TOKENIZER-TYPE=SMILES><MOLECULAR_ENTITY><MOLECULAR_ENTITY_SMALL_MOLECULE><SMILES_SEQUENCE>{drug_smiles}
    <@TOKENIZER-TYPE=GENE><MOLECULAR_ENTITY><MOLECULAR_ENTITY_CELL_GENE_EXPRESSION_RANKED>[GENE1][GENE2]...
    <EOS>

NEW prompt template (this module)::

    <@TOKENIZER-TYPE=SMILES><MASK>
    <@TOKENIZER-TYPE=SMILES><MOLECULAR_ENTITY><MOLECULAR_ENTITY_SMALL_MOLECULE><SMILES_SEQUENCE>{smiles}
    <DOSE_*nM>                                                          # NEW: dose token
    <@TOKENIZER-TYPE=GENE><MOLECULAR_ENTITY><MOLECULAR_ENTITY_CELL_GENE_EXPRESSION_RANKED>[GENE1][GENE2]...
    <EOS>

Token-type tags are byte-identical to the existing template.
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch
from fuse.data.tokenizers.modular_tokenizer.op import ModularTokenizerOp

from mammal.keys import (
    ENCODER_INPUTS_ATTENTION_MASK,
    ENCODER_INPUTS_STR,
    ENCODER_INPUTS_TOKENS,
    LABELS_SCALARS_VALID_MASK,
    LABELS_SCALARS_VALUES,
    SCALARS_PREDICTION_HEAD_LOGITS,
)

DOSE_TOKENS = ("<DOSE_10nM>", "<DOSE_100nM>", "<DOSE_1000nM>", "<DOSE_10000nM>")
SLICED_PRED_KEY = "model.out.lfc_pred"  # where the [B, G] sliced predictions land

CELLCAST_TOKENIZER_PATH = os.path.expanduser("~/cellcast/data/tokenizer/cellcast_v0")


# --------------------------------------------------------------------------- #
# Tokenizer + embedding expansion                                             #
# --------------------------------------------------------------------------- #
def _sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_or_expand_tokenizer(
    *,
    base_hf_id: str = "ibm/biomed.omics.bl.sm.ma-ted-458m",
    saved_path: str | os.PathLike = CELLCAST_TOKENIZER_PATH,
    new_tokens: tuple[str, ...] = DOSE_TOKENS,
) -> tuple[ModularTokenizerOp, dict[str, Any]]:
    """Return an expanded ModularTokenizerOp. Idempotent across runs:

      - If `saved_path` exists, load directly from there (assumes the
        on-disk snapshot already has the dose tokens; verified post-load).
      - Otherwise, load from `base_hf_id`, add the dose tokens, save to
        `saved_path`, and return.

    Returns (tokenizer_op, report).  `report['loaded_from']` is either the
    saved path (cache hit) or the HF id (fresh expansion).
    """
    saved_path = Path(saved_path).expanduser()
    report: dict[str, Any] = {}
    if saved_path.exists() and any(saved_path.glob("*.json")):
        tk = ModularTokenizerOp.from_pretrained(str(saved_path))
        # Verify the dose tokens are present in the loaded snapshot.
        for t in new_tokens:
            tk.get_token_id(t)  # raises if missing
        report["loaded_from"] = str(saved_path)
        report["n_added"] = 0
        report["vocab_size"] = tk.get_vocab_size()
        report["token_ids"] = {t: tk.get_token_id(t) for t in new_tokens}
        report["sha256"] = {p.name: _sha256_file(p) for p in sorted(saved_path.rglob("*"))
                            if p.is_file()}
        return tk, report

    tk = ModularTokenizerOp.from_pretrained(base_hf_id)
    report["vocab_before"] = tk.get_vocab_size()
    report["n_added"] = tk.add_new_special_tokens(list(new_tokens))
    report["vocab_after"] = tk.get_vocab_size()
    report["token_ids"] = {t: tk.get_token_id(t) for t in new_tokens}

    saved_path.mkdir(parents=True, exist_ok=True)
    tk.save_pretrained(str(saved_path))
    report["loaded_from"] = base_hf_id
    report["saved_to"] = str(saved_path)
    report["sha256"] = {p.name: _sha256_file(p) for p in sorted(saved_path.rglob("*"))
                        if p.is_file()}
    return tk, report


def init_dose_token_embeddings(
    model,
    tokenizer_op: ModularTokenizerOp,
    new_tokens: tuple[str, ...] = DOSE_TOKENS,
    used_vocab_size: int = 99028,
    perturbation_frac: float = 0.10,
    seed: int = 42,
) -> dict[str, Any]:
    """Initialize dose-token embedding rows: mean-of-pretrained + per-token
    orthogonal perturbation.

    Per DECISIONS.md 2026-05-14 ("Dose embedding init + trainable rows"):
      - Mean centers the magnitude (avoids the small-norm "hole row" init).
      - Per-token orthogonal perturbation differentiates the four tokens so
        the frozen encoder sees four distinguishable inputs at position N.
      - Perturbations are scaled to ~`perturbation_frac` of the mean's L2 norm.
      - Dose tokens are ordered low->high (10nM, 100nM, 1000nM, 10000nM) and
        assigned perturbation directions in that order; the dose-ID ordering
        in MAMMAL's token table happens to be reverse, but we don't rely on
        that — we order by dose magnitude.

    Idempotent: safe to call before training. Should NOT be called when
    loading a checkpoint with already-trained dose embeddings.
    """
    emb = model.t5_model.get_input_embeddings()
    needed_size = max(tokenizer_op.get_token_id(t) for t in new_tokens) + 1
    resize_done = False
    if emb.num_embeddings < needed_size:
        model.resize_token_embeddings(needed_size)
        emb = model.t5_model.get_input_embeddings()
        resize_done = True

    # Order dose tokens by ascending dose magnitude so that perturbation index
    # i corresponds to dose rank i (10nM gets perturbation 0, 10000nM gets 3).
    def _dose_value(tok: str) -> float:
        s = tok.replace("<DOSE_", "").replace("nM>", "")
        return float(s)
    new_tokens_sorted = tuple(sorted(new_tokens, key=_dose_value))

    dim = emb.embedding_dim
    g = torch.Generator(device="cpu").manual_seed(seed)
    rand = torch.randn(dim, len(new_tokens_sorted), generator=g, dtype=torch.float32)
    q, _ = torch.linalg.qr(rand, mode="reduced")  # [dim, k] orthonormal columns

    with torch.no_grad():
        device = emb.weight.device
        mean_vec = emb.weight[:used_vocab_size].float().mean(dim=0)  # on `device`
        mean_norm = mean_vec.norm().item()
        target_perturb_scale = perturbation_frac * mean_norm
        perturb_vectors = (q.t() * target_perturb_scale).to(device=device)  # [k, dim]

        token_ids_assigned: dict[str, int] = {}
        rows_after: dict[int, torch.Tensor] = {}
        for i, tok in enumerate(new_tokens_sorted):
            tid = tokenizer_op.get_token_id(tok)
            new_row = mean_vec + perturb_vectors[i]
            emb.weight[tid].copy_(new_row.to(emb.weight.dtype))
            token_ids_assigned[tok] = tid
            rows_after[tid] = emb.weight[tid].detach().float().cpu()

    # Pairwise L2 + cosine distance diagnostics so callers can confirm distinctness.
    row_list = [rows_after[token_ids_assigned[t]] for t in new_tokens_sorted]
    pairwise_l2: dict[str, float] = {}
    pairwise_cos: dict[str, float] = {}
    for i in range(len(new_tokens_sorted)):
        for j in range(i + 1, len(new_tokens_sorted)):
            a = row_list[i]
            b = row_list[j]
            l2 = (a - b).norm().item()
            cos = torch.nn.functional.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()
            pairwise_l2[f"{new_tokens_sorted[i]}|{new_tokens_sorted[j]}"] = l2
            pairwise_cos[f"{new_tokens_sorted[i]}|{new_tokens_sorted[j]}"] = cos

    return {
        "emb_resize_done": resize_done,
        "emb_num_embeddings": emb.num_embeddings,
        "embedding_dim": emb.embedding_dim,
        "mean_norm": mean_norm,
        "perturb_scale": target_perturb_scale,
        "perturb_method": "QR-orthonormal columns of random Gaussian, ordered by ascending dose",
        "seed": seed,
        "token_ids_in_dose_order": [token_ids_assigned[t] for t in new_tokens_sorted],
        "pairwise_l2_distance": pairwise_l2,
        "pairwise_cosine_similarity": pairwise_cos,
    }


def configure_frozen_backbone_with_trainable_dose_rows(
    model,
    tokenizer_op: ModularTokenizerOp,
    new_tokens: tuple[str, ...] = DOSE_TOKENS,
) -> dict[str, Any]:
    """Freeze everything except the prediction head and the 4 dose-token rows.

    Implementation:
      1. Set requires_grad=False on every parameter.
      2. Set requires_grad=True on `scalars_prediction_head.*`.
      3. Set requires_grad=True on the input embedding weight, then register
         a backward hook that masks gradient rows outside the dose-token IDs.

    Returns a report dict with trainable-param counts and the masked id range.
    """
    dose_ids = sorted(tokenizer_op.get_token_id(t) for t in new_tokens)

    # 1. Freeze everything
    for p in model.parameters():
        p.requires_grad_(False)

    # 2. Unfreeze the prediction head
    if model.scalars_prediction_head is None:
        raise RuntimeError("model.scalars_prediction_head is not attached")
    for p in model.scalars_prediction_head.parameters():
        p.requires_grad_(True)

    # 3. Unfreeze input-embedding weight but mask gradient outside dose rows.
    emb = model.t5_model.get_input_embeddings()
    emb.weight.requires_grad_(True)

    dose_id_tensor = torch.tensor(dose_ids, dtype=torch.long)

    def _mask_grad(grad: torch.Tensor) -> torch.Tensor:
        masked = torch.zeros_like(grad)
        masked.index_copy_(0, dose_id_tensor.to(grad.device), grad.index_select(0, dose_id_tensor.to(grad.device)))
        return masked

    # Stash to allow detach during checkpoint load.
    handle = emb.weight.register_hook(_mask_grad)
    if not hasattr(model, "_cellcast_hooks"):
        model._cellcast_hooks = []
    model._cellcast_hooks.append(handle)

    head_params = sum(p.numel() for p in model.scalars_prediction_head.parameters() if p.requires_grad)
    dose_rows_params = len(dose_ids) * emb.embedding_dim
    return {
        "dose_token_ids": dose_ids,
        "embedding_dim": emb.embedding_dim,
        "head_trainable_params": head_params,
        "dose_rows_trainable_params": dose_rows_params,
        "total_trainable_params": head_params + dose_rows_params,
        "gradient_mask": f"backward hook on input-embedding masks rows outside {dose_ids}",
    }


def expand_tokenizer_and_embeddings(
    tokenizer_op: ModularTokenizerOp,
    model,
    new_tokens: tuple[str, ...] = DOSE_TOKENS,
    *,
    saved_path: str | os.PathLike | None = CELLCAST_TOKENIZER_PATH,
) -> dict[str, Any]:
    """Compatibility wrapper preserving the original signature.

    Adds the dose tokens to `tokenizer_op` in-place, initializes the
    corresponding embedding rows on `model`, and persists the expanded
    tokenizer to `saved_path` if provided.
    """
    report: dict[str, Any] = {}
    report["vocab_before"] = tokenizer_op.get_vocab_size()
    report["n_added"] = tokenizer_op.add_new_special_tokens(list(new_tokens))
    report["vocab_after"] = tokenizer_op.get_vocab_size()

    token_ids = {t: tokenizer_op.get_token_id(t) for t in new_tokens}
    report["token_ids"] = token_ids

    # Differentiated init via mean-plus-orthogonal-perturbation.
    # See DECISIONS.md 2026-05-14 "Dose embedding init + trainable rows".
    init_report = init_dose_token_embeddings(
        model, tokenizer_op, new_tokens=new_tokens,
        used_vocab_size=report["vocab_before"],
    )
    report.update({f"emb_{k}": v for k, v in init_report.items()})

    # Persist the expanded tokenizer (idempotent; overwrite OK).
    if saved_path is not None:
        sp = Path(saved_path).expanduser()
        sp.mkdir(parents=True, exist_ok=True)
        tokenizer_op.save_pretrained(str(sp))
        report["saved_to"] = str(sp)
        report["sha256"] = {p.name: _sha256_file(p) for p in sorted(sp.rglob("*"))
                            if p.is_file()}
    return report


# --------------------------------------------------------------------------- #
# Sample-dict construction                                                    #
# --------------------------------------------------------------------------- #
def build_sample_dict(
    *,
    smiles: str,
    dose_bin: str,
    ranked_genes: list[str],
    lfc_vector: np.ndarray,
    tokenizer_op: ModularTokenizerOp,
    encoder_input_max_seq_len: int = 1500,
) -> dict:
    """Build one CPU-side sample_dict for the model. No device moves."""
    if dose_bin not in {t.strip("<>") for t in DOSE_TOKENS}:
        # accept either bare "DOSE_100nM" or full token "<DOSE_100nM>"
        if dose_bin not in DOSE_TOKENS:
            raise ValueError(f"unknown dose_bin: {dose_bin!r}")
        dose_tok = dose_bin
    else:
        dose_tok = f"<{dose_bin}>"

    genes_formatted = [f"[{g}]" for g in ranked_genes]

    # Identical token-type tags to the existing task (cell_line_drug_response/task.py:147–151).
    prompt = (
        "<@TOKENIZER-TYPE=SMILES><MASK>"
        f"<@TOKENIZER-TYPE=SMILES><MOLECULAR_ENTITY><MOLECULAR_ENTITY_SMALL_MOLECULE><SMILES_SEQUENCE>{smiles}"
        f"{dose_tok}"
        "<@TOKENIZER-TYPE=GENE><MOLECULAR_ENTITY><MOLECULAR_ENTITY_CELL_GENE_EXPRESSION_RANKED>"
        + "".join(genes_formatted)
        + "<EOS>"
    )

    sample_dict: dict = {ENCODER_INPUTS_STR: prompt}
    tokenizer_op(
        sample_dict=sample_dict,
        key_in=ENCODER_INPUTS_STR,
        key_out_tokens_ids=ENCODER_INPUTS_TOKENS,
        key_out_attention_mask=ENCODER_INPUTS_ATTENTION_MASK,
        max_seq_len=encoder_input_max_seq_len,
        on_unknown="warn",
        verbose=0,
    )
    sample_dict[ENCODER_INPUTS_TOKENS] = torch.tensor(sample_dict[ENCODER_INPUTS_TOKENS], dtype=torch.long)
    sample_dict[ENCODER_INPUTS_ATTENTION_MASK] = torch.tensor(
        sample_dict[ENCODER_INPUTS_ATTENTION_MASK], dtype=torch.long
    )

    # Tensor bypass for the regression label (3A-confirmed). No LABELS_STR written.
    # Explicit copy avoids "non-writable numpy array" warnings when the input
    # came from pyarrow (parquet columns return read-only numpy views).
    lfc_arr = np.array(lfc_vector, dtype=np.float32, copy=True)
    sample_dict[LABELS_SCALARS_VALUES] = torch.from_numpy(lfc_arr)
    sample_dict[LABELS_SCALARS_VALID_MASK] = torch.ones(len(lfc_arr), dtype=torch.bool)

    return sample_dict


# --------------------------------------------------------------------------- #
# Output processing                                                            #
# --------------------------------------------------------------------------- #
def process_model_output(batch_dict: dict, *, sliced_pred_key: str = SLICED_PRED_KEY) -> dict:
    """Slice the <MASK>-position prediction. Shape goes [B, S, G] -> [B, G]."""
    raw = batch_dict[SCALARS_PREDICTION_HEAD_LOGITS]  # [B, S, G]
    if raw.dim() != 3:
        raise RuntimeError(f"expected scalars_pred [B,S,G], got shape {tuple(raw.shape)}")
    batch_dict[sliced_pred_key] = raw[:, 0, :]
    return batch_dict


# --------------------------------------------------------------------------- #
# Metric functions (per-gene macro)                                            #
# --------------------------------------------------------------------------- #
def _stack(items: list) -> np.ndarray:
    """MetricDefault hands us a list of tensors/arrays. Stack into [N, G]."""
    parts = []
    for it in items:
        x = it.detach().cpu().numpy() if hasattr(it, "detach") else np.asarray(it)
        if x.ndim == 1:
            x = x[None, :]
        parts.append(x)
    return np.concatenate(parts, axis=0)


def per_gene_pearson_macro(pred: list, target: list) -> float:
    """Macro-averaged per-gene Pearson correlation across the HVG axis."""
    P = _stack(pred)        # [N, G]
    T = _stack(target)
    # Standardize columns
    Pm = P - P.mean(axis=0, keepdims=True)
    Tm = T - T.mean(axis=0, keepdims=True)
    Pstd = Pm.std(axis=0)
    Tstd = Tm.std(axis=0)
    valid = (Pstd > 1e-8) & (Tstd > 1e-8)
    num = (Pm * Tm).mean(axis=0)
    denom = (Pstd * Tstd)
    r = np.where(valid, num / np.where(denom == 0, 1.0, denom), np.nan)
    return float(np.nanmean(r))


def per_gene_spearman_macro(pred: list, target: list) -> float:
    """Macro-averaged per-gene Spearman correlation across the HVG axis."""
    from scipy.stats import rankdata
    P = _stack(pred)
    T = _stack(target)
    Pr = np.apply_along_axis(rankdata, 0, P)
    Tr = np.apply_along_axis(rankdata, 0, T)
    return per_gene_pearson_macro([Pr], [Tr])


def top_k_deg_direction_accuracy(pred: list, target: list, *, k: int = 50) -> float:
    """For each condition, take the top-k genes by |true LFC|; fraction whose predicted
    sign matches the true sign."""
    P = _stack(pred)
    T = _stack(target)
    if P.shape != T.shape:
        raise RuntimeError(f"shape mismatch P={P.shape} T={T.shape}")
    abs_T = np.abs(T)
    accs = []
    for i in range(T.shape[0]):
        top = np.argpartition(abs_T[i], -k)[-k:]
        sign_match = (np.sign(P[i, top]) == np.sign(T[i, top])).mean()
        accs.append(float(sign_match))
    return float(np.mean(accs))


# --------------------------------------------------------------------------- #
# Convenience: collate for moving CPU tensors -> device                        #
# --------------------------------------------------------------------------- #
def collate_to_device(
    samples: list[dict],
    device: str | torch.device,
) -> dict:
    """Batched collate with explicit device move. Standard PyTorch DataLoader idiom;
    keeps preprocessing CPU-side per DECISIONS.md 2026-05-14."""
    from fuse.data.utils.collates import CollateDefault
    batch_dict = CollateDefault()(samples)
    # Move the tensors the model needs to the requested device.
    for k in (ENCODER_INPUTS_TOKENS, ENCODER_INPUTS_ATTENTION_MASK,
              LABELS_SCALARS_VALUES, LABELS_SCALARS_VALID_MASK):
        if k in batch_dict and torch.is_tensor(batch_dict[k]):
            batch_dict[k] = batch_dict[k].to(device)
    return batch_dict
