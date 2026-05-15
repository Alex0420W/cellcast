"""3D-d evaluation: CellCast v0 vs StratifiedMeanBaseline on the held-out test set.

  - Load best checkpoint
  - Predict per-condition LFC vectors for all 456 test conditions
  - Compute macro-Pearson, macro-Spearman, top-50 DEG dir acc, MSE
    overall + per-cell-line, also per pathway_level_1
  - Compare against baseline_predictions.npz
  - Save CellCast preds, JSON metrics, human-readable markdown
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import rankdata

ROOT = Path(os.path.expanduser("~/cellcast"))
sys.path.insert(0, str(ROOT))

from scripts.train import CellCastModule  # noqa: E402
from src.tasks.drug_response_vector import (  # noqa: E402
    SLICED_PRED_KEY,
    build_sample_dict,
    configure_frozen_backbone_with_trainable_dose_rows,
    load_or_expand_tokenizer,
    process_model_output,
)

CKPT = ROOT / "runs/cellcast_v0/checkpoints/best-7-816-pcorr=0.1282.ckpt"
PARQUET = ROOT / "data/sciplex/processed/cellcast_v0.parquet"
HVG_PATH = ROOT / "data/sciplex/processed/hvg_genes.txt"
SPLITS = ROOT / "data/sciplex/processed/splits.json"
BASELINE = ROOT / "results/baseline_predictions.npz"
OUT_NPZ = ROOT / "results/cellcast_v0_predictions.npz"
OUT_JSON = ROOT / "results/3d_metrics.json"
OUT_MD = ROOT / "results/3d_metrics.md"
BATCH = 32


# ---------------- metrics --------------------------------------------------- #
def macro_pearson(P: np.ndarray, T: np.ndarray) -> float:
    Pm = P - P.mean(axis=0, keepdims=True)
    Tm = T - T.mean(axis=0, keepdims=True)
    Ps = Pm.std(axis=0)
    Ts = Tm.std(axis=0)
    valid = (Ps > 1e-8) & (Ts > 1e-8)
    num = (Pm * Tm).mean(axis=0)
    denom = Ps * Ts
    r = np.where(valid, num / np.where(denom == 0, 1.0, denom), np.nan)
    return float(np.nanmean(r))


def macro_spearman(P: np.ndarray, T: np.ndarray) -> float:
    Pr = np.apply_along_axis(rankdata, 0, P)
    Tr = np.apply_along_axis(rankdata, 0, T)
    return macro_pearson(Pr, Tr)


def top_k_dir_acc(P: np.ndarray, T: np.ndarray, k: int = 50) -> float:
    abs_T = np.abs(T)
    out = []
    for i in range(T.shape[0]):
        idx = np.argpartition(abs_T[i], -k)[-k:]
        out.append(float((np.sign(P[i, idx]) == np.sign(T[i, idx])).mean()))
    return float(np.mean(out))


def metrics_block(P: np.ndarray, T: np.ndarray) -> dict:
    return {
        "pcorr_macro": macro_pearson(P, T),
        "spearcorr_macro": macro_spearman(P, T),
        "top50_dir_acc": top_k_dir_acc(P, T, k=50),
        "mse": float(((P - T) ** 2).mean()),
        "n_conditions": int(T.shape[0]),
    }


# ---------------- prediction ------------------------------------------------ #
def _cellcast_collate(samples: list[dict]) -> dict:
    from fuse.data.utils.collates import CollateDefault
    return CollateDefault()(samples)


def predict_cellcast(test_df: pd.DataFrame, n_hvg: int) -> np.ndarray:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer_op, _ = load_or_expand_tokenizer()
    # PyTorch >= 2.6 defaults weights_only=True, which blocks Lightning ckpts
    # carrying numpy types in hparams. Bypass: load with weights_only=False
    # (own ckpt, trusted) and manually restore state dict + hparams.
    ckpt = torch.load(str(CKPT), map_location="cpu", weights_only=False)
    hparams = dict(ckpt.get("hyper_parameters", {}))
    hparams.pop("tokenizer_op", None)
    lm = CellCastModule(**hparams, tokenizer_op=tokenizer_op)
    lm.load_state_dict(ckpt["state_dict"], strict=True)
    # The freeze hook was registered during training; on a fresh module instance
    # we re-apply it before model.to(device) -- needed for grad-mask, harmless
    # for inference but matches train-time state.
    configure_frozen_backbone_with_trainable_dose_rows(lm.model, tokenizer_op)
    lm.eval().to(device)

    preds = np.empty((len(test_df), n_hvg), dtype=np.float32)
    with torch.inference_mode():
        for start in range(0, len(test_df), BATCH):
            chunk = test_df.iloc[start:start + BATCH]
            samples = []
            for _, r in chunk.iterrows():
                samples.append(build_sample_dict(
                    smiles=r["smiles"],
                    dose_bin=r["dose_bin"],
                    ranked_genes=list(r["input_gene_ranked_list"]),
                    lfc_vector=np.asarray(r["label_lfc_vector"], dtype=np.float32),
                    tokenizer_op=tokenizer_op,
                ))
            batch_dict = _cellcast_collate(samples)
            for k in ("data.encoder_input_token_ids", "data.encoder_input_attention_mask",
                      "data.labels.scalars.values", "data.labels.scalars.valid_mask"):
                if k in batch_dict and torch.is_tensor(batch_dict[k]):
                    batch_dict[k] = batch_dict[k].to(device)
            with torch.autocast(device_type="cuda" if device == "cuda" else "cpu", dtype=torch.bfloat16):
                out = lm.model.forward_encoder_only(batch_dict)
            out = process_model_output(out)
            preds[start:start + len(chunk)] = out[SLICED_PRED_KEY].float().cpu().numpy()
            if start % (BATCH * 4) == 0:
                print(f"  {start + len(chunk):>4} / {len(test_df)}", flush=True)
    return preds


# ---------------- main ------------------------------------------------------ #
def main():
    t0 = time.time()
    df = pd.read_parquet(PARQUET)
    split = json.loads(SPLITS.read_text())
    test_df = df[df["drug_name"].isin(split["test_drugs"])].reset_index(drop=True)
    n_hvg = len([ln for ln in HVG_PATH.read_text().splitlines() if ln.strip()])
    print(f"test conditions: {len(test_df)}  n_HVG: {n_hvg}")

    targets = np.stack([np.asarray(v, dtype=np.float32)
                        for v in test_df["label_lfc_vector"]])

    print("[1/3] running CellCast predictions ...")
    cc_preds = predict_cellcast(test_df, n_hvg)
    np.savez(
        OUT_NPZ,
        preds=cc_preds, targets=targets,
        condition_ids=test_df["condition_id"].to_numpy(),
        cell_lines=test_df["cell_line"].to_numpy(),
        drug_names=test_df["drug_name"].to_numpy(),
        dose_nM=test_df["dose_nM"].to_numpy(),
    )
    print(f"  wrote {OUT_NPZ}")

    print("[2/3] loading baseline predictions ...")
    bl = np.load(BASELINE, allow_pickle=True)
    # Order must match test_df
    bl_ids = list(bl["condition_ids"])
    df_ids = list(test_df["condition_id"])
    if bl_ids != df_ids:
        order = {cid: i for i, cid in enumerate(bl_ids)}
        idx = np.array([order[c] for c in df_ids])
        bl_preds = bl["preds"][idx]
    else:
        bl_preds = bl["preds"]
    bl_targets = targets  # same

    print("[3/3] computing metrics ...")
    drug_to_path = split["drug_to_pathway"]
    pathways = np.array([drug_to_path[d] for d in test_df["drug_name"]])

    out_metrics = {
        "overall": {
            "cellcast": metrics_block(cc_preds, targets),
            "baseline": metrics_block(bl_preds, targets),
        },
        "per_cell_line": {},
        "per_pathway": {},
    }
    for cl in sorted(test_df["cell_line"].unique()):
        mask = (test_df["cell_line"] == cl).to_numpy()
        out_metrics["per_cell_line"][cl] = {
            "cellcast": metrics_block(cc_preds[mask], targets[mask]),
            "baseline": metrics_block(bl_preds[mask], targets[mask]),
        }
    for pw in sorted(set(pathways)):
        mask = (pathways == pw)
        if mask.sum() < 2:
            continue
        out_metrics["per_pathway"][pw] = {
            "cellcast": metrics_block(cc_preds[mask], targets[mask]),
            "baseline": metrics_block(bl_preds[mask], targets[mask]),
        }

    OUT_JSON.write_text(json.dumps(out_metrics, indent=2))
    print(f"  wrote {OUT_JSON}")

    # ---- Markdown table ----
    lines: list[str] = []
    lines.append("# 3D metrics — CellCast v0 vs StratifiedMeanBaseline (held-out test, 456 conditions)")
    lines.append("")
    lines.append("## Overall")
    lines.append("")
    o = out_metrics["overall"]
    lines.append("| metric | CellCast | baseline | Δ |")
    lines.append("|---|---:|---:|---:|")
    for m in ("pcorr_macro", "spearcorr_macro", "top50_dir_acc", "mse"):
        c = o["cellcast"][m]; b = o["baseline"][m]
        d = c - b
        sign = ("+" if d >= 0 else "")
        lines.append(f"| {m} | {c:+.4f} | {b:+.4f} | {sign}{d:.4f} |")
    lines.append("")
    lines.append("## Per cell line")
    lines.append("")
    lines.append("| cell_line | metric | CellCast | baseline | Δ |")
    lines.append("|---|---|---:|---:|---:|")
    for cl, blk in out_metrics["per_cell_line"].items():
        for m in ("pcorr_macro", "spearcorr_macro", "top50_dir_acc", "mse"):
            c = blk["cellcast"][m]; b = blk["baseline"][m]
            d = c - b
            sign = ("+" if d >= 0 else "")
            lines.append(f"| {cl} | {m} | {c:+.4f} | {b:+.4f} | {sign}{d:.4f} |")
    lines.append("")
    lines.append("## Per pathway_level_1 (top-50 DEG direction accuracy only, sorted by Δ)")
    lines.append("")
    pw_rows = []
    for pw, blk in out_metrics["per_pathway"].items():
        c = blk["cellcast"]["top50_dir_acc"]
        b = blk["baseline"]["top50_dir_acc"]
        pw_rows.append((pw, c, b, c - b, blk["cellcast"]["n_conditions"]))
    pw_rows.sort(key=lambda r: -r[3])
    lines.append("| pathway | n_conditions | CellCast top50 | baseline top50 | Δ |")
    lines.append("|---|---:|---:|---:|---:|")
    for pw, c, b, d, n in pw_rows:
        sign = ("+" if d >= 0 else "")
        lines.append(f"| {pw} | {n} | {c:+.4f} | {b:+.4f} | {sign}{d:.4f} |")
    OUT_MD.write_text("\n".join(lines) + "\n")
    print(f"  wrote {OUT_MD}")
    print(f"\nDONE  elapsed={time.time()-t0:.1f}s")
    # Quick verdict
    pcorr_delta = out_metrics["overall"]["cellcast"]["pcorr_macro"] - out_metrics["overall"]["baseline"]["pcorr_macro"]
    top50_delta = out_metrics["overall"]["cellcast"]["top50_dir_acc"] - out_metrics["overall"]["baseline"]["top50_dir_acc"]
    print(f"\nVERDICT: Δpcorr={pcorr_delta:+.4f}   Δtop50={top50_delta:+.4f}")
    print(f"  threshold: Δpcorr >= 0.05 OR Δtop50 >= 0.05 to call this 'beats baseline'")


if __name__ == "__main__":
    main()
