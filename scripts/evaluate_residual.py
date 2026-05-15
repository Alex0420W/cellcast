"""M4B.1 — evaluate the trained CellCast-residual model on the held-out test set.

Loads the best checkpoint, runs forward inference (which produces RESIDUAL
predictions because that's what the head was trained to output), then
RECONSTRUCTS full-LFC predictions:

    full_pred = residual_pred + stratum_mean_full_train_drugs

The stratum mean used at test time is refit on ALL 150 train drugs (matches
StratifiedMeanBaseline's test-time convention from M3) — this is the
apples-to-apples comparison with M3's baseline numbers.

Metrics: same as M3 + P6 — macro Pearson, macro Spearman, top-50 DEG dir
acc, MSE — overall and per cell line. Comparison includes:
  - StratifiedMeanBaseline (M3)
  - CellCast v0 (M3)
  - P6 Fingerprint MLP
  - CellCast v0 RESIDUAL (this run)
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

ROOT = Path(os.path.expanduser("~/cellcast"))
sys.path.insert(0, str(ROOT))

from scripts.train_residual import CellCastResidualModule  # noqa: E402
from src.tasks.drug_response_residual import (  # noqa: E402
    SLICED_PRED_KEY,
    StratumMean,
    build_sample_dict,
    configure_frozen_backbone_with_trainable_dose_rows,
    load_or_expand_tokenizer,
    process_model_output,
)
from scripts.evaluate import macro_pearson, macro_spearman, top_k_dir_acc  # noqa: E402

PARQUET = ROOT / "data/sciplex/processed/cellcast_v0.parquet"
SPLITS_JSON = ROOT / "data/sciplex/processed/splits.json"
HVG_PATH = ROOT / "data/sciplex/processed/hvg_genes.txt"
RUN_DIR = ROOT / "runs/cellcast_v0_residual"
BASELINE_NPZ = ROOT / "results/baseline_predictions.npz"
CC_M3_NPZ = ROOT / "results/cellcast_v0_predictions.npz"

OUT_NPZ = ROOT / "results/cellcast_residual_predictions.npz"
OUT_JSON = ROOT / "results/m4b1_metrics.json"
BATCH = 32


def metrics_block(P: np.ndarray, T: np.ndarray) -> dict:
    return {
        "pcorr_macro":     macro_pearson(P, T),
        "spearcorr_macro": macro_spearman(P, T),
        "top50_dir_acc":   top_k_dir_acc(P, T, k=50),
        "mse":             float(((P - T) ** 2).mean()),
        "n_conditions":    int(T.shape[0]),
    }


def find_best_ckpt(run_dir: Path) -> Path:
    cks = sorted((run_dir / "checkpoints").glob("best-*.ckpt"))
    if not cks:
        raise FileNotFoundError(f"no best-*.ckpt in {run_dir / 'checkpoints'}")
    return cks[-1]


def predict_residual(test_df: pd.DataFrame, n_hvg: int) -> np.ndarray:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer_op, _ = load_or_expand_tokenizer()
    ckpt_path = find_best_ckpt(RUN_DIR)
    print(f"  loading {ckpt_path.name}", flush=True)

    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    hparams = dict(ckpt.get("hyper_parameters", {}))
    hparams.pop("tokenizer_op", None)
    lm = CellCastResidualModule(**hparams, tokenizer_op=tokenizer_op)
    lm.load_state_dict(ckpt["state_dict"], strict=True)
    configure_frozen_backbone_with_trainable_dose_rows(lm.model, tokenizer_op)
    lm.eval().to(device)

    preds = np.empty((len(test_df), n_hvg), dtype=np.float32)
    from fuse.data.utils.collates import CollateDefault
    with torch.inference_mode():
        for start in range(0, len(test_df), BATCH):
            chunk = test_df.iloc[start:start + BATCH]
            samples = []
            for _, r in chunk.iterrows():
                samples.append(build_sample_dict(
                    smiles=r["smiles"], dose_bin=r["dose_bin"],
                    ranked_genes=list(r["input_gene_ranked_list"]),
                    # Use the FULL LFC for the placeholder LABELS_SCALARS_VALUES;
                    # it doesn't enter the forward path, so any vector of the
                    # right shape works.
                    lfc_vector=np.asarray(r["label_lfc_vector"], dtype=np.float32),
                    tokenizer_op=tokenizer_op,
                ))
            batch_dict = CollateDefault()(samples)
            for k in ("data.encoder_input_token_ids", "data.encoder_input_attention_mask",
                      "data.labels.scalars.values", "data.labels.scalars.valid_mask"):
                if k in batch_dict and torch.is_tensor(batch_dict[k]):
                    batch_dict[k] = batch_dict[k].to(device)
            with torch.autocast(device_type="cuda" if device == "cuda" else "cpu",
                                dtype=torch.bfloat16):
                out = lm.model.forward_encoder_only(batch_dict)
            out = process_model_output(out)
            preds[start:start + len(chunk)] = out[SLICED_PRED_KEY].float().cpu().numpy()
            if start % (BATCH * 4) == 0:
                print(f"  {start + len(chunk):>4} / {len(test_df)}", flush=True)
    return preds


def main():
    t0 = time.time()
    df = pd.read_parquet(PARQUET)
    splits = json.loads(SPLITS_JSON.read_text())
    n_hvg = len([ln for ln in HVG_PATH.read_text().splitlines() if ln.strip()])

    train_df = df[df["drug_name"].isin(splits["train_drugs"])].reset_index(drop=True)
    test_df = df[df["drug_name"].isin(splits["test_drugs"])].reset_index(drop=True)
    print(f"  train conds={len(train_df)}  test conds={len(test_df)}  G={n_hvg}", flush=True)

    # Stratum means refit on FULL 150 train drugs (matches StratifiedMeanBaseline
    # test-time convention from M3).
    sm_test = StratumMean.fit(train_df)
    print(f"  stratum_mean refit on FULL train drugs: {len(sm_test.means)} strata", flush=True)

    targets_full = np.stack([np.asarray(v, dtype=np.float32)
                             for v in test_df["label_lfc_vector"]])

    print("[1/3] running CellCast-residual predictions ...", flush=True)
    preds_residual = predict_residual(test_df, n_hvg)
    preds_full = sm_test.reconstruct(preds_residual, test_df)
    np.savez(
        OUT_NPZ,
        preds_residual=preds_residual,
        preds_full=preds_full,
        targets=targets_full,
        condition_ids=test_df["condition_id"].to_numpy(),
        cell_lines=test_df["cell_line"].to_numpy(),
        drug_names=test_df["drug_name"].to_numpy(),
        dose_nM=test_df["dose_nM"].to_numpy(),
    )
    print(f"  wrote {OUT_NPZ}", flush=True)

    print("[2/3] loading comparison predictions ...", flush=True)
    bl = np.load(BASELINE_NPZ, allow_pickle=True)
    cc = np.load(CC_M3_NPZ, allow_pickle=True)

    def align(other_preds_npz):
        ids = list(other_preds_npz["condition_ids"])
        df_ids = list(test_df["condition_id"])
        if ids != df_ids:
            order = {c: i for i, c in enumerate(ids)}
            idx = np.array([order[c] for c in df_ids])
            return other_preds_npz["preds"][idx]
        return other_preds_npz["preds"]

    bl_preds = align(bl)
    cc_preds = align(cc)

    p6_npz = ROOT / "results" / "p6_predictions.npz"
    if p6_npz.exists():
        p6 = np.load(p6_npz, allow_pickle=True)
        p6_ids = list(p6["condition_ids"])
        df_ids = list(test_df["condition_id"])
        if p6_ids != df_ids:
            order = {c: i for i, c in enumerate(p6_ids)}
            idx = np.array([order[c] for c in df_ids])
            p6_preds = p6["preds_full"][idx]
        else:
            p6_preds = p6["preds_full"]
    else:
        # P6 NPZ is gitignored; if missing, regen note for the report.
        p6_preds = None
        print("  WARNING: p6_predictions.npz missing — P6 column will be NaN. "
              "Regen via: python scripts/evaluate_fingerprint.py", flush=True)

    print("[3/3] computing metrics ...", flush=True)
    out_metrics = {
        "overall": {
            "cellcast_residual_m4b1": metrics_block(preds_full, targets_full),
            "stratified_mean_baseline": metrics_block(bl_preds, targets_full),
            "cellcast_v0_m3": metrics_block(cc_preds, targets_full),
        },
        "per_cell_line": {},
    }
    if p6_preds is not None:
        out_metrics["overall"]["p6_fingerprint_mlp"] = metrics_block(p6_preds, targets_full)

    for cl in sorted(test_df["cell_line"].unique()):
        mask = (test_df["cell_line"] == cl).to_numpy()
        out_metrics["per_cell_line"][cl] = {
            "cellcast_residual_m4b1": metrics_block(preds_full[mask], targets_full[mask]),
            "stratified_mean_baseline": metrics_block(bl_preds[mask], targets_full[mask]),
            "cellcast_v0_m3": metrics_block(cc_preds[mask], targets_full[mask]),
        }
        if p6_preds is not None:
            out_metrics["per_cell_line"][cl]["p6_fingerprint_mlp"] = metrics_block(
                p6_preds[mask], targets_full[mask]
            )

    OUT_JSON.write_text(json.dumps(out_metrics, indent=2))
    print(f"  wrote {OUT_JSON}", flush=True)

    # ---- Print headline tables ---- #
    print(f"\n[m4b1.eval] OVERALL ({len(test_df)} test conditions)")
    cols = ["m4b1_residual", "baseline", "cellcast_v0"]
    keys = ["cellcast_residual_m4b1", "stratified_mean_baseline", "cellcast_v0_m3"]
    if p6_preds is not None:
        cols.append("P6_FP_MLP"); keys.append("p6_fingerprint_mlp")
    header = f"  {'metric':<18s}  " + "  ".join(f"{c:>14s}" for c in cols)
    print(header)
    for m in ("pcorr_macro", "spearcorr_macro", "top50_dir_acc", "mse"):
        vals = [out_metrics['overall'][k][m] for k in keys]
        line = "  ".join(f"{v:>+14.4f}" for v in vals)
        print(f"  {m:<18s}  {line}")

    print(f"\n[m4b1.eval] PER CELL LINE — pcorr_macro (the M4B primary metric)")
    print(f"  {'cl':<5s}  " + "  ".join(f"{c:>14s}" for c in cols))
    for cl in sorted(out_metrics["per_cell_line"]):
        blk = out_metrics["per_cell_line"][cl]
        vals = [blk[k]["pcorr_macro"] for k in keys]
        line = "  ".join(f"{v:>+14.4f}" for v in vals)
        print(f"  {cl:<5s}  {line}")

    print(f"\n[m4b1.eval] PER CELL LINE — top50_dir_acc")
    print(f"  {'cl':<5s}  " + "  ".join(f"{c:>14s}" for c in cols))
    for cl in sorted(out_metrics["per_cell_line"]):
        blk = out_metrics["per_cell_line"][cl]
        vals = [blk[k]["top50_dir_acc"] for k in keys]
        line = "  ".join(f"{v:>+14.4f}" for v in vals)
        print(f"  {cl:<5s}  {line}")

    print(f"\n[m4b1.eval] DONE  {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
