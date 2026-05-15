"""P6 — evaluate the trained fingerprint MLP on the held-out 38-drug test set.

Stratum means are refit on the FULL train-drug set (150) — same convention
as M3's StratifiedMeanBaseline (i.e. the baseline the residual is *defined*
relative to at test time uses all 150 train drugs).

Final test prediction = mlp_residual + stratum_mean
This matches M3's apples-to-apples baseline numbers.
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

from src.models.fingerprint_mlp import (  # noqa: E402
    INPUT_DIM, FingerprintMLP, StratumMean,
    build_drug_fp_table, encode_features,
)
from scripts.evaluate import macro_pearson, macro_spearman, top_k_dir_acc  # noqa: E402

PARQUET = ROOT / "data/sciplex/processed/cellcast_v0.parquet"
SPLITS_JSON = ROOT / "data/sciplex/processed/splits.json"
HVG_PATH = ROOT / "data/sciplex/processed/hvg_genes.txt"
RUN_DIR = ROOT / "runs/p6_fingerprint"
BASELINE = ROOT / "results/baseline_predictions.npz"
CC_PRED = ROOT / "results/cellcast_v0_predictions.npz"

OUT_NPZ = ROOT / "results/p6_predictions.npz"
OUT_JSON = ROOT / "results/p6_metrics.json"


def metrics_block(P: np.ndarray, T: np.ndarray) -> dict:
    return {
        "pcorr_macro":     macro_pearson(P, T),
        "spearcorr_macro": macro_spearman(P, T),
        "top50_dir_acc":   top_k_dir_acc(P, T, k=50),
        "mse":             float(((P - T) ** 2).mean()),
        "n_conditions":    int(T.shape[0]),
    }


def main():
    t0 = time.time()
    print("[P6.eval] loading data + checkpoints ...", flush=True)
    df = pd.read_parquet(PARQUET)
    splits = json.loads(SPLITS_JSON.read_text())
    n_HVG = len([ln for ln in HVG_PATH.read_text().splitlines() if ln.strip()])

    train_df = df[df["drug_name"].isin(splits["train_drugs"])].reset_index(drop=True)
    test_df = df[df["drug_name"].isin(splits["test_drugs"])].reset_index(drop=True)
    print(f"  train conds: {len(train_df)}  test conds: {len(test_df)}  G={n_HVG}", flush=True)

    # Stratum means refit on FULL 150 train drugs (test-time convention).
    sm = StratumMean.fit(train_df)
    print(f"  stratum_mean refit on FULL train drugs: {len(sm.means)} strata", flush=True)

    # Fingerprints
    fp_table = build_drug_fp_table()
    missing = [d for d in test_df["drug_name"].unique() if d not in fp_table]
    if missing:
        raise RuntimeError(f"missing FPs for test drugs: {missing}")

    # Encode test features
    X_test = encode_features(test_df["drug_name"], test_df["cell_line"],
                             test_df["dose_nM"], fp_table)
    targets = np.stack([np.asarray(v, dtype=np.float32) for v in test_df["label_lfc_vector"]])

    # Load model + best weights
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = FingerprintMLP(input_dim=INPUT_DIM, num_HVGs=n_HVG).to(device)
    state = torch.load(str(RUN_DIR / "best.pt"), map_location=device, weights_only=True)
    model.load_state_dict(state, strict=True)
    model.eval()
    print("  loaded best.pt", flush=True)

    # Forward
    print("[P6.eval] running forward ...", flush=True)
    with torch.no_grad():
        preds_residual = model(torch.from_numpy(X_test).float().to(device)).float().cpu().numpy()
    preds_full = sm.reconstruct(preds_residual, test_df)

    # Save predictions
    np.savez(OUT_NPZ,
             preds_residual=preds_residual.astype(np.float32),
             preds_full=preds_full.astype(np.float32),
             targets=targets,
             condition_ids=test_df["condition_id"].to_numpy(),
             cell_lines=test_df["cell_line"].to_numpy(),
             drug_names=test_df["drug_name"].to_numpy(),
             dose_nM=test_df["dose_nM"].to_numpy())
    print(f"  wrote {OUT_NPZ}", flush=True)

    # Load StratifiedMeanBaseline + CellCast predictions for direct comparison
    bl_data = np.load(BASELINE, allow_pickle=True)
    bl_ids = list(bl_data["condition_ids"])
    df_ids = list(test_df["condition_id"])
    if bl_ids != df_ids:
        order = {c: i for i, c in enumerate(bl_ids)}
        idx = np.array([order[c] for c in df_ids])
        bl_preds = bl_data["preds"][idx]
    else:
        bl_preds = bl_data["preds"]

    cc_data = np.load(CC_PRED, allow_pickle=True)
    cc_ids = list(cc_data["condition_ids"])
    if cc_ids != df_ids:
        order = {c: i for i, c in enumerate(cc_ids)}
        idx = np.array([order[c] for c in df_ids])
        cc_preds = cc_data["preds"][idx]
    else:
        cc_preds = cc_data["preds"]

    # Metrics
    out_metrics = {
        "overall": {
            "p6_fingerprint_mlp": metrics_block(preds_full, targets),
            "stratified_mean_baseline": metrics_block(bl_preds, targets),
            "cellcast_v0_m3": metrics_block(cc_preds, targets),
        },
        "per_cell_line": {},
    }
    for cl in sorted(test_df["cell_line"].unique()):
        mask = (test_df["cell_line"] == cl).to_numpy()
        out_metrics["per_cell_line"][cl] = {
            "p6_fingerprint_mlp": metrics_block(preds_full[mask], targets[mask]),
            "stratified_mean_baseline": metrics_block(bl_preds[mask], targets[mask]),
            "cellcast_v0_m3": metrics_block(cc_preds[mask], targets[mask]),
        }

    OUT_JSON.write_text(json.dumps(out_metrics, indent=2))
    print(f"  wrote {OUT_JSON}", flush=True)

    o = out_metrics["overall"]
    print(f"\n[P6.eval] OVERALL ({len(test_df)} test conditions)")
    print(f"  {'metric':<18s}  {'P6_FP_MLP':>10s}  {'baseline':>10s}  {'CellCast':>10s}  "
          f"{'Δ_vs_BL':>10s}  {'Δ_vs_CC':>10s}")
    for m in ("pcorr_macro", "spearcorr_macro", "top50_dir_acc", "mse"):
        p6 = o["p6_fingerprint_mlp"][m]
        bl = o["stratified_mean_baseline"][m]
        cc = o["cellcast_v0_m3"][m]
        print(f"  {m:<18s}  {p6:>+10.4f}  {bl:>+10.4f}  {cc:>+10.4f}  "
              f"{p6-bl:>+10.4f}  {p6-cc:>+10.4f}")
    print(f"\n[P6.eval] PER CELL LINE (pcorr_macro)")
    for cl, blk in out_metrics["per_cell_line"].items():
        print(f"  {cl}: P6={blk['p6_fingerprint_mlp']['pcorr_macro']:+.4f}  "
              f"BL={blk['stratified_mean_baseline']['pcorr_macro']:+.4f}  "
              f"CC={blk['cellcast_v0_m3']['pcorr_macro']:+.4f}  "
              f"Δ_BL={blk['p6_fingerprint_mlp']['pcorr_macro']-blk['stratified_mean_baseline']['pcorr_macro']:+.4f}")

    print(f"\n[P6.eval] DONE  {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
