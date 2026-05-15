"""P6 — train the Morgan-fingerprint MLP baseline.

Splits:
  - test drugs (38) held out per data/sciplex/processed/splits.json
  - 90/10 internal val split BY DRUG (seed 1234) within the 150 train drugs

Targets: residual-to-stratum-mean (per-(cell_line, dose) mean computed on
TRAIN drugs only — no leakage from test drugs into the residual baseline).

Outputs to runs/p6_fingerprint/:
  - best.pt         -> best-by-val-pcorr_resid state dict
  - last.pt         -> final-epoch state dict
  - train_summary.json
  - curves.npz      -> per-epoch loss/pcorr histories
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
    INPUT_DIM, StratumMean, build_drug_fp_table, encode_features,
    train_fingerprint_mlp,
)

PARQUET = ROOT / "data/sciplex/processed/cellcast_v0.parquet"
SPLITS_JSON = ROOT / "data/sciplex/processed/splits.json"
HVG_PATH = ROOT / "data/sciplex/processed/hvg_genes.txt"
OUT_DIR = ROOT / "runs/p6_fingerprint"
OUT_DIR.mkdir(parents=True, exist_ok=True)

VAL_FRAC = 0.10
SEED = 1234


def split_by_drug(train_drugs: list[str], val_frac: float, seed: int) -> tuple[list[str], list[str]]:
    rng = np.random.default_rng(seed)
    drugs = list(train_drugs)
    rng.shuffle(drugs)
    n_val = max(1, int(round(len(drugs) * val_frac)))
    val_drugs = sorted(drugs[:n_val])
    train_drugs_inner = sorted(drugs[n_val:])
    return train_drugs_inner, val_drugs


def main():
    t0 = time.time()
    print("[P6.train] loading data ...", flush=True)
    df = pd.read_parquet(PARQUET)
    splits = json.loads(SPLITS_JSON.read_text())
    n_HVG = len([ln for ln in HVG_PATH.read_text().splitlines() if ln.strip()])

    # By-drug 90/10 split within TRAIN drugs
    inner_train_drugs, val_drugs = split_by_drug(splits["train_drugs"], VAL_FRAC, SEED)
    print(f"  train_drugs total: {len(splits['train_drugs'])}  "
          f"-> inner_train: {len(inner_train_drugs)}  val: {len(val_drugs)}",
          flush=True)

    # Build dataframes
    train_df = df[df["drug_name"].isin(inner_train_drugs)].reset_index(drop=True)
    val_df = df[df["drug_name"].isin(val_drugs)].reset_index(drop=True)
    print(f"  conditions: train={len(train_df)}  val={len(val_df)}", flush=True)

    # Stratum means computed on inner_train ONLY (so they're truly OOD for val drugs)
    sm = StratumMean.fit(train_df)
    print(f"  stratum_mean fit: {len(sm.means)} strata, G={sm.G}", flush=True)
    if sm.G != n_HVG:
        raise RuntimeError(f"G mismatch: stratum_mean.G={sm.G}, hvg_genes.txt={n_HVG}")

    # Build fingerprint table
    print("[P6.train] building Morgan fingerprints ...", flush=True)
    fp_table = build_drug_fp_table()
    print(f"  fp_table size: {len(fp_table)}", flush=True)
    # Confirm all train + val drugs covered
    missing_train = [d for d in inner_train_drugs if d not in fp_table]
    missing_val = [d for d in val_drugs if d not in fp_table]
    if missing_train or missing_val:
        raise RuntimeError(f"missing FPs: train={missing_train[:5]}  val={missing_val[:5]}")

    # Encode features
    X_train = encode_features(train_df["drug_name"], train_df["cell_line"],
                              train_df["dose_nM"], fp_table)
    X_val = encode_features(val_df["drug_name"], val_df["cell_line"],
                            val_df["dose_nM"], fp_table)
    Y_train = sm.residual_target(train_df)
    Y_val = sm.residual_target(val_df)
    print(f"  X_train: {X_train.shape}  Y_train: {Y_train.shape}", flush=True)
    print(f"  X_val:   {X_val.shape}  Y_val:   {Y_val.shape}", flush=True)
    print(f"  Y_train stats: mean={Y_train.mean():+.5f}  std={Y_train.std():.5f}  "
          f"|max|={np.abs(Y_train).max():.4f}", flush=True)

    print("[P6.train] training ...", flush=True)
    artifacts = train_fingerprint_mlp(
        X_train=X_train, Y_train=Y_train, X_val=X_val, Y_val=Y_val,
        num_HVGs=n_HVG, epochs=50, batch_size=64, lr=1e-3,
        weight_decay=0.01, warmup_frac=0.05, patience=5, seed=SEED,
    )

    # Persist artifacts
    torch.save(artifacts["best_state_dict"], OUT_DIR / "best.pt")
    print(f"  wrote {OUT_DIR / 'best.pt'}", flush=True)
    np.savez(OUT_DIR / "curves.npz",
             train_loss=np.array(artifacts["train_loss_hist"]),
             val_loss=np.array(artifacts["val_loss_hist"]),
             val_pcorr_resid=np.array(artifacts["val_pcorr_resid_hist"]))
    print(f"  wrote {OUT_DIR / 'curves.npz'}", flush=True)

    summary = {
        "n_inner_train_drugs": len(inner_train_drugs),
        "n_val_drugs": len(val_drugs),
        "n_train_conditions": int(len(train_df)),
        "n_val_conditions": int(len(val_df)),
        "n_HVGs": int(n_HVG),
        "input_dim": int(INPUT_DIM),
        "best_val_pcorr_resid": float(artifacts["best_val_pcorr_resid"]),
        "best_epoch": int(artifacts["best_epoch"]),
        "epochs_run": int(artifacts["epochs_run"]),
        "wall_clock_s": float(artifacts["wall_clock_s"]),
        "warmup_steps": int(artifacts["warmup_steps"]),
        "total_steps_planned": int(artifacts["total_steps_planned"]),
        "seed": SEED,
        "val_drugs": val_drugs,
    }
    (OUT_DIR / "train_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"  wrote {OUT_DIR / 'train_summary.json'}", flush=True)
    print(f"\n[P6.train] DONE  total {time.time()-t0:.1f}s  best epoch {artifacts['best_epoch']} "
          f"val_pcorr_resid={artifacts['best_val_pcorr_resid']:+.4f}", flush=True)


if __name__ == "__main__":
    main()
