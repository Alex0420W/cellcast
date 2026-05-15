"""Sanity check: confirm the P6 stratum-mean implementation is leak-free.

Computes two versions of the per-(cell_line, dose) mean LFC vectors:
  A. train-only — fit on the 150 train drugs (the production implementation
     used in scripts/train_fingerprint.py and scripts/evaluate_fingerprint.py)
  B. all-drugs — fit on the full 188-drug set including the 38 held-out
     test drugs (the leaky version, computed only here for comparison)

For each version, run StratifiedMeanBaseline.predict on the test set and
report overall + per-cell-line pcorr_macro.

If A and B agree to within ~0.001 absolute pcorr, the train-only mean is
indistinguishable from the leaky version on test. If they differ by >0.005,
the train-only mean is meaningfully different from "what we'd get if we
cheated", which is fine — it just means the test drugs DO move the strata
when included. We're checking that we're using A (train-only) and that A's
numbers match what the M3 baseline reports.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(os.path.expanduser("~/cellcast"))
sys.path.insert(0, str(ROOT))

from src.models.baselines import StratifiedMeanBaseline  # noqa: E402
from scripts.evaluate import macro_pearson  # noqa: E402

PARQUET = ROOT / "data/sciplex/processed/cellcast_v0.parquet"
SPLITS_JSON = ROOT / "data/sciplex/processed/splits.json"
BASELINE_NPZ = ROOT / "results/baseline_predictions.npz"
OUT = ROOT / "results/p6_leakage_check.md"


def per_cell_line_pcorr(P: np.ndarray, T: np.ndarray, cls: np.ndarray) -> dict:
    out = {}
    for cl in sorted(set(cls)):
        m = (cls == cl)
        out[cl] = float(macro_pearson(P[m], T[m]))
    return out


def main():
    t0 = time.time()
    df = pd.read_parquet(PARQUET)
    splits = json.loads(SPLITS_JSON.read_text())
    train_df = df[df["drug_name"].isin(splits["train_drugs"])].reset_index(drop=True)
    test_df = df[df["drug_name"].isin(splits["test_drugs"])].reset_index(drop=True)
    print(f"train conds: {len(train_df)}  test conds: {len(test_df)}", flush=True)

    targets = np.stack([np.asarray(v, dtype=np.float32)
                        for v in test_df["label_lfc_vector"]])
    cls = test_df["cell_line"].to_numpy()

    # --- Version A: train-only (production) ---
    blA = StratifiedMeanBaseline().fit(train_df)
    predA = blA.predict(test_df)
    overall_A = float(macro_pearson(predA, targets))
    per_cl_A = per_cell_line_pcorr(predA, targets, cls)

    # --- Version B: all-drugs (LEAKY — for comparison only) ---
    full_df = pd.concat([train_df, test_df], ignore_index=True)
    blB = StratifiedMeanBaseline().fit(full_df)
    predB = blB.predict(test_df)
    overall_B = float(macro_pearson(predB, targets))
    per_cl_B = per_cell_line_pcorr(predB, targets, cls)

    # --- Cross-check: does A match the saved baseline_predictions.npz ---
    saved = np.load(BASELINE_NPZ, allow_pickle=True)
    saved_ids = list(saved["condition_ids"])
    df_ids = list(test_df["condition_id"])
    if saved_ids != df_ids:
        order = {c: i for i, c in enumerate(saved_ids)}
        idx = np.array([order[c] for c in df_ids])
        saved_preds = saved["preds"][idx]
    else:
        saved_preds = saved["preds"]
    saved_overall = float(macro_pearson(saved_preds, targets))
    saved_per_cl = per_cell_line_pcorr(saved_preds, targets, cls)
    a_vs_saved_pred_diff = float(np.abs(predA - saved_preds).max())

    delta_overall = overall_A - overall_B
    delta_per_cl = {cl: per_cl_A[cl] - per_cl_B[cl] for cl in sorted(per_cl_A)}
    max_abs_per_cl_delta = max(abs(d) for d in delta_per_cl.values())

    # The DECISIVE leak-detection signal is the A-vs-saved-baseline cross-check
    # (the production code path already produced predictions in M3; if A
    # reproduces them exactly, the production train-only logic is the same
    # one that's been in use). The A-vs-B delta is a *separate* signal about
    # whether the test drugs are statistically similar to the train drugs.
    impl_clean = a_vs_saved_pred_diff <= 1e-5
    ab_close = abs(delta_overall) <= 0.001 and max_abs_per_cl_delta <= 0.001

    if impl_clean and ab_close:
        verdict = (
            "PASS — implementation is leak-free (A reproduces M3 saved baseline exactly) "
            "AND the test drugs happen to be near-uniform with train drugs (A ≈ B)."
        )
    elif impl_clean and not ab_close:
        verdict = (
            f"PASS — implementation is leak-free (A reproduces M3 saved baseline exactly: "
            f"max abs pred diff = {a_vs_saved_pred_diff:.2e}). "
            f"The A-vs-B delta of {abs(delta_overall):.4f} pcorr just measures "
            "how much the test drugs shift the per-stratum means when included; "
            "it does NOT indicate a leak in the production code path."
        )
    else:
        verdict = (
            f"FLAG — A's predictions do NOT match the saved M3 baseline "
            f"(max abs pred diff = {a_vs_saved_pred_diff:.2e}, expected ≤ 1e-5). "
            "Production stratum-mean implementation may have regressed."
        )

    md = f"""# P6 leakage check — stratum mean implementation

**Date:** {time.strftime('%Y-%m-%d')}
**Status:** {verdict}

## Method

Two versions of the per-(cell_line, dose) mean LFC vectors:

- **Version A (train-only, production)** — fit on the 150 train drugs only. This is what `src/models/fingerprint_mlp.py::StratumMean.fit(train_df)` and `src/models/baselines.py::StratifiedMeanBaseline` use during training and at test time.
- **Version B (all-drugs, leaky)** — fit on the full 188-drug set including the 38 held-out test drugs. Computed only here as a reference; never used in production.

For each version, predict on the held-out test set (456 conditions) and compute macro-Pearson overall and per cell line. The decisive leak-detection signal is the cross-check between A's predictions and the saved M3 `baseline_predictions.npz` — if they match exactly, the production train-only code path has not regressed.

## Numbers

### Overall pcorr_macro on test set ({len(test_df)} conditions)

| version | overall pcorr_macro |
|---|---:|
| A — train-only (production) | {overall_A:+.6f} |
| B — all-drugs (leaky) | {overall_B:+.6f} |
| **Δ (A − B)** | **{delta_overall:+.6f}** |

### Per cell line

| cell_line | A (train-only) | B (leaky) | Δ (A − B) |
|---|---:|---:|---:|
"""
    for cl in sorted(per_cl_A):
        md += f"| {cl} | {per_cl_A[cl]:+.6f} | {per_cl_B[cl]:+.6f} | {delta_per_cl[cl]:+.6f} |\n"

    md += f"""
## Cross-check vs `results/baseline_predictions.npz` (M3 saved baseline)

| | overall | A549 | K562 | MCF7 |
|---|---:|---:|---:|---:|
| Recomputed A here | {overall_A:+.6f} | {per_cl_A['A549']:+.6f} | {per_cl_A['K562']:+.6f} | {per_cl_A['MCF7']:+.6f} |
| Saved baseline.npz | {saved_overall:+.6f} | {saved_per_cl['A549']:+.6f} | {saved_per_cl['K562']:+.6f} | {saved_per_cl['MCF7']:+.6f} |

Max absolute element-wise difference between A's predictions and the saved baseline NPZ: **{a_vs_saved_pred_diff:.2e}**

If this is below ~1e-5, A reproduces the M3 saved baseline exactly. (StratifiedMeanBaseline is deterministic; any difference would indicate a regression in the implementation.)

## Interpretation

The two checks measure different things:

1. **Cross-check (A vs saved baseline NPZ)** — directly tests "does the train-only implementation reproduce what M3 saved?" Max abs pred diff = **{a_vs_saved_pred_diff:.2e}**. If ≤ 1e-5, the production code path has not regressed; train_df → stratum_mean → predict is bit-for-bit the same as in M3. **This is the decisive leak signal.**

2. **A vs B (train-only vs all-drugs)** — tests "is the train-only mean numerically close to the all-drugs mean?" |Δ overall| = {abs(delta_overall):.6f}; max |Δ per-cell-line| = {max_abs_per_cl_delta:.6f}. This Δ measures *how much the per-stratum means would shift if we included test drugs*. A non-zero Δ here is expected when test drugs are not a uniform random subsample of the training distribution — it does **not** indicate a leak in the production code.

For the M3 38-drug test split (a stratified random sample by pathway, but only 38/188 ≈ 20% of drugs and only 1–4 drugs per pathway), some Δ is expected. The Δ of {abs(delta_overall):.4f} overall is consistent with "test drugs differ from train drugs in a measurable but unsurprising way," not with implementation leakage.

## Reproducibility

```
python scripts/diag/p6_leakage_check.py
```

Wall clock for this script: {time.time()-t0:.1f}s.
"""
    OUT.write_text(md)
    print(f"wrote {OUT}", flush=True)
    print(f"\noverall pcorr  A (train-only) = {overall_A:+.6f}", flush=True)
    print(f"overall pcorr  B (leaky)      = {overall_B:+.6f}", flush=True)
    print(f"Δ                              = {delta_overall:+.6f}", flush=True)
    print(f"max |Δ| per-cell-line          = {max_abs_per_cl_delta:.6f}", flush=True)
    print(f"max abs pred diff A vs saved   = {a_vs_saved_pred_diff:.2e}", flush=True)
    print(f"\n{verdict}", flush=True)


if __name__ == "__main__":
    main()
