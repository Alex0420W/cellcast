# P6 leakage check — stratum mean implementation

**Date:** 2026-05-15
**Status:** PASS — implementation is leak-free (A reproduces M3 saved baseline exactly: max abs pred diff = 0.00e+00). The A-vs-B delta of 0.0240 pcorr just measures how much the test drugs shift the per-stratum means when included; it does NOT indicate a leak in the production code path.

## Method

Two versions of the per-(cell_line, dose) mean LFC vectors:

- **Version A (train-only, production)** — fit on the 150 train drugs only. This is what `src/models/fingerprint_mlp.py::StratumMean.fit(train_df)` and `src/models/baselines.py::StratifiedMeanBaseline` use during training and at test time.
- **Version B (all-drugs, leaky)** — fit on the full 188-drug set including the 38 held-out test drugs. Computed only here as a reference; never used in production.

For each version, predict on the held-out test set (456 conditions) and compute macro-Pearson overall and per cell line. The decisive leak-detection signal is the cross-check between A's predictions and the saved M3 `baseline_predictions.npz` — if they match exactly, the production train-only code path has not regressed.

## Numbers

### Overall pcorr_macro on test set (456 conditions)

| version | overall pcorr_macro |
|---|---:|
| A — train-only (production) | +0.178383 |
| B — all-drugs (leaky) | +0.202391 |
| **Δ (A − B)** | **-0.024008** |

### Per cell line

| cell_line | A (train-only) | B (leaky) | Δ (A − B) |
|---|---:|---:|---:|
| A549 | +0.048816 | +0.088680 | -0.039864 |
| K562 | +0.047834 | +0.088767 | -0.040933 |
| MCF7 | +0.088264 | +0.115543 | -0.027279 |

## Cross-check vs `results/baseline_predictions.npz` (M3 saved baseline)

| | overall | A549 | K562 | MCF7 |
|---|---:|---:|---:|---:|
| Recomputed A here | +0.178383 | +0.048816 | +0.047834 | +0.088264 |
| Saved baseline.npz | +0.178383 | +0.048816 | +0.047834 | +0.088264 |

Max absolute element-wise difference between A's predictions and the saved baseline NPZ: **0.00e+00**

If this is below ~1e-5, A reproduces the M3 saved baseline exactly. (StratifiedMeanBaseline is deterministic; any difference would indicate a regression in the implementation.)

## Interpretation

The two checks measure different things:

1. **Cross-check (A vs saved baseline NPZ)** — directly tests "does the train-only implementation reproduce what M3 saved?" Max abs pred diff = **0.00e+00**. If ≤ 1e-5, the production code path has not regressed; train_df → stratum_mean → predict is bit-for-bit the same as in M3. **This is the decisive leak signal.**

2. **A vs B (train-only vs all-drugs)** — tests "is the train-only mean numerically close to the all-drugs mean?" |Δ overall| = 0.024008; max |Δ per-cell-line| = 0.040933. This Δ measures *how much the per-stratum means would shift if we included test drugs*. A non-zero Δ here is expected when test drugs are not a uniform random subsample of the training distribution — it does **not** indicate a leak in the production code.

For the M3 38-drug test split (a stratified random sample by pathway, but only 38/188 ≈ 20% of drugs and only 1–4 drugs per pathway), some Δ is expected. The Δ of 0.0240 overall is consistent with "test drugs differ from train drugs in a measurable but unsurprising way," not with implementation leakage.

## Reproducibility

```
python scripts/diag/p6_leakage_check.py
```

Wall clock for this script: 0.4s.
