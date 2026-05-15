# Milestone 4A — diagnostic probe SUMMARY

**Date:** 2026-05-15
**Status:** All 5 probes (P1–P5) ran without runtime issues. One non-runtime surprise worth flagging (P3 design clarification). Diagnosis is unambiguous.

---

## Pre-flight (recorded for future reference)

| Item | Value |
|---|---|
| M3 best checkpoint | `runs/cellcast_v0/checkpoints/best-7-816-pcorr=0.1282.ckpt` (2,526,771,651 B) |
| Best ckpt SHA256 | `2ccb407a889293d133364d52d79293b97e842d139d12c210ea8943fd554a34cd` (best == last, byte-identical) |
| Test parquet | `data/sciplex/processed/cellcast_v0.parquet` (59,860,886 B, 2,256 rows = 752 × 3 cell lines) |
| Parquet SHA256 | `404d35e07a0baaa1d1b730a413a81846b845d717c064246aa7ce0e09f48a5b46` |
| Tests | 33 passed in 15.46s |

Note: M3 artifacts did not record SHAs; the values above are now the canonical baseline.

---

## P1 — Encoder output sensitivity to SMILES

**Setup:** 5 chemically-diverse held-out test drugs (Lomustine, Quercetin, Dasatinib, Bisindolylmaleimide IX, 2-Methoxyestradiol) × 3 cell lines × dose=1000nM. For each forward, mean-pool the encoder last_hidden_state over four spans (full / SMILES tokens / MASK position / gene tokens) and compute the 5×5 cosine-distance matrix.

**Mean off-diagonal cosine distance:**

| span | K562 | A549 | MCF7 |
|---|---:|---:|---:|
| full (all valid) | 0.0006 | 0.0003 | 0.0003 |
| **SMILES tokens** | **0.0581** | **0.0521** | **0.0538** |
| **MASK position** | **0.0127** | **0.0093** | **0.0206** |
| gene tokens | 0.0005 | 0.0003 | 0.0003 |

**SMILES → MASK attenuation ratio: 4.6× (K562), 5.6× (A549), 2.6× (MCF7).**

The encoder DOES distinguish drugs at SMILES token positions (cosdist ~0.05 — meaningful for high-D embeddings — much higher than the dose- and drug-invariant gene/full background of ~0.0003). But that signal is attenuated by 3–6× by the time it reaches the MASK position where the head reads. Gene span is essentially flat across drugs (sanity check: input gene rank list is control-derived and doesn't depend on drug).

→ Files: `p1/p1_summary.json`, `p1/p1_distances.npz`, `p1/p1_cosdist_{cell}_{span}.png` (12 heatmaps).

---

## P2 — Head sensitivity to its input

**Setup:** Take the most-distant MASK pair from P1 (Lomustine ↔ Dasatinib in K562; cosdist 0.0247). Linearly interpolate 100 synthetic MASK vectors between them, run each through the head. Also compute the head's Jacobian at one anchor.

**Smoothness check:** head output L2 distance from the midpoint varies smoothly with the interpolation parameter (max 0.0708, ≈ ½ of endpoint-to-endpoint distance 0.1384) — a clean linear-ish ramp, not a step or plateau. The head is NOT in a saturated/dead regime.

**Contraction:**

| quantity | value |
|---|---:|
| ||head(a) − head(b)||₂ | 0.1384 |
| ||head(midpoint)||₂ | 2.0881 |
| Jacobian Frobenius norm | 1.551 |
| Jacobian operator norm (largest σ) | **0.831** |
| ||head(a) − head(b)|| / ||a − b|| (effective gain in drug-direction) | **0.054** |

The head is **highly contractive in the drug direction**: a unit input perturbation produces only ~5% output response. Operator norm < 1 means contraction in every direction. This is consistent with a head that learned during training that the MASK input doesn't carry actionable drug-specific signal, so it tightened down its input→output gain.

→ Files: `p2/p2_summary.json`, `p2/p2_traces.npz`, `p2/p2_interp_and_jacobian.png`.

---

## P3 — Dose token influence

**Setup:** (Dasatinib, K562) at all 4 doses (10/100/1000/10000 nM). MASK cosine distance + prediction L2 distance, real conditions vs dose-token-swap (anchor=1000nM, cycle the dose token).

**Important design clarification (worth noting in writing):** in the CellCast pipeline, the input gene rank list is derived from each cell-line's *control* pseudobulk and is therefore **dose-invariant by construction** (verified: all 4 dose conditions for any (drug, cell-line) share the same gene rank list). So in our pipeline, the dose token is the *only* dose-conditional input — REAL and SWAP setups happen to be the same experiment. Numbers are bit-identical between the two:

| | MASK cosdist mean | MASK cosdist max | pred L2 mean | pred L2 max |
|---|---:|---:|---:|---:|
| REAL (4 doses, real labels)   | 0.0013 | 0.0025 | 0.0442 | 0.0442 |
| SWAP (anchor + 4 token swaps) | 0.0013 | 0.0025 | 0.0442 | 0.0753 |

(SWAP max is slightly higher because it includes the anchor compared to itself with a different token.)

**Comparison with drug swap (P1):** dose-token swap moves MASK cosdist by ~0.0013 mean. Drug swap (P1) moves MASK cosdist by 0.0093–0.0206 mean. **Dose token has roughly 1/7th–1/15th of the MASK-level effect of a drug swap.**

The dose token isn't completely ignored — predictions do change by L2 ~0.044 (~2% of typical output norm 2.0). But against a true LFC-vs-dose change of L2 ~12.4 between min-dose and max-dose for the same drug, this 0.044 is essentially nothing.

→ Files: `p3/p3_summary.json`, `p3/p3_arrays.npz`, `p3/p3_real_mask_cosdist.png`, `p3/p3_swap_mask_cosdist.png`, `p3/p3_real_pred_l2.png`, `p3/p3_swap_pred_l2.png`.

---

## P4 — SMILES ablation

**Setup:** 50 random held-out test conditions (seed=1234). For each, compare prediction with the real SMILES vs with SMILES replaced by `"CCO"` (ethanol, the same dummy for all 50). Same dose token, same gene rank list.

**Result:**

| metric | real SMILES | dummy "CCO" | Δ |
|---|---:|---:|---:|
| pcorr_macro | +0.1323 | +0.1319 | **+0.0004** |
| spearcorr_macro | +0.1196 | +0.1195 | +0.0001 |
| top50_dir_acc | +0.6860 | +0.6872 | **−0.0012** |
| mse | +0.0058 | +0.0058 | +0.0000 |
| per-sample pred-diff L2 mean | — | — | 0.172 (~9.9% of output norm) |

The model produces **the same quality of predictions** with the real SMILES vs ethanol. Per-sample predictions DO move by ~10% in L2 — the SMILES isn't a literal no-op — but the differences don't correlate with the truth either way. **The model has effectively learned that the SMILES input is uninformative.**

→ Files: `p4/p4_summary.json`, `p4/p4_preds.npz`, `p4/p4_ablation.png`.

---

## P5 — Baseline-residual ceiling

**Setup:** For all 456 test conditions, compute residual = true_LFC − baseline_prediction. Report variance(residual) / variance(true_LFC) three ways.

| ratio | value | interpretation |
|---|---:|---|
| flat (all (condition, gene) entries pooled) | **0.916** | baseline captures only **8.4%** of total variance |
| per-condition (var across genes, mean across conditions) | 0.995 | baseline barely changes within-condition gene-axis spread |
| per-gene (var across conditions, mean across genes) | 0.961 | baseline barely changes within-gene cross-condition spread |

The flat ratio is the right "ceiling" number per the prompt's threshold framing. **91.6% well above the 30% "substantial signal" threshold.** There is enormous drug-specific signal in the test set that any properly drug-aware model could in principle capture.

Per-cell-line: A549 0.915, K562 0.920, MCF7 0.909 — uniform across cell lines, so the headroom isn't just an MCF7 artifact.

Per-dose: 10nM 0.942 → 10000nM 0.892 — slight gradient (higher dose conditions have somewhat more variance captured by per-stratum mean), but nowhere close to "ceiling reached."

**Reconciling with M3:** the baseline beats CellCast on pcorr (0.178 vs 0.127) not because the baseline captures most of the signal — it captures only 8% — but because CellCast captures even less in the *correct direction*. CellCast's predictions move around (it's not a constant) but they move in directions that don't correlate with the truth as well as a per-stratum mean does.

→ Files: `p5/p5_summary.json`, `p5/p5_residual.npz`, `p5/p5_variance_scatter.png`.

---

## What the data says

### Diagnosis: L2 (dominant) + L3 (compounding) — not L1

| Location | Verdict | Evidence |
|---|---|---|
| **L1 — encoder collapses drug at SMILES** | **Refuted.** | P1: SMILES-position cosdist 0.05 across drugs, ~150× higher than the gene/full background. The encoder is producing drug-distinguishing representations at SMILES token positions. |
| **L2 — drug signal at SMILES doesn't propagate to MASK** | **Strongly supported.** | P1: MASK cosdist drops to 0.01–0.02 from SMILES cosdist 0.05 — a 3–6× attenuation in a single layer. The MASK-position hidden state (where the head reads) is dominated by the gene span (1294 of 1310 valid tokens) rather than the SMILES span (8–26 tokens). |
| **L3 — head ignores MASK variation that does arrive** | **Contributes.** | P2: head is smooth (not saturated/dead) but highly contractive — operator norm 0.83 < 1, gain in drug-direction only 0.054. Combined with P1's 4–6× SMILES→MASK attenuation, the compounded SMILES→prediction gain is ~1% — explains why P4's SMILES ablation shows essentially zero effect on metrics. |

### Side-finding: dose token is barely engaged either

The dose mechanism has the same shape of failure as the drug mechanism: it produces *some* perturbation in MASK (cosdist 0.0013) but is so much smaller than the drug-axis perturbation (0.01–0.02) that it can't contribute meaningfully to dose-discriminative predictions, even though true LFCs vary substantially across doses (L2 ~12 between extreme doses for the same drug).

### The headroom is large

**91.6% of LFC variance is unaccounted for by the per-stratum baseline.** This is not a "the data is too hard / there's no signal" situation. The signal exists in the data; the model just isn't using it.

---

## Milestone 4B recommendation

Given the diagnosis (L2 dominant with L3 compounding, large headroom), the prioritization in the post-3D addendum needs adjustment. I'd recommend M4B do **two things in parallel**, not the addendum's "diagnostic first then reframe":

### Recommended primary lever: residual-to-baseline reframe + LoRA on encoder (combined)

1. **Reframe the target** as residual-to-stratum-mean: `target_residual = true_LFC − stratum_mean_LFC`. The model now *cannot* win by predicting the per-stratum mean (the ground truth has it removed); it has to capture drug-specific deviation. This puts the entire loss-gradient on the drug-specific component, eliminating the "easy mode" the model fell into during M3.
2. **LoRA on the MAMMAL encoder** (rank ~16–32 on attention QKV + FFN projections in the encoder layers, ~1–3M added trainable params). The frozen encoder is the proximate cause of L2: gradient from the head can't reshape how the encoder routes drug information to the MASK position because the encoder weights are frozen. Even with the residual reframe, a frozen encoder may still attenuate drug signal exactly as it does today. LoRA is the lightest-touch way to give the encoder degrees of freedom.

These two should be applied together because:
- Reframe alone (frozen encoder + new target) is unlikely to help — the bottleneck is L2 which is encoder-internal, and gradient via the head + residual target still has to propagate the same broken pipe to influence the encoder's behavior. With everything frozen, only the head adapts, and the head can't undo encoder routing.
- LoRA alone (full LFC target, encoder unfrozen) may still let the model learn the per-stratum mean as an attractor — the residual reframe specifically penalizes that.
- Together they address both the architectural bottleneck and the gradient-dynamics issue.

### Secondary: dose mechanism revisit

Independently of the main fix, investigate why the dose token barely registers (P3). Two candidates worth a small experiment each:
- The dose token sits between the SMILES and the gene block — possibly the MASK position attends mostly to the gene block (1294 tokens) and largely ignores the few SMILES + dose tokens. A *prompt-position* experiment (move the dose token to position 1, right after MASK) would test this cheaply.
- Alternatively, encode dose as a continuous scalar literal in MASK's adjacent position (not as a learned embedding) — the way MAMMAL was pretrained to handle dose-like quantitative inputs.

### Deferred

- **Per-cell training** (M2 → M3 → M4 deferred): bottleneck is signal usage, not signal volume. Deferred per the post-3D addendum.
- **Better baselines (ChemCPA, fingerprint MLP)**: useful framing context but not a fix; can run in parallel with M4B.

### What NOT to do

- Don't tune hyperparameters (lr, warmup, epochs, dropout) on the M3 architecture. P4 proves the SMILES is being treated as noise; no amount of optimizer tuning will fix that.
- Don't make the head wider/deeper. P2 shows the head is already smooth and uses its input — the problem isn't head capacity, it's input-signal quality.
- Don't lengthen training. P5 shows huge headroom but the training trajectory was healthy and converged; more steps on the same architecture will continue to converge to the same place.

---

## Status

- All 5 probes completed cleanly. Total wall-clock: ~50s of GPU work + plotting.
- Stopping here per the prompt's "do not start 4B" constraint.
- All artifacts under `results/4a_diagnostics/` (per-probe subdirectory + this SUMMARY).
- Probe scripts under `scripts/diag/` are reusable for follow-up: e.g., re-running P4 with a chemically distinct dummy (rather than ethanol) would test whether the model's no-effect under ablation is specifically about ethanol or general SMILES insensitivity.
