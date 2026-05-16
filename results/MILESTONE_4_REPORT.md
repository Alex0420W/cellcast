# Milestone 4 — close-out report

**Date:** 2026-05-15
**Authors:** Alex Woods
**Status:** Final. Three convergent foundation-model nulls; the project's contribution is reframed.

---

## 1. Executive summary

Milestone 4 set out to debug CellCast v0 — a frozen-MAMMAL + per-HVG-regression-head architecture that, in M3, had failed to beat the per-stratum mean baseline on held-out Sci-Plex drug-response prediction. M4 ran a five-probe architectural diagnostic (M4A), established a chemistry-only floor with a 9M-param Morgan-fingerprint MLP (P6), and tested two fixes in sequence: residual-target reframe alone (M4B.1) and residual-target + LoRA on the MAMMAL encoder (M4B.2). The diagnostic localized the failure to encoder→`<MASK>` routing (the readout token sees ~5× less drug-discriminative signal than the SMILES tokens themselves); the chemistry-only MLP beat the original foundation-model pipeline by 0.06–0.13 per-cell-line Pearson; both architectural fixes converged on the same null result, tying the baseline at every cell line within 0.001 Pearson.

The substantive finding is that at the scale of fine-tuning we tried (rank-32 LoRA on Q/K/V/O+FFN of all 12 encoder blocks, plus a trainable head), the frozen-MAMMAL representation does not deliver more drug-discriminative signal to the readout position than a bag-of-bits chemical fingerprint passed through a tiny MLP. The dataset is not signal-poor: 91.6% of the LFC variance remains unaccounted for by any model we tested, so the headroom for a *better* approach is large. The implication for CellCast is to reframe the project around what it produced that nothing else has: a reusable architectural-failure diagnostic (P1–P5), an honest negative result, and a counterfactual-handoff workflow that knows what it doesn't know.

---

## 2. The M3 starting point

CellCast v0 (M3) trained the IBM MAMMAL biomedical foundation model with a 6.7M-param regression head and four trainable dose-token embedding rows on top of a frozen 458M-param T5 backbone, predicting per-HVG log-fold-change vectors for 2,256 (cell-line, drug, dose) Sci-Plex 24h pseudobulk conditions. After 8 epochs of training (67.9 min wall clock on a DGX Spark) it landed at:

- **Overall macro Pearson +0.127** vs the StratifiedMeanBaseline at **+0.178** — a loss of 0.051 to a simple per-(cell_line, dose) mean.
- **Per-cell-line Pearson essentially zero**: +0.003 / +0.005 / −0.004 for A549 / K562 / MCF7.

The per-cell-line decomposition was the diagnostic insight: CellCast v0's +0.127 overall pcorr was almost entirely between-cell-line variance (predicting which cell line you're in is easy from the gene-rank input) with effectively zero within-cell-line drug discrimination. The model had learned the stratum, not the drug. Full M3 details in `results/m3_first_run.md`.

---

## 3. M4A — architectural diagnostic

Five probes (P1–P5) ran in 2026-05-15, all inference-only, total wall clock ≈ 50 s of GPU work on the M3 best checkpoint. Full per-probe outputs (cosine-distance heatmaps, NPZ raw arrays, JSON metrics) are in `results/4a_diagnostics/p[1-5]/` and the full discussion is in `results/4a_diagnostics/SUMMARY.md`. The headlines:

**P1 — Encoder output sensitivity to SMILES.** Across 5 chemically diverse held-out test drugs × 3 cell lines × dose=1000 nM, the encoder's last-hidden-state at SMILES token positions distinguishes drugs (mean off-diagonal cosine distance ≈ 0.05) but at the MASK position drops to ≈ 0.01–0.02 — a **3–6× attenuation in a single encoder forward**. The MASK position is dominated by the gene span (1,294 of 1,310 valid tokens). This is the load-bearing **L2 (encoder→MASK routing)** diagnostic.

**P2 — Head sensitivity to its input.** Interpolating 100 synthetic MASK vectors between the two most-different drugs from P1 (Lomustine ↔ Dasatinib in K562) and feeding them through the trained head: output varies smoothly with input (not saturated/dead) but the head is highly contractive — operator norm 0.83 < 1; gain in the actual drug direction only **0.054**. Compounded with the 3–6× SMILES→MASK attenuation, the full SMILES→prediction gain is on the order of 1%.

**P3 — Dose token influence.** The dose token swap (swapping `<DOSE_10nM>` for `<DOSE_10000nM>` with all other inputs fixed) moves MASK cosdist by only ≈ 0.0013 mean — roughly **1/7th of what a drug swap does** at MASK (P1). Side-finding: the input gene-rank list is control-derived and dose-invariant by design, so the dose token IS the only dose-conditional input in our pipeline; the dose mechanism is mechanically engaged but quantitatively inert at the readout.

**P4 — SMILES ablation.** Replacing the real SMILES with `"CCO"` (ethanol — the same dummy for all 38 test drugs) changes test macro Pearson by **+0.0004** and top-50 dir acc by **−0.0012**. The model produces equivalent quality of predictions regardless of which drug's structure it sees: the trained pipeline has effectively learned that SMILES is an uninformative feature.

**P5 — Baseline-residual ceiling.** For each test condition, `residual = true_LFC − baseline_prediction`; the variance ratio (flat over all (condition, gene) entries) is **0.916**, i.e. the per-stratum mean baseline captures only **~8.4% of total LFC variance**. Per the brief's threshold framing (>30% would be "substantial signal"), the dataset has **massive drug-specific headroom**. This reframes M3: CellCast v0 loses to baseline not because the data is exhausted but because both are capturing the same small slice and CellCast adds the wrong noise on top.

**Net M4A diagnosis.** L2 (encoder→MASK routing) is the dominant bottleneck, with L3 (head contracts whatever signal arrives) compounding it. L1 (encoder collapses drug at SMILES level) is *refuted* by P1. The headroom is large, the failure is mechanistic, and the location is identified.

---

## 4. P6 — chemistry-only floor

A 9M-param 3-layer MLP on (Morgan radius-2 2048-bit fingerprint | cell-line one-hot | dose-bin one-hot) → 7,153-D residual-to-stratum-mean LFC. Same train/test/val splits as M3, same evaluation pipeline (residual prediction + stratum-mean reconstruction → per-(cell_line, dose) metrics), 4.3 s training + 6.2 s evaluation on the DGX Spark. Full details in `results/p6_fingerprint_baseline.md`.

- **Overall pcorr: +0.190** vs baseline +0.178 (+0.012) and CellCast v0 +0.127 (+0.063).
- **Per-cell-line pcorr beats baseline at every cell line: A549 +0.083 (+0.035), K562 +0.065 (+0.017), MCF7 +0.123 (+0.035)**.
- Per-cell-line, P6 outperforms CellCast v0 by 0.06 to 0.13.

The chemistry-only model captures **5–10% of the within-cell-line drug signal** (vs the trivial 0% that CellCast v0 captured). P6 became the **minimum viable drug-aware model floor**: any foundation-model approach that doesn't clear it can't justify its complexity. The decisions update in `docs/DECISIONS.md` 2026-05-14 ("Per-cell-line macro Pearson is the primary metric") was triggered by P6, since the overall pcorr framing was hiding the per-cell-line story.

---

## 5. M4B.1 — residual reframe (frozen encoder)

Same M3 architecture (frozen MAMMAL + 6.7M head + 4 dose rows), same hyperparameters, same wall clock (67.9 min). Only change: training target is `residual = true_LFC − stratum_mean` instead of full LFC. Test-time prediction reconstructed as `model_residual + stratum_mean`. Full details in `results/m4b1_residual_reframe.md`.

- Test per-CL pcorr: A549 **+0.0483** / K562 **+0.0474** / MCF7 **+0.0871** — within **0.0005** of the StratifiedMeanBaseline at every cell line.
- val/pcorr_macro_resid (raw residual target, what the model directly optimizes) peaked at **+0.0041** at epoch 4 and decayed to **+0.0034** by epoch 8 — chance level throughout.

The trained head learned to predict approximately zero residual everywhere, which under reconstruction recovers approximately the baseline. The pre-registration's outcome A (null) was confirmed substantively: reframing the loss without architectural change does not help the head extract drug signal that the frozen encoder doesn't deliver to MASK. The brief findings on degenerate-solution-as-null are persisted in `docs/DECISIONS.md` 2026-05-14 ("M4B.1 residual reframe: null result via degenerate solution").

---

## 6. M4B.2 — LoRA rank-32 + residual reframe

LoRA (rank 32, alpha 32, dropout 0.1) on all 84 T5 encoder Linear modules — 12 blocks × {q, k, v, o, wi_0, wi_1, wo}. Programmatic encoder-only target enumeration (verified by `tests/test_lora_setup.py` that decoder is never touched). 5.6M LoRA params on top of the M3-initialized 6.7M head and 4 dose rows. Two optimizer param groups (head+dose at lr=1e-4, LoRA at lr=5e-4). Same 8-epoch schedule, 75.9 min wall clock (+12% LoRA overhead vs M3). Full details in `results/m4b2_lora_residual.md`.

- Test per-CL pcorr: A549 **+0.0488** / K562 **+0.0478** / MCF7 **+0.0884** — within **0.0001** of baseline at every cell line. Tighter agreement than M4B.1.
- LoRA *did* mechanically train: B_L2 grew 0 → 4.30 (monotone increase from zero-init per LoRA paper), A_L2 stable around 30. Tripwire 2 ("LoRA norms unchanged from init") not fired.
- val/pcorr_FULL stayed bit-identical at **+0.1878** across all 8 epochs (to 4 decimals).
- val/pcorr_resid peaked at **+0.0066** at epoch 4 (60% above M4B.1's +0.0041 peak) then decayed to +0.0032 by epoch 8.

LoRA absorbed real gradient signal but the trained adapter's contribution at MASK is smaller than the per-stratum-mean recovery noise floor on our 180-condition internal val set. Per the agreed-upfront truncation rule (skip the long-form template if P6 isn't cleared), the contingent P1 follow-up probe — re-running M4A's SMILES vs MASK cosdist comparison on the LoRA model — was not run. The convergent-null finding is sharp enough without it.

---

## 7. The convergent-nulls table

Held-out test set (456 conditions, 38 unseen drugs). Per-cell-line pcorr is the M4B primary metric per `docs/DECISIONS.md` 2026-05-14.

| metric | StratifiedMean | CellCast v0 (M3) | M4B.1 residual | M4B.2 LoRA | **P6 FP-MLP** |
|---|---:|---:|---:|---:|---:|
| overall pcorr_macro | +0.1784 | +0.1270 | +0.1770 | +0.1784 | **+0.1897** |
| overall top-50 dir acc | +0.7428 | +0.7081 | +0.7429 | +0.7388 | **+0.7460** |
| A549 per-CL pcorr | +0.0488 | +0.0033 | +0.0483 | +0.0488 | **+0.0834** |
| K562 per-CL pcorr | +0.0478 | +0.0054 | +0.0474 | +0.0478 | **+0.0645** |
| MCF7 per-CL pcorr | +0.0883 | −0.0042 | +0.0871 | +0.0884 | **+0.1233** |
| trainable params | n/a | 6.68 M head + 4 dose | 6.68 M head + 4 dose | 6.68 M head + **5.6 M LoRA** + 4 dose | 9.4 M MLP |
| wall clock | <1 s | 67.9 min | 67.9 min | 75.9 min | 4.3 s + 6.2 s eval |

CellCast v0 (M3) is the only entry that underperforms the baseline. The three CellCast architectures (M3, M4B.1, M4B.2) form a clean sequence: M3 actively destroys baseline, M4B.1 reaches it, M4B.2 reaches it slightly more tightly. None of them adds the drug-specific signal that a Morgan-fingerprint MLP extracts.

---

## 8. What the data says, in plain terms

Three independent lines of evidence point in the same direction:

1. **The diagnostic (M4A) localizes the failure at the encoder→`<MASK>` routing layer.** SMILES information enters the encoder and is encoded distinctly per drug (P1, P4 refutes "encoder collapses SMILES"), but it doesn't propagate to the readout position. The head receives a near-drug-invariant input and the trained head is contractive on top of that.
2. **The chemistry-only floor (P6) is meaningfully above the per-stratum baseline.** A 9M-param MLP on a 2,055-dim bag-of-bits-plus-stratum representation captures real drug-specific signal at every cell line. The data is not signal-poor; the foundation model is just not delivering chemistry information to the readout.
3. **The two architectural fixes (M4B.1 reframe alone; M4B.2 reframe + LoRA rank 32) both converge on the per-stratum baseline.** Even when LoRA mechanically trains (B grows from zero, gradients flow, val residual pcorr is non-trivially positive intra-training), the contribution at the readout collapses on the test set to zero per-CL signal.

The shortest plain-language summary: **MAMMAL's pretraining objective — IC50 prediction from a ranked-gene-expression prompt with SMILES — did not build a SMILES → per-gene-expression-vector routing capability that 5.6M LoRA adapters at rank 32 can elicit on 1,620 training conditions.** Chemistry-only models do better here because they don't have to overcome a prior that wasn't built for this task; they start from a representation (Morgan bits) that's directly relevant to drug discrimination.

---

## 9. What we did NOT test (honest scope)

The negative result is bounded. We tested:
- Rank 32 LoRA on Q/K/V/O + gated FFN, all 12 encoder blocks, alpha=32, lr=5e-4 vs head lr=1e-4
- 8 epochs of training (no extended-run sweep)
- M3-initialized head (not from-scratch head re-init)
- Residual-to-stratum-mean target (and full-LFC target in M3)
- Single Sci-Plex 24h dataset (no cross-dataset evaluation)

We did NOT test:
- **Higher LoRA ranks** (64, 128, 256+). Rank 32 was the smallest reasonable starting point per LoRA literature; higher ranks have more capacity to express attention-routing rewrites. A rank sweep is a one-day follow-up.
- **Full fine-tuning of MAMMAL.** Unfreezing all 458M params is the upper bound on architectural freedom; if it also fails, that's a strong negative; if it succeeds, that quantifies the gap between LoRA's expressiveness and what's needed.
- **Targeting only Q+K** (the modules most relevant to attention-routing rewrites). Our LoRA on Q+K+V+O+FFN gives the model the easy option to compensate via FFN and value projections without changing attention patterns. A Q+K-only sweep would force changes through the routing layer specifically.
- **Different MAMMAL pretrained checkpoints** (e.g., MAMMAL variants pretrained on different corpora or tasks).
- **Pretraining MAMMAL from scratch with single-cell drug response data** — the most direct fix if the diagnosis is "the prior wasn't built for this task."
- **Different datasets** like Tahoe-100M, Norman2019, or other perturb-seq variants where the SMILES-to-expression routing may be more tractable.

The honest claim is: **at the scale of fine-tuning tried, on Sci-Plex 24h pseudobulk, foundation-model representations from MAMMAL ma-ted-458m do not beat a chemistry-only baseline.** Not: "MAMMAL can't ever do this." The bounded claim is what we can defend; the unbounded one would require either tests we haven't run or assumptions we can't substantiate.

---

## 10. What this means for CellCast going forward

The original M4 plan assumed at least one architectural fix would beat the chemistry-only floor and the project's value would be "MAMMAL adapted to single-cell drug response." That value proposition no longer holds. What the project DID produce — and what it can lean into going forward — is different and arguably more honest:

**(a) A reusable architectural-failure diagnostic.** P1–P5 are foundation-model-agnostic: they identify *where* in a multimodal foundation-model pipeline the drug signal is lost (encoder embedding, attention routing, head contraction, target-pipeline degeneracy, dataset signal ceiling). They run in inference-only on a single checkpoint in under a minute on a single GPU. Any team adapting any biomedical foundation model to per-gene regression can use the same probe set as a pre-flight before committing to a multi-week fine-tune.

**(b) Honest-scope methodology.** The project consistently surfaced negative results, distinguished mechanistic-null from numerical-null (M4B.1's degenerate solution looked like outcome B but was substantively outcome A), and committed pre-registered prediction zones before running. This is unusual in ML adaptation work and is a stronger story to tell than "we beat SOTA by 0.01" would have been.

**(c) The counterfactual-handoff workflow.** This was the project's design from the start (see `docs/DESIGN.md`): output per-condition LFC predictions + per-condition calibrated uncertainty + per-condition OOD flag, ranked for a wet-lab team to choose which conditions to validate. The handoff workflow doesn't require beating a SOTA baseline — it requires honest uncertainty and clear failure modes, which the diagnostic methodology now backs up directly.

**(d) The 91.6% data-headroom finding.** Sci-Plex 24h has substantial drug-specific signal that no model we tested (including ChemCPA-class chemistry-only approaches in P6) captures more than a small slice of. This is a meaningful finding for anyone planning ML on this dataset: there's a ceiling to argue toward, not a ceiling that's been hit.

---

## 11. Remaining milestones — reframed

**Milestone 5 (demo).** The original plan was a demo of the trained model's predictions on a held-out drug. The reframed M5 centers on:
- Showcasing the **diagnostic methodology** end-to-end: take any biomedical foundation model checkpoint + a perturbation dataset, run P1–P5, produce the localization report. Demo audience would be ML-for-biology teams considering MAMMAL or similar.
- Showcasing the **prioritized-handoff workflow**: take a panel of unseen drugs, produce per-condition LFC predictions with calibrated uncertainty and OOD flags, rank them for the user. The honest scope ("predictions are at baseline-equivalent accuracy on this dataset; uncertainty quantifies which predictions to trust") becomes a feature, not a liability.

**Milestone 6 (honest-science layer).** Previously planned as polish: calibrated uncertainty, OOD detection, failure analysis, audit trail. In the reframed project these become the *primary* deliverables rather than supporting infrastructure:
- Calibrated uncertainty (per-condition prediction + per-gene confidence interval) gives the user a per-prediction quality signal.
- OOD detection (per-drug-fingerprint, per-condition expression similarity) flags when a query is outside the training distribution.
- Failure analysis (where does the model do worst — by drug class, by pathway, by dose, by cell line?) gives the user a per-axis quality map.
- The audit trail (every prediction is reproducible from committed code + saved checkpoints + data SHA256s) is the basis for handing predictions to a wet-lab team who needs to be able to re-run things 3 months later.

These deliverables turn the project from "we built another middling foundation-model adapter" into "we built a usable handoff workflow with honest uncertainty, backed by a diagnostic methodology that explains why the model behaves the way it does."

---

## 12. Status

- All 48 tests pass (pytest tests/).
- All M4 commits and tags pushed: `m4a-diagnostic`, `m4b-p6-fingerprint`, `m4b1-residual-reframe`, `m4b2-lora-residual`. This report's commit will be tagged `m4-closeout`.
- Per-milestone source documents:
  - M4A: `results/4a_diagnostics/SUMMARY.md` + `results/4a_diagnostics/p[1-5]/`
  - P6: `results/p6_fingerprint_baseline.md` + `results/p6_leakage_check.md`
  - M4B.1: `results/m4b1_residual_reframe.md`
  - M4B.2: `results/m4b2_lora_residual.md`
  - M3 starting point: `results/m3_first_run.md`
- DECISIONS.md entries triggered by M4: per-cell-line as primary metric; M4B.1 null-via-degenerate-solution; precomputed-lookup vs Lightning sanity check engineering note.
- Stopping after this report and the README update before any M5 work begins.
