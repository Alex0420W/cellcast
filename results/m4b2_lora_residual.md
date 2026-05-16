# M4B.2 — LoRA on MAMMAL encoder + residual reframe

**Date:** 2026-05-15
**Status:** **Pre-registered outcome D (LoRA didn't engage).** Per-CL pcorr ties baseline at every cell line within ±0.0001, even more tightly than M4B.1. LoRA mechanically trained (B_L2 grew from 0 → 4.30) but the trained adapter produced essentially zero contribution at the readout. P6 floor (0.065/0.083/0.123) **NOT cleared**. Per the agreed truncation rule, this is the brief findings note rather than the long-form report.

---

## Pre-registration (committed before training)

Expected per-CL pcorr **+0.10 to +0.18** (outcome A — LoRA opens L2 bottleneck).
Decision rule recorded landing zones B/C/D/E for other outcomes.

## What happened

| | M4B.2 LoRA | Baseline | M4B.1 residual | CellCast v0 | P6 FP-MLP |
|---|---:|---:|---:|---:|---:|
| overall pcorr | +0.1784 | +0.1784 | +0.1770 | +0.1270 | +0.1897 |
| top50 dir_acc | +0.7388 | +0.7428 | +0.7429 | +0.7081 | +0.7460 |
| **A549 per-CL** | **+0.0488** | **+0.0488** | +0.0483 | +0.0033 | +0.0834 |
| **K562 per-CL** | **+0.0478** | **+0.0478** | +0.0474 | +0.0054 | +0.0645 |
| **MCF7 per-CL** | **+0.0884** | **+0.0883** | +0.0871 | −0.0042 | +0.1233 |

Per-CL pcorr is **identical to the StratifiedMeanBaseline at 4 decimal places** at every cell line. M4B.2 is even closer to baseline than M4B.1 (which differed by ~0.0005 per CL). Outcome: **D (LoRA didn't engage)** — though "didn't engage" needs nuance below.

## Setup recap

| | |
|---|---|
| Backbone | `ibm/biomed.omics.bl.sm.ma-ted-458m` (base weights frozen) |
| LoRA targets | 84 modules: 12 encoder blocks × {q, k, v, o, wi_0, wi_1, wo} |
| LoRA config | rank=32, alpha=32, dropout=0.1, exact encoder-only target list |
| LoRA params | 5,603,328 (matches analytical formula exactly per test_lora_setup.py) |
| Head | M3-initialized (`best-7-816-pcorr=0.1282.ckpt`); trainable |
| Dose rows | 4 trainable rows of embedding (via gradient-mask hook) |
| Effective trainable | 12,288,241 params (2.61% of 470M total) |
| Optimizer | AdamW two param groups: head+dose lr=1e-4, LoRA lr=5e-4 |
| Schedule | Cosine + 25-step warmup, weight_decay=0.01 |
| Loss | MSE on residual target (true_LFC − stratum_mean) |
| Precision | bf16-mixed |
| Batch | 16; Epochs 8 (816 steps); Wall clock **75.9 min** (vs M3 67.9; +12% LoRA overhead) |
| Best ckpt | `runs/cellcast_v0_residual_lora32/checkpoints/best-0-102-pcorr=0.1878.ckpt` (epoch 1) |

## Training trajectory

| epoch | val/pcorr_FULL | val/pcorr_resid | top50_resid | A_L2 | B_L2 |
|---:|---:|---:|---:|---:|---:|
| sanity | 0.1716 | −0.0044 | 0.44 | 29.94 | 0.00 |
| 1 | **0.1878** | +0.0041 | 0.47 | 30.13 | 3.70 |
| 2 | **0.1878** | +0.0055 | 0.51 | 30.16 | 4.04 |
| 3 | **0.1878** | +0.0046 | 0.50 | 30.15 | 4.16 |
| 4 | **0.1878** | **+0.0066** | 0.49 | 30.16 | 4.23 |
| 5 | **0.1878** | +0.0053 | 0.49 | 30.16 | 4.27 |
| 6 | **0.1878** | +0.0053 | 0.50 | 30.15 | 4.29 |
| 7 | **0.1878** | +0.0046 | 0.49 | 30.15 | 4.30 |
| 8 | **0.1878** | +0.0032 | 0.50 | 30.15 | 4.30 |

Three things to read out of this:

1. **val/pcorr_FULL is bit-identical to 4 decimals across 8 epochs.** That's strong evidence the LoRA-induced contribution at the MASK position falls inside the per-stratum-mean noise floor for our 180-condition val set.
2. **val/pcorr_resid traces the same shape as M4B.1**: early peak (+0.0066 at ep 4 — slightly above M4B.1's +0.0041 peak), then decay (down to +0.0032 by ep 8). M4B.2 peak is ~60% higher than M4B.1's peak, but in absolute terms still near zero and still gets undone by the end.
3. **LoRA mechanically trained.** B_L2 grew 0 → 4.30 (from zero init, monotone increase) and A_L2 stayed near 30 (expected — A only gets gradient through B which started at zero). Tripwire 2 not fired; gradients flowed correctly. The adapter has nontrivial weights at end of training; they just don't combine to move the encoder output at MASK in any drug-discriminative direction.

## Why "didn't engage" needs nuance

The LoRA adapters absorbed real gradient signal (B_L2 = 4.30 is well above noise) and the residual pcorr peaked higher than M4B.1's peak. So LoRA is doing *something*. But:

- Whatever it learned is too small or too noise-shaped to survive test-set evaluation. Per-CL pcorr collapses to baseline.
- The fact that the full-LFC val pcorr is flat at 0.1878 for 8 straight epochs is the giveaway: the LoRA contribution is in the noise of the per-stratum-mean recovery.
- This is the same null-via-degenerate-solution pattern from M4B.1, just with more parameters available to express the null. The model has enough capacity to produce a small drug-shaped output but the optimization still settles on "predict near-zero residual" because that minimizes MSE given the limited signal in the (frozen-weight + LoRA-modified) encoder→MASK path.

## What this means for M4B

Three independent attempts now converge:
1. M3: frozen encoder + head + full-LFC target → CellCast v0 underperforms baseline.
2. M4B.1: frozen encoder + head + **residual target** → ties baseline (degenerate zero-prediction).
3. M4B.2: **LoRA encoder** + head + residual target → ties baseline even more tightly.

Adding LoRA to the encoder did not change the outcome. The L2 diagnosis says the bottleneck is encoder→MASK routing; LoRA on rank-32 attention QKV+FFN was the lightest-touch architectural intervention that *should* have opened that path. It didn't.

Plausible candidates for why LoRA didn't help (not all mutually exclusive):
- **Insufficient LoRA capacity.** rank-32 might still be too small for the kind of attention-routing rewrite that's needed to move drug info from SMILES tokens to MASK. Worth probing rank 64 or 128.
- **Wrong target modules.** The L2 attenuation happens at the *MASK position*, which is determined by attention weights to it. The attention-output paths (V, O) and the FFN are targeted, but maybe what's needed is a different layer's attention or a learned re-weighting of which positions MASK attends to. Probe via attention-pattern analysis (e.g., a re-run of M4A P1 on the trained LoRA model — but we agreed to skip that contingent on P6 clearing, so it's deferred).
- **MAMMAL pretraining objective is hostile to this readout.** MAMMAL was pretrained on cell_line_drug_response IC50, where MASK was supposed to attend to genes (the cell-line-defining signal), not SMILES. Even with LoRA freedom on the encoder, the pretrained attention pattern is a hard prior that LoRA can't quickly undo.
- **The head init from M3 was a poor choice for residual target.** M3 head was trained on full LFC and so encodes stratum-mean-shaped outputs. Starting from there means the head has to *unlearn* its dominant signal before LoRA can contribute on top. Worth a follow-up with from-scratch head init.

## Tripwires

| # | Tripwire | Status |
|---|---|---|
| 1 | Train loss flat after 100 post-warmup steps | NOT FIRED — loss declines smoothly |
| 2 | LoRA weight norms unchanged from init at ep 1 | NOT FIRED — B grew 0 → 3.70 in epoch 1 |
| 3 | Val pcorr ep 1 < M4B.1's 0.176 | NOT FIRED — M4B.2 ep 1 = 0.188 |
| 4 | GPU OOM | NOT FIRED |
| 5 | Per-CL pcorr > +0.20 (investigate-before-celebrate) | NOT FIRED — landed at baseline |

## Status

- All 48 tests pass (43 prior + 5 new LoRA-setup tests).
- Wall clock 75.9 min (vs M3 67.9; +12% LoRA overhead).
- Per-CL pcorr ties baseline; P6 floor NOT cleared.
- Per agreement, the contingent P1 follow-up probe (re-run M4A's SMILES vs MASK cosdist on LoRA model) is **skipped** — that probe was justified only as a "narrative payoff" once P6 was cleared.
- Stopping here for discussion before considering follow-ups (rank sweep, target-module ablation, head re-init, or pivot away from MAMMAL).

## Artifacts

| Path | Purpose |
|---|---|
| `runs/cellcast_v0_residual_lora32/checkpoints/best-0-102-pcorr=0.1878.ckpt` | best-by-val/pcorr_full ckpt (epoch 1) |
| `runs/cellcast_v0_residual_lora32/train_summary.json` | hparams + LoRA report + final LoRA norms |
| `runs/cellcast_v0_residual_lora32/tb/events.*` | per-step train loss, per-epoch val + lora_norms |
| `results/cellcast_residual_lora_predictions.npz` | per-condition test predictions (residual + full reconstruction) |
| `results/m4b2_metrics.json` | five-way overall + per-cell-line metrics |
| `src/models/lora_setup.py` | LoRA injection + freeze-with-LoRA + per-module enumeration |
| `configs/cellcast_v0_residual_lora32.yaml` | doc-only config |
| `scripts/train_residual_lora.py` | training entry point |
| `scripts/evaluate_residual_lora.py` | test-set eval with reconstruction |
| `tests/test_lora_setup.py` | 5 tests: encoder-only targeting, param-count formula, frozen-weights unchanged after step, B-at-zero/A-Gaussian init, suffix exact-match |
