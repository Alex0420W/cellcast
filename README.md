# cellcast

A systematic investigation of foundation-model adaptation for single-cell drug response prediction, framed around handoff to experimental teams. The codebase produces a reusable architectural-failure diagnostic for multimodal biomedical foundation models, three convergent negative results on fine-tuning IBM's MAMMAL for per-HVG log-fold-change prediction on Sci-Plex 24h, and a counterfactual prediction workflow designed to flag what it doesn't know.

---

## Key findings

- **The Sci-Plex 24h dataset has 91.6% LFC variance unaccounted for by the per-stratum mean baseline** — substantial drug-specific signal remains for any model to capture (M4A P5).
- **A 9M-parameter Morgan-fingerprint MLP outperforms a 458M-parameter frozen MAMMAL pipeline by 0.06–0.13 per-cell-line Pearson** on the 38-drug held-out test set (P6).
- **Three independent foundation-model approaches converge on the per-stratum-mean baseline** at every cell line within 0.001 Pearson: frozen MAMMAL + full-LFC target (M3), frozen MAMMAL + residual target (M4B.1), and rank-32 LoRA + residual target (M4B.2). At the fine-tuning scale tested, MAMMAL's representation does not propagate drug-discriminative signal to the readout position with enough magnitude to clear the chemistry-only floor.

The negative result is bounded: see the [scope and limitations](#scope-and-limitations) section.

**Full milestone-4 synthesis:** [`results/MILESTONE_4_REPORT.md`](results/MILESTONE_4_REPORT.md).

---

## What's in the repo

- **`docs/`** — design (`DESIGN.md`), append-only decision log (`DECISIONS.md`), open questions, M2 architecture report, M4A diagnostic prompt template.
- **`src/`** — preprocessing (`preprocess.py`), splits (`splits.py`), task modules (`tasks/drug_response_vector.py`, `tasks/drug_response_residual.py`), models (`models/baselines.py`, `models/fingerprint_mlp.py`, `models/lora_setup.py`).
- **`scripts/`** — training and evaluation entry points: `train.py` / `evaluate.py` (M3); `train_residual.py` / `evaluate_residual.py` (M4B.1); `train_residual_lora.py` / `evaluate_residual_lora.py` (M4B.2); `train_fingerprint.py` / `evaluate_fingerprint.py` (P6). Diagnostic probes under `scripts/diag/`.
- **`tests/`** — 48 tests covering preprocessing, splits, task forward path, residual computation, fingerprint MLP, LoRA setup. `pytest tests/` runs in ~16 s.
- **`results/`** — per-milestone reports (M3 first run, P6 baseline, M4A diagnostic SUMMARY + per-probe outputs, M4B.1, M4B.2, M4 close-out synthesis), figures, and machine-readable metric JSONs.
- **`configs/`** — YAML configs for each training run (documentation-only; entry points are the Python scripts).

Large artifacts (parquet data, model checkpoints, full prediction NPZs) are gitignored — see [reproduce](#reproduce) for regeneration commands.

---

## Reproduce

Prerequisites: NVIDIA GPU (training was done on a DGX Spark / GB10), Python 3.11, the `mammal` package and IBM `ma-ted-458m` checkpoint accessible. The repo uses a `.venv` for dependency isolation.

```bash
# Sanity check the pipeline
pytest tests/                          # 48 tests, ~16 s

# Reproduce a milestone end-to-end
python src/models/baselines.py         # fits StratifiedMeanBaseline → results/baseline_predictions.npz
python scripts/train.py                # M3: frozen MAMMAL + head + full-LFC target (~68 min)
python scripts/evaluate.py             # M3 evaluation → results/cellcast_v0_predictions.npz, 3d_metrics.json

python scripts/train_fingerprint.py    # P6: Morgan-FP MLP (~5 s training)
python scripts/evaluate_fingerprint.py # P6 evaluation → results/p6_predictions.npz

python scripts/train_residual.py       # M4B.1: frozen MAMMAL + residual (~68 min)
python scripts/evaluate_residual.py    # M4B.1 evaluation → results/cellcast_residual_predictions.npz

python scripts/train_residual_lora.py  # M4B.2: LoRA r=32 + residual (~76 min)
python scripts/evaluate_residual_lora.py

# Re-run the M4A architectural diagnostic
python scripts/diag/p1_encoder_smiles_sensitivity.py
python scripts/diag/p2_head_sensitivity.py
python scripts/diag/p3_dose_token_influence.py
python scripts/diag/p4_smiles_ablation.py
python scripts/diag/p5_baseline_residual.py
# Outputs land under results/4a_diagnostics/p[1-5]/
```

Every committed result is reproducible from the committed code on the committed checkpoint SHAs. Best checkpoints (which are 2.5–4.9 GB each and gitignored) are saved under `runs/<run_name>/checkpoints/`.

---

## Architectural-diagnostic methodology (M4A probes)

The P1–P5 probe set is foundation-model-agnostic and identifies *where* in a multimodal pipeline the drug-conditioning signal is lost. Each probe is inference-only, runs in seconds on a single GPU, and produces a localizable verdict:

| Probe | Question | Output |
|---|---|---|
| P1 | Does the encoder distinguish drugs at SMILES vs `<MASK>` positions? | Cosine-distance heatmaps; quantifies SMILES → MASK attenuation. |
| P2 | Is the head's input → output mapping smooth or saturated? | Interpolation curve + Jacobian operator norm. |
| P3 | Does the dose token influence predictions at all? | Token-swap MASK cosdist comparison. |
| P4 | What happens if SMILES is replaced with a dummy molecule? | Test metrics with real vs ablated SMILES. |
| P5 | What's the ceiling for any drug-aware model beyond the baseline? | Residual / total variance ratio. |

See `docs/MILESTONE_4A_PROMPT.md` for the full template (re-usable by any team adapting a biomedical foundation model to per-feature regression on perturbation data).

---

## Scope and limitations

The M4 negative result is bounded. **At the fine-tuning scale tested, on Sci-Plex 24h pseudobulk, frozen-or-LoRA-rank-32 MAMMAL ma-ted-458m does not beat a chemistry-only baseline at per-cell-line drug discrimination.** This is the defensible claim. It does *not* claim:

- That higher LoRA ranks (64, 128, 256+) would fail.
- That full fine-tuning of all 458 M parameters would fail.
- That targeting only attention Q+K (most relevant to routing rewrites) would fail.
- That MAMMAL variants pretrained on different objectives would fail.
- That this would hold on other datasets (e.g., Tahoe-100M, Norman2019, other perturb-seq variants).

The honest interpretation is that the project's exploration of foundation-model fine-tuning for this task found three convergent nulls at the tested scale and stopped there rather than chasing increasingly expensive variants. The diagnostic methodology, the chemistry-only floor, and the convergent-null pattern together form a stronger story to tell than "we beat SOTA by 0.01" would have been.

The handoff workflow (counterfactual predictions + calibrated uncertainty + OOD flags + ranked recommendations, designed for a wet-lab user) does not depend on beating a SOTA baseline — it depends on honest uncertainty and clear failure modes. That is the M5/M6 focus going forward.

---

## Milestone history

| tag | what landed |
|---|---|
| `m1-setup` | MAMMAL installed and verified on DGX Spark. |
| `m2-recon` | Sci-Plex EDA + MAMMAL surface recon + design decisions persisted. |
| `m3-baseline` | CellCast v0: frozen MAMMAL + per-HVG regression head; loses to baseline, diagnostic clarity preserved. |
| `m4a-diagnostic` | Drug signal pathway probes P1–P5; encoder→`<MASK>` routing confirmed as primary failure mode; 91.6% headroom remains. |
| `m4b-p6-fingerprint` | Morgan-fingerprint MLP baseline beats CellCast v0 by 0.06–0.13 per-cell-line pcorr. |
| `m4b1-residual-reframe` | Residual-target reframe alone; null result via degenerate zero-prediction. |
| `m4b2-lora-residual` | LoRA r=32 on encoder + residual target; third convergent null; P6 floor not cleared. |
| `m4-closeout` | Synthesis report; project reframed around diagnostic methodology and handoff workflow. |

---

## License

(Single-author research codebase; license forthcoming. Contact for collaboration.)
