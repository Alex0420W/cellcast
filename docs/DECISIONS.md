# CellCast — Decision log

This is an append-only record of project-level decisions. Each entry: date, decision, context, rationale, and alternatives considered. New decisions append to the bottom; existing entries are not edited (corrections go in a new entry referencing the prior one).

---

## 2026-05-14 — Train on 24h only; 72h becomes counterfactual target

**Context:** Sci-Plex contains 24h (fully crossed, 188 × 4 × 3) and 72h (A549 only, 47 drugs only) data. The 72h subset confounds time × cell_line × drug class.

**Decision:** Main training uses 24h only. 72h is reserved for Tier 4 evaluation (where ground truth exists) and flagged counterfactual demo predictions (where it doesn't).

**Rationale:** Avoid learning entangled signals. Enables clean comparison to ChemCPA. Reframes the data limit as a scientific feature (validated extrapolation) rather than a workaround.

**Alternatives considered:**
- (a) Train on both with time as a token — rejected, learns confounded shortcuts.
- (b) Two-stage 24h-then-72h fine-tune — deferred as a milestone 4 stretch experiment.

---

## 2026-05-14 — Input cell representation: pseudobulk first

**Context:** Sci-Plex 24h gives us 680,685 cells across 2,256 `(drug, dose, cell_line)` conditions. MAMMAL's existing `cell_line_drug_response` task represents the cell via ranked gene expression of one cell-line vector — pseudobulk in spirit. Two paths exist for CellCast: pseudobulk per condition (2,256 training samples, one ranked-expression vector per condition) or per-cell (680k samples, each cell's own ranked-expression vector, sharing labels across cells within a condition).

**Decision:** Train on **pseudobulk** representation in milestone 3. Per-cell stays on the milestone-4 roadmap as a potential follow-up.

**Rationale:** Pseudobulk matches the input distribution MAMMAL was pretrained on for this task family — the ranked-expression-of-a-cell-line encoding has billions of pretrained activations behind it; per-cell ranked expression has none. Pseudobulk also reduces the LFC target to its proper resolution (per-condition LFCs are the population-level signal we want; per-cell LFCs would force the model to fit cell-level noise that's not part of the prediction goal). The 2,256-condition training set is small but each sample is information-dense; pseudobulk is the right place to start.

**Alternatives considered:**
- (a) **Per-cell training** — rejected for M3 because the per-cell input distribution is OOD for the pretrained encoder, and because per-cell LFCs aren't well-defined (control reference is per-cell-line, not per-cell). Worth revisiting in M4 once we know what pseudobulk achieves.
- (b) **Hybrid (per-cell input with pseudobulk-derived LFC label)** — defers complexity without clear benefit at this stage. Reconsider in M4.

---

## 2026-05-14 — Dose encoding: discrete dose tokens

**Context:** MAMMAL's prompt template has no slot for dose; the existing IC50 task has no dose conditioning at all. Sci-Plex doses are exactly four levels (10, 100, 1000, 10000 nM), identical across all 188 drugs.

**Decision:** Add four discrete dose tokens to the tokenizer vocabulary — `<DOSE_10nM>`, `<DOSE_100nM>`, `<DOSE_1000nM>`, `<DOSE_10000nM>` — and emit exactly one in each prompt, immediately after the SMILES block.

**Rationale:** Dose is a 4-level categorical in our training data, not a continuous regressor. Tokens give the model a clean, embedding-learnable representation per dose, mirror how the rest of the MAMMAL prompt is structured, and avoid the need for a parallel scalar-input channel that would require pretrained-encoder retraining to interpret. The 4-token addition is the smallest possible vocabulary delta. At inference time, off-grid doses are explicitly out-of-distribution and should be flagged (consistent with `DESIGN.md` §1 honest-scope pillar) rather than silently extrapolated.

**Alternatives considered:**
- (a) **Scalar literal via `<@TOKENIZER-TYPE=SCALARS_LITERALS>`** — would in principle support arbitrary doses at inference, but flows through `ENCODER_INPUTS_SCALARS`, a channel the pretrained encoder has never seen used on the input side. Pretraining mismatch outweighs the inference-time flexibility, given our training grid only has 4 doses anyway.
- (b) **Log-dose as a single scalar concatenated to the encoder embedding** — clean theoretically, but requires an architectural change to inject a scalar into the encoder input alongside token embeddings. Token form gets the same expressiveness with no architectural delta.
- (c) **No dose conditioning** — rejected outright; dose-response is the whole point.

---

## 2026-05-14 — Label encoding: tensor bypass at LABELS_SCALARS_VALUES (gated by 3A spike test)

**Context:** MAMMAL's scalar regression target is built by parsing a literal numeric value out of `LABELS_STR` via the `SCALARS_LITERALS` sub-tokenizer. For CellCast we need to populate `LABELS_SCALARS_VALUES` as a `[B, G]` tensor (G ≈ 1–5k). The string-literal path was designed for one scalar per sample, not a G-vector.

**Decision (provisional, gated by 3A):** Skip `LABELS_STR` entirely. Populate `LABELS_SCALARS_VALUES` and `LABELS_SCALARS_VALID_MASK` directly as `[B, G]` tensors in `data_preprocessing`, before the batch reaches `ScalarsPredictionsLoss`. The loss is shape-agnostic (`mammal/losses.py:122–147`), so a `[B, G]` preds × `[B, G]` targets MSE works out of the box.

**Rationale:** Tensor bypass is the smallest change. It avoids inventing a multi-literal label-string protocol (which the tokenizer may or may not support — never tested) and keeps the SCALARS tokenizer used only for what it was designed for. If `LABELS_SCALARS_VALUES` can be hand-populated without breaking the collate function or the rest of the data pipeline, we have a clean upgrade path; if not, we know exactly what to fall back to (option b).

**Gate:** This decision is contingent on the 3A spike test (`scripts/spike_label_tensor.py`) confirming that a pre-filled `LABELS_SCALARS_VALUES` flows end-to-end (forward + finite loss + backward + nonzero head gradient). If the spike fails, this entry will be superseded by a new dated entry switching to option (a).

**Alternatives considered:**
- (a) **Inline-literal list in `LABELS_STR`** (`f"<@TOKENIZER-TYPE=SCALARS_LITERALS>{v1} {v2} … {vG}<@TOKENIZER-TYPE=AA>…"`) — fallback if tensor bypass fails. Untested whether the SCALARS tokenizer parses literal lists into multi-element `LABELS_SCALARS_VALUES`; would need its own spike. Adds string-tokenization overhead on every sample.
- (b) **Multi-`<MASK>` readout** (G `<MASK>` tokens in the encoder prompt, one per gene) — rejected: eats encoder context budget that we need for SMILES + ranked genes, and multi-`<MASK>` prompts are OOD for the pretrained encoder.

---

## 2026-05-14 — Device placement is collate's job, not preprocessing's

**Context:** Surfaced during the 3A spike. `Mammal.forward_encoder_only([sample_dict])` invokes `CollateDefault` to batch sample dicts, but `CollateDefault` does not move tensors to the model's device. The existing `cell_line_drug_response.task.data_preprocessing` accepts a `device=` kwarg and calls `.to(device)` inside preprocessing — this couples preprocessing to GPU readiness and makes preprocessing impossible to run on a worker process whose tensors will only later be batched and moved to the GPU.

**Decision:** `data_preprocessing` stays **CPU-side and device-agnostic** in the CellCast task. The collate function (custom if needed, otherwise a small wrapper around `CollateDefault`) takes responsibility for `.to(device)` on every required tensor field (`ENCODER_INPUTS_*`, `LABELS_SCALARS_*`). The `device=` kwarg is removed from preprocessing.

**Rationale:** Standard PyTorch idiom — DataLoader workers materialize CPU tensors, the main process moves them to GPU at batch time. Decouples preprocessing from the runtime device, which is required for any multi-worker DataLoader and for any CPU-only debugging path. Costs one small wrapper; saves a class of bugs and lets us reuse preprocessing for inference on different devices without modification.

---

## 2026-05-14 — HVG flavor: cell_ranger (deviation from M3 plan)

**Context:** M3 plan specified `scanpy.pp.highly_variable_genes(flavor='seurat_v3')` for HVG selection on control populations. seurat_v3 requires scikit-misc, which has no aarch64 wheels and requires gfortran to build from source. gfortran is not installed on the DGX Spark.

**Decision:** Use `flavor='cell_ranger'` instead. It works on log-normalized data, has no exotic dependencies, and is a standard scanpy flavor.

**Rationale:**
- HVG selection is upstream of CellCast's actual research contribution (the multimodal task, counterfactual framing, handoff workflow). Defensible HVG set matters; specific flavor does not.
- cell_ranger is standard, well-documented, and aarch64-clean. Install story for the repo stays simple (`pip install` works on any architecture, no sudo, no Fortran).
- seurat_v3 and cell_ranger typically agree on 70–85% of top-N genes; results should not materially differ.

**Mitigation:** During milestone 4's ChemCPA comparison, install gfortran in a separate environment and recompute the HVG list with seurat_v3. Report the Jaccard overlap. If <70%, escalate.

**Alternatives considered:**
- Install gfortran now — rejected. Adds a sudo-level install step to the repo's reproducibility story, becomes a tax on every collaborator setup.
- Inline reimplementation of seurat_v3 — rejected. Maintenance burden of owning a reference algorithm we then have to validate.

---

## 2026-05-14 — Dose embedding init + trainable rows (3C correction)

Context: 3C initialized the four <DOSE_*> embedding rows to identical values (mean of all existing rows). With the 3D plan of a frozen MAMMAL backbone, identical embeddings produce identical hidden states at the dose-token position, making the model structurally incapable of producing different outputs per dose. The frozen encoder cannot compensate.

Decision: (a) initialize the four dose rows with *differentiated* perturbations of the same magnitude as the mean-init, AND (b) add the four dose embedding rows (4 × 768 = 3,072 params) to the trainable parameter set. All other backbone weights remain frozen.

Rationale:
- Differentiated init gives the frozen encoder four distinguishable inputs at the dose-token position from step 0.
- Trainable dose rows let the head's gradient pull the dose embeddings to wherever they're most useful, breaking out of any unhelpful starting region.
- Combined cost: 3,072 added params (~0.0007% of model). Negligible.
- This is a strict reading violation of "trainable: scalars_prediction_head only" but preserves the spirit (no pretrained weight moves; only never-before-trained new rows).

Caught by: pre-training inspection during 3D setup. The 3C spec's mean-init instruction did not anticipate the frozen-backbone case. Logged here so future-me doesn't have to rediscover it.

Alternatives considered:
- Option A alone (differentiated init, frozen) — rejected: frozen encoder may not interpret the perturbations usefully.
- Option B alone (trainable, identical init) — rejected: symmetry-breaking is slow with identical starting rows.

---

## 2026-05-14 — Training schedule correction (3D)

Context: First training attempt halted at step 144/510 on the near-zero val pcorr tripwire (val/pcorr_macro = +0.0055). Investigation showed the warmup was 200 steps against a 510-step total run (39% warmup) — the model had not yet reached peak LR at the epoch-1 validation checkpoint, so the tripwire fired on a measurement from a half-warmed model.

Decision: (a) reduce warmup from 200 → 25 steps (5% of total run), (b) extend run from 5 → 8 epochs (816 total steps). All other config unchanged.

Rationale:
- 200/510 warmup was a spec error; 5-15% is normal, 39% prevents the model from training before the first val check.
- The halt tripwire is correct in principle but needs to evaluate against a model that has actually had a chance to learn.
- 8 epochs gives the trainer ~775 full-LR steps to converge or plateau, vs ~310 in the original spec.
- Training loss was declining 17% window-to-window and top-50 dir accuracy was +4.2 pp above chance even at step 101 — there is real signal to investigate.

Caught by: the halt tripwire firing exactly when it should have, and the agent diagnosing schedule vs architecture as the proximate cause. The tripwire worked.

### Addendum 2026-05-14 — HVG min_cells filter

cell_ranger HVG selection failed at the quantile-binning step because many genes have ~zero mean expression in the per-cell-line control subsets, causing bin-edge collisions. Resolved by applying `sc.pp.filter_genes(min_cells=10)` to the per-cell-line control subset copy used for HVG selection. This filter is local to the HVG step only — the global gene pool used for LFC computation and the input gene-rank list remains unfiltered. Threshold of 10 cells = 0.13–0.26% of each cell line's control population (A549 = 3,773; K562 = 3,935; MCF7 = 7,786). Conservative: drops only genes with essentially no expression signal in controls.

---

## 2026-05-14 — Per-cell-line macro Pearson is the primary metric

Context: Overall macro Pearson aggregates across cell lines and is dominated by between-cell-line variance. Any model that distinguishes A549 / K562 / MCF7 from the input expression profile inherits a meaningful share of this signal without learning anything drug-specific. CellCast v0 achieved overall pcorr +0.127 with per-cell-line pcorr ≈ 0 — i.e., it looked "okay" overall while learning nothing about drugs.

Decision: per-cell-line macro Pearson is the primary metric for evaluating drug-conditioning. Overall is reported for continuity and historical comparison but is interpretive context, not the headline.

Top-50 DEG direction accuracy and per-cell-line pcorr together form the meaningful evaluation pair.

Caught by: M4A P5 diagnosis (residual ceiling 91.6% but per-cell-line pcorr near zero) and P6 confirmation (a 9M-param MLP beats CellCast v0 by 0.06–0.13 per-cell-line). The overall-pcorr framing was hiding the real picture.

---

## 2026-05-14 — M4B.1 residual reframe: null result via degenerate solution

Context: M4B.1 tested whether residual-to-stratum-mean target reframe alone (head-only training, frozen MAMMAL encoder) would unlock drug-specific signal extraction. Predicted per-CL pcorr 0.02–0.05; result landed at 0.05/0.05/0.09 per-CL — appearing to confirm the hypothesis quantitatively.

Interpretation: the result is mechanistically a null. The trained head learned to predict approximately zero residual everywhere, which under prediction reconstruction (model_output + stratum_mean) recovers the baseline exactly (Δ < 0.001 at every cell line). Val pcorr_resid peaks at +0.004 at epoch 4 and decays — the head correctly identified that the frozen encoder provides essentially no exploitable signal at <MASK>, and degenerated to the zero-residual prediction that minimizes MSE.

The residual reframe by itself does not address the L2 (encoder→<MASK> routing) bottleneck identified in M4A. Reframing the loss without architectural changes leaves the model unable to access drug-specific information, regardless of target.

Implication for M4B.2: LoRA on the encoder is required to open the L2 bottleneck. M4B.2 must demonstrate that LoRA-adapted MAMMAL clears the chemistry-only floor (P6 at 0.06–0.12 per-CL) to justify the foundation-model approach. Below P6 would be a project-defining negative result; well above P6 would justify the architecture.

Lesson on pre-registration: numbers landing in a predicted zone is necessary but not sufficient for hypothesis confirmation. Investigate the mechanism, not just the magnitude. Without per-cell-line decomposition and the val_pcorr_resid trajectory analysis, this would have been miscategorized as a partial success.

---

## 2026-05-14 — Engineering note: precomputed lookups vs Lightning sanity checks

During M4B.1, an initial implementation precomputed a stratum_mean lookup indexed by global condition position. This broke during Lightning's pre-training sanity check (which subsamples to 2 batches), since the global indices didn't align with the subsampled batches. Fixed by looking up stratum_mean per-row from batch metadata.

Going forward: when sample-level metadata needs to be retrieved during training/validation, retrieve it from the batch itself, not from an external precomputed array indexed by global position. Lightning's subsampling for sanity checks, distributed training, and gradient accumulation all break global-index assumptions.
