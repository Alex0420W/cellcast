# CellCast — Design

## 1. Project framing: predictions for handoff

CellCast is a tool for generating drug-response predictions that are intended to be **handed off** — to a wet-lab biologist, a target-discovery team, or a downstream computational pipeline. That framing is load-bearing. A handoff-grade prediction is not the same artifact as a leaderboard number: it has to be ranked, defensible, and accompanied by the context a recipient needs to act on it. The system is built around four pillars:

**Calibration.** When CellCast assigns 90% confidence to a predicted log-fold-change being in a stated direction, that statement should be right 90% of the time on held-out data. Calibration is what separates a model whose probabilities can be reasoned about from one whose outputs are merely ordinal. We commit to evaluating and reporting calibration alongside accuracy at every milestone; an uncalibrated model is treated as broken, not as a tuning opportunity.

**Prioritization.** For any query — a drug class, a target, a phenotype direction — CellCast returns a *ranked* list of predicted effects with their uncertainties, not a single best guess. The unit of output is a top-N table that a downstream user can scan, filter, and pull from. Ranking quality is evaluated directly (NDCG@K, recall@K on held-out experiments), not derived from regression error.

**Auditability.** Every prediction is traceable. For each (drug, cell line, dose) query, the system surfaces the *k* nearest training examples in the relevant embedding space, the similarity scores, the uncertainty estimate, and any mechanism notes attached to the drug (target, pathway). A user looking at a prediction can answer "why does the model think this?" without retraining. This is what makes the output handoff-grade rather than oracle-grade.

**Honest scope.** Predictions outside the training distribution are flagged, not hidden. If a query involves a cell line CellCast hasn't seen, a dose outside the training range, or a chemical scaffold poorly represented in training data, the response carries an explicit out-of-distribution flag and a wider uncertainty band. The system never silently extrapolates as though it were interpolating.

Within this frame, **counterfactual prediction — predicting drug responses for combinations not measured in the training data — is a first-class feature, not an afterthought.** The scientific value of CellCast is largely in the predictions for things we *haven't* run yet. The validation strategy in §3 is designed around this; the 24h-only training decision in §2 is what makes those counterfactuals clean enough to defend.

## 2. Sci-Plex time-point decision: train on 24h only

Sci-Plex 3 (Srivatsan et al. 2020, scPerturb v1.4 release) contains two time-points, but they are not symmetric. The **24-hour subset is a fully crossed design**: 188 drugs × 4 doses × 3 cell lines = 2,256 conditions, every cell populated, with a median of 264 cells per condition. The **72-hour subset is severely confounded**: it covers A549 only, 47 of the 188 drugs only, and the 47 drugs are heavily skewed toward epigenetic regulators. Time, cell line, and drug class are all aliased in that subset.

If we trained on both time-points jointly with time as an input feature, the model would have no clean signal to disentangle them. A "72h effect" learned during training would in practice be an "A549 effect" or an "epigenetic drug effect"; the model would happily attribute variance to whichever covariate the optimizer found convenient, and we would have no way to know which one it picked. That kind of entangled shortcut is hard to detect at evaluation time and devastating at handoff time.

**Decision: main training uses 24h cells only** (≈680k cells across A549/K562/MCF7, with the demultiplex-failure cells also dropped). The 72h data is not discarded; it is repurposed:

- **72h A549, 47 drugs:** this becomes a Tier 4 held-out evaluation set. Ground truth exists, so we can measure how well a model trained on 24h generalizes to a longer exposure window — a genuine counterfactual prediction with a label.
- **72h MCF7/K562 and 72h other-drug combinations:** no ground truth exists for these. They become flagged demo-mode counterfactual predictions: the system will produce them on request, but with explicit OOD warnings and elevated uncertainty.

This decision reframes a data limitation as a scientific feature: instead of pretending we have a balanced 24h-vs-72h dataset, we use the asymmetry to demonstrate calibrated extrapolation under shift.

## 3. Counterfactual validation strategy

A handoff-grade counterfactual prediction needs to be defensible on more than one axis. Single metrics are gameable, and a single held-out split can hide pathological behavior under distribution shift. CellCast uses a layered validation strategy:

**(a) Tier 4 held-out evaluation where ground truth exists.** The 72h A549 subset described above. Models are scored on per-gene log-fold-change error, direction accuracy, and rank correlation against measured responses, with calibration plots overlaid. This is the strongest possible test: a genuine counterfactual with a label.

**(b) Within-24h held-out splits designed to mimic the 72h-gap structure.** We construct held-out splits inside the 24h data that reproduce the *shape* of the counterfactual gap — for example, held-out (drug, cell-line) combinations where the drug has been seen in other cell lines and the cell line has been seen with other drugs, but the specific pair is novel. This gives many independent test points with the same generalization structure as the real counterfactual use case, without waiting on the small Tier 4 set.

**(c) Calibrated uncertainty that grows with distance from training distribution.** The model's reported uncertainty is required to widen as queries move further from training data — measured as distance to the nearest training neighbor in the relevant embedding space. We evaluate this monotonicity directly: bin queries by nearest-neighbor distance and verify that empirical error tracks reported uncertainty within each bin. A model that is confidently wrong on OOD queries fails this check and does not ship.

**(d) Mechanistic plausibility checks against known drug biology.** For drugs with well-characterized targets, predicted responses are checked against expected signatures — HDAC inhibitors should perturb histone-related gene programs, JAK inhibitors should suppress STAT targets, and so on. These checks are not the primary metric, but they catch silent failure modes where regression error is low but biology is wrong. Implemented as a fixed plausibility test suite over a curated drug set.

**(e) Optional cross-dataset checks against LINCS L1000.** Where the same drug × cell-line combinations exist in LINCS, we compare CellCast's predictions to the corresponding L1000 signatures. Cross-platform agreement is a weak signal individually (different technology, different time points) but a useful aggregate consistency check.

**(f) Direct wet-lab validation as the formal handoff path.** The terminal form of validation is the experiment itself: a prioritized top-N prediction goes to a collaborator, they run the assay, and the result either confirms or falsifies the prediction. This is out of scope for this repository but is the **intended downstream workflow** that the four pillars are designed to support. Every internal validation step above exists because it makes a real handoff more or less likely to succeed.
