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
