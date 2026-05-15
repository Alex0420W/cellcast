# Milestone 4A — diagnostic probe of the drug signal pathway (agent prompt)

This file is the verbatim prompt to hand to an agent at the start of milestone 4A. It assumes the agent has access to the M3 best-by-val-pcorr checkpoint and the rest of the cellcast repo.

---

Starting milestone 4, sub-step 4A — diagnostic probe of the drug signal pathway.

Context: M3 closed with CellCast failing to beat the StratifiedMeanBaseline on per-gene Pearson and top-50 DEG direction accuracy. Per-cell-line pcorr is near zero (+0.005 / +0.005 / −0.004 for A549/K562/MCF7), while overall pcorr is +0.127 driven almost entirely by between-cell-line variance. The architectural diagnosis: drug-conditioning signal is not propagating to the readout in a way that produces drug-specific predictions. Full context in ~/cellcast/results/m3_first_run.md and its addendum.

Goal for 4A: empirically determine WHERE the drug signal is being lost. Three candidate locations:
  L1. Encoder input: SMILES tokens encode but produce hidden states that are not very different across different drugs (encoder is "averaging out" the drug)
  L2. <MASK>-token hidden state: encoder produces drug-specific hidden states at SMILES positions, but the <MASK> position (where the head reads) is dominated by gene-expression tokens and doesn't reflect the drug
  L3. Head: <MASK>-position hidden state IS drug-specific, but the head's gradient was dominated by the cell-line signal during training so it learned to ignore the drug-dependent variation

Each location implies a different milestone-4 strategy. We need to know which one before choosing.

---

## Probes (each is a small, focused script — none are training; pure inference + analysis)

Build all probes under ~/cellcast/scripts/diag/. Save outputs to ~/cellcast/results/4a_diagnostics/. Each probe must work with the existing best-by-val-pcorr checkpoint from M3.

### Probe P1 — Encoder output sensitivity to SMILES

Question: does the trained encoder produce meaningfully different last-hidden-state representations for different SMILES?

1. Pick 5 drugs from the held-out test set spanning very different chemistries (e.g., a small-molecule HDAC inhibitor + a large kinase inhibitor + a DNA-damage agent + something structurally weird). For each, take the K562 control + 1000nM condition (fix cell line and dose so they vary only in drug).
2. Run forward inference; extract the full encoder last_hidden_state ([1, seq_len, 768]).
3. Compute pairwise cosine distance between the *full encoder outputs* across the 5 drugs. Compute separately the pairwise cosine distance at:
   (a) The full encoder output (mean-pooled over sequence)
   (b) The SMILES-token positions only (mean-pooled over the SMILES span)
   (c) The <MASK> position (single 768-dim vector)
   (d) The gene-token positions only (mean-pooled over the gene span)

4. Report the four 5×5 cosine-distance matrices. The pattern interpretation:
   - If (b) shows high distances but (c) shows low distances → drug signal exists at SMILES tokens but doesn't propagate to <MASK> (location L2)
   - If (b) shows low distances → encoder is collapsing the drug signal at the encoder level (location L1)
   - If both (b) and (c) show high distances → signal is reaching the head (location L3 — the head is the problem)

5. Repeat the analysis swapping cell line (do the same 5 drugs with A549 and MCF7). Are the distance patterns consistent across cell lines?

### Probe P2 — Head sensitivity to its input

Question: does the head produce meaningfully different predictions when given meaningfully different <MASK>-position inputs?

1. Take the head from the trained checkpoint.
2. Generate 100 synthetic <MASK>-position vectors by interpolating between two real <MASK> hidden states from probe P1 (use the two most-different drugs from P1's pairwise distances).
3. For each synthetic vector, run it through the head and compute the resulting LFC prediction.
4. Plot: x-axis = interpolation parameter (0 to 1), y-axis = L2 distance of the head output from the midpoint. If the head output varies smoothly across the interpolation, the head IS using its input. If the head output is nearly constant regardless of input, the head learned to ignore <MASK> variation (location L3 strongly indicated).

5. Compute the Jacobian norm of the head at one real <MASK> point: `||∂head_output / ∂head_input||`. Compare to head weight norm. A near-zero Jacobian relative to weight norm = head is in a saturated / dead regime.

### Probe P3 — Dose token influence

Question: does the model use the dose token at all?

1. Pick 1 drug + 1 cell line + all 4 doses (10/100/1000/10000 nM). For each, run inference; extract the <MASK> hidden state.
2. Compute pairwise cosine distance between the 4 <MASK> vectors. Are they meaningfully different?
3. Compare to: the same 4 forward passes but with the dose token *manually swapped* (e.g., feed dose=10000nM data but with the <DOSE_10nM> token in the prompt). If swapping the token doesn't change the prediction, the dose embedding is being ignored.

### Probe P4 — Drug-token ablation

Question: what happens if we *remove* the SMILES information entirely?

1. Run inference on 50 random held-out test conditions, twice each:
   (a) Normal: real SMILES in the prompt
   (b) Ablated: replace the entire SMILES span with `<PAD>` tokens (or with a fixed dummy SMILES like "CCO" for all conditions)
2. Compute pcorr and top-50 dir_acc on both runs against ground truth.
3. If ablated performance ≈ normal performance, the model wasn't using SMILES at all. If ablated performance is meaningfully worse, the model was using SMILES (just not enough to beat the baseline).

### Probe P5 — Baseline-residual analysis

Question: how much variance in the true LFC is *not* captured by the per-stratum mean?

1. For each test condition, compute: `residual = true_LFC - baseline_prediction`. This is what's left for any model to learn after the trivial cell-line/dose-mean is subtracted.
2. Report: variance(residual) / variance(true_LFC). If residual variance is <10% of total, there's almost nothing for any model to learn beyond the baseline on this dataset. If it's >30%, there's substantial drug-specific signal we should be able to capture.

This probe doesn't directly diagnose CellCast's failure — it tells us the *ceiling* for any model. Critical context for milestone 4 decisions.

---

## Report

Write ~/cellcast/results/4a_diagnostics/SUMMARY.md including:
- Each probe's outputs (matrices, plots, numbers) inline or linked
- A "what the data says" section: which of L1 / L2 / L3 is most supported, and what's the residual variance ceiling
- A "milestone 4B recommendation" section: based on the diagnosis, which of the remaining hypotheses (residual-to-baseline reframe, LoRA/unfreeze, head redesign, etc.) should we pursue first

## Constraints

- This is all inference + analysis. No training, no checkpoints written, no config changes.
- Use the M3 best checkpoint (best-by-val-pcorr). Confirm its SHA256 against the M3 report.
- If any probe surfaces something genuinely surprising (e.g., the model's <MASK> hidden state is identical across all inputs — a sign of a serious bug), stop and report immediately. Don't proceed to the next probe.
- Each probe should take <30 min of wall clock. Total: 2-3 hours of agent time.

Stop after 4A. Do not start 4B.
