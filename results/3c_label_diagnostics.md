# Label distribution diagnostic (pre-3C)

Run on `data/sciplex/processed/cellcast_v0.parquet` — 2,256 conditions × 7,153 HVGs = 16.1M LFC values. Full notebook at `notebooks/02_label_distribution.ipynb`.

- **Distribution is sharply zero-centered.** Median LFC = −0.003; 50% of values fall in [−0.028, +0.023]; 99% in [−0.24, +0.23]. Only **0.04%** of values exceed |LFC| > 1.0.
- **|LFC| bracket fractions:** <0.1 = 87.3%, 0.1–0.5 = 12.4%, 0.5–1.0 = 0.28%, ≥1.0 = 0.04%. The model spends most of its capacity on small movements; the long tail is sparse but where the biological signal lives.
- **Dead-gene rate is low: 143 of 7,153 HVGs (2.0%)** never reach |LFC| > 0.1 in any condition (mostly antisense / pseudogene loci e.g. `AC004803.1`, `ABHD11-AS1`). Could be pruned but probably not worth it pre-training — they're 2% of head outputs and act as anchor/zero predictors.
- **MCF7 has materially smaller LFCs than A549/K562.** Per-condition fraction with |LFC| > 0.1: A549 = 15.4%, K562 = 17.0%, MCF7 = **5.8%** (~3× weaker response). MCF7 mean|LFC| = 0.033 vs A549/K562 ≈ 0.054–0.057. Worth keeping in mind during training/eval — MCF7 conditions are systematically harder targets and **dominate the row count** (344k of 680k cells → larger pseudobulk denominators → smaller per-gene LFC magnitudes).
- **Headline implication for milestone 4:** any aggregate "fit" metric will be dominated by the bulk of small LFC values. Per-gene Pearson on the full HVG set will overweight the small-noise regime; the **top-50 DEG direction-accuracy** metric (which we're adding in 3C) is the more informative number for handoff-grade ranking and should be the primary headline.
