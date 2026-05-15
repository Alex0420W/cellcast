# MAMMAL single-cell / expression tasks — recon for CellCast

Two finetune examples in `biomed-multi-alignment` touch single-cell / expression data:

| Task | File | Inputs | Output | Mode |
|---|---|---|---|---|
| `scrna_cell_type` | `mammal/examples/scrna_cell_type/task.py` | ranked gene expression (one cell) | cell-type token `[CL:000xxxx]` | encoder–decoder, classification |
| `cell_line_drug_response` | `mammal/examples/cell_line_drug_response/task.py` | drug SMILES + ranked bulk expression (cell line) | IC50 scalar | **encoder-only + scalar regression head** |

There is no task that predicts an expression *vector* as output. Both expression tasks consume expression on the input side only.

## Closest analog to CellCast: `cell_line_drug_response`

Same input modality combo CellCast needs (drug + cell expression context). Only the output shape differs (we need per-gene Δexpression, they emit one scalar IC50).

### Prompt template (from `task.py:146–152`)
```
<@TOKENIZER-TYPE=SMILES><MASK>
<@TOKENIZER-TYPE=SMILES><MOLECULAR_ENTITY><MOLECULAR_ENTITY_SMALL_MOLECULE><SMILES_SEQUENCE>{drug_smiles}
<@TOKENIZER-TYPE=GENE><MOLECULAR_ENTITY><MOLECULAR_ENTITY_CELL_GENE_EXPRESSION_RANKED>[GENE1][GENE2]...[GENEn]
<EOS>
```

Key pieces:
- `<@TOKENIZER-TYPE=...>` switches the active sub-tokenizer (SMILES BPE / GENE vocab / etc.).
- `<MASK>` at the very start is the placeholder where the scalar head reads out — the encoder hidden state at that position is fed to a linear regression head (`ic50_pred`).
- SMILES is character-BPE tokenized; gene names go through the gene-symbol tokenizer with bracketed format `[BRCA1]`.

### Input pipeline (the part we'd reuse for CellCast)
1. Cell-line expression vector + gene names arrive as raw input (in their case from a GDSC TDC dataloader).
2. `sort_genes_by_value_and_name(expressions, genes)` — sorts genes by expression value descending, then by name. Top `max_genes` kept, where `max_genes = encoder_input_max_seq_len − 200 − 6` (200 reserved for SMILES, 6 for format tokens). With default `encoder_input_max_seq_len=1500`, that's **1294 gene tokens** max.
3. Genes wrapped as `[GENE_NAME]` and concatenated into the prompt above.
4. `tokenizer_op(...)` produces `ENCODER_INPUTS_TOKENS`, `ENCODER_INPUTS_ATTENTION_MASK`, and crucially `ENCODER_INPUTS_SCALARS` (parallel scalar stream — only populated when the prompt contains literal numbers, which here it doesn't on the input side).

### Output decoding (`task.py:211–231`)
Encoder-only, no decoder. The scalar regression head (`SCALARS_PREDICTION_HEAD_LOGITS`) outputs `[batch, 1]` at the position of the `<MASK>` token. `process_model_output` reads `scalars_preds[:, 0]` as the IC50 prediction. Loss is MSE against `LABELS_SCALARS_VALUES`, the IC50 ground truth.

### Label encoding for scalar regression (the trick worth copying)
```
LABELS_STR = "<@TOKENIZER-TYPE=SCALARS_LITERALS>{ic50}<@TOKENIZER-TYPE=AA><PAD><PAD>..."
```
The numeric literal is parsed into `LABELS_SCALARS_VALUES` by the tokenizer; the AA + PAD tail is padding to match encoder length. PAD ids in `LABELS_TOKENS` are then replaced with `-100` so cross-entropy ignores them.

## The expression encoding (GeneFormer-style ranked, shared by both tasks)

This is the load-bearing design choice for any sc-task on MAMMAL:

1. **Discretize at preprocessing time** (not at runtime). `scrna_cell_type/data/process_h5ad_data.py` does: filter cells with <200 expressed genes → normalize total counts per cell to 1.0 → `log1p` → uniformly bin into N bins (default 10). The h5ad's `.X` is overwritten with bin numbers.
2. **Sort + drop bin numbers at runtime.** Sort genes descending by bin, then ascending by name within a bin. Drop the bin numbers. The model sees only an ordered list of gene names. **Expression magnitude is implicit in position.**
3. **Truncate to a fixed length.** Sequences past `encoder_input_max_seq_len` are clipped — descending sort means lower-expressed genes get cut first.

Reference implementation: `mammal/examples/scrna_cell_type/task.py:222–240` (`convert_to_double_sorted_geneformer_sequence`).

The README is explicit: "After this double sorting, the expression level (or bin number) are ignored, and the expression profile is represented by the list of gene names in this order."

## Gotchas for CellCast

1. **No continuous expression scalars on the input side.** MAMMAL has a scalar input channel (`ENCODER_INPUTS_SCALARS`), but neither sc task uses it for expression. If we want to feed continuous expression (rather than bins), we'd be doing something the pre-trained model has never seen — likely bad. Stick with ranked-bin encoding.
2. **Gene-name vocabulary must match MAMMAL's tokenizer.** Genes appear as `[BRCA1]` etc. via the GENE sub-tokenizer (`tokenizer/gene_tokenizer.json`). Sci-Plex uses HGNC symbols + Ensembl IDs in `var`; we need to map to whatever vocab `gene_tokenizer.json` was trained on. Unrecognized gene tokens silently disappear (the tokenizer is called with `on_unknown="warn"` in `cell_line_drug_response`; the cell-type task doesn't even set this).
3. **Preprocessing binning is destructive.** Once `.X` is replaced with bin ids, you lose the original counts. Save a copy of the raw data first; do binning in a derived h5ad.
4. **Drug response uses encoder-only + scalar head, cell-type uses encoder-decoder + token output.** Two different model branches. For CellCast (vector output) we'll likely need either (a) encoder + multi-dim scalar head, or (b) encoder-decoder generating a token sequence that decodes back to expression. (a) is closer to drug-response and probably saner.
5. **Truncation to 1294–1500 genes.** Sci-Plex has ~25k genes in `var`. We have to pick a top-N expressed subset per cell. The "expression magnitude is implicit in position" assumption means we should keep the ranking stable: sort by descending expression, truncate.
6. **`limit_samples` in config** is set up for quick subset runs — useful pattern when iterating on the new task.
7. **The cell-type task assumes `adata.obs['cell_type']` exists and gene names already match the tokenizer's gene vocab** (per README). No automatic ID conversion. For Sci-Plex we'd need this conversion as a preprocessing step.

## Concrete checklist when we write the CellCast task in M3

- [ ] Decide control encoding: pre-perturbation cell expression goes into the encoder prompt the same way `cell_line_drug_response` does it (ranked gene names).
- [ ] Drug encoding: reuse the SMILES branch of `cell_line_drug_response` verbatim.
- [ ] Dose encoding: doesn't exist in MAMMAL's existing prompts. Options: discretize into a dose token, or append as a `<@TOKENIZER-TYPE=SCALARS_LITERALS>` literal.
- [ ] Output head: extend the scalar prediction head to `[batch, num_target_genes]` instead of `[batch, 1]`. This is structurally the same as IC50, just wider.
- [ ] Gene-symbol mapping: build a Sci-Plex `var_names` → `gene_tokenizer.json` vocab cross-walk. Drop or rename unknown genes.
- [ ] Decide whether to predict raw post-perturbation log-expression or Δ vs. matched vehicle.
- [ ] Truncate input genes to top-N expressed (~1200), preserving descending-rank ordering used during pretraining.
