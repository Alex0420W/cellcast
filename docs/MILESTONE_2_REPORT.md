# CellCast — Milestone 2 wrap-up (Step D report)

**Date:** 2026-05-14
**Scope:** Sci-Plex 3 acquisition, structural EDA, drug-SMILES resolution, MAMMAL `cell_line_drug_response` task reconnaissance, design decisions and open questions for milestone 3.
**Status:** complete. The repo is in a state where someone joining cold can read this report + `docs/DESIGN.md` + `docs/OPEN_QUESTIONS.md`, run `pytest tests/`, and have a full picture.

---

## 1. Project framing (one paragraph for newcomers)

CellCast predicts per-gene log-fold-change in expression given `(drug SMILES, cell type, dose)`. It's a fine-tune of IBM's MAMMAL foundation model — specifically, of MAMMAL's existing `cell_line_drug_response` task (SMILES + ranked gene expression → scalar IC50), widened from a 1-D scalar regression head to a G-D per-HVG regression head. The target users are wet-lab and target-discovery teams who need ranked, calibrated, audit-traceable handoff predictions, including for combinations not present in training data. See `docs/DESIGN.md` for the four pillars (calibration, prioritization, auditability, honest scope) and the validation strategy.

---

## 2. Repo inventory

All paths relative to `~/cellcast/` unless absolute.

| Path | Bytes | SHA256 |
|---|---:|---|
| `/home/phofan/data/sciplex/SrivatsanTrapnell2020_sciplex3.h5ad` | 2,526,631,614 | `603ed16c5e25401c8a7f5bb0b2b045179701017d65dcfc6aeea71722a66cd10a` |
| `data/sciplex` (→ `/home/phofan/data/sciplex`) | symlink | — |
| `data/PROVENANCE.md` | 1,127 | `9fc92586a1c504ae70f6e2b0f504409a9d2ae5a9d8c02c41d9200e481fa9d1d5` |
| `data/sciplex/drug_smiles.csv` | 29,452 | `cb214bb358e7dbe7dbd132ba0af79949702743bccb2e499b8f502c04bbe43e70` |
| `docs/DESIGN.md` | 7,580 | `86973c0dd769faf02d5e3d2728b70732a4552a2be3c5dd6a061e6b4d1ec6f065` |
| `docs/DECISIONS.md` | 1,098 | `12c7e6fbff6a0a2add714d12d6d4040ed04619ddd15f0fd2291439c53087f0fa` |
| `docs/OPEN_QUESTIONS.md` | 4,676 | `f3b11f08425f1a5104f92b9db147bac1032909c2648acfd114db4a1014d0b835` |
| `docs/MILESTONE_2_REPORT.md` | this file | — |
| `notebooks/01_sciplex_eda.ipynb` | 504,021 | `c4962a2ac5c43ea97cbf1c7c511fa99e79f66399523f005a93826edac0219f41` |
| `notes/mammal_singlecell_tasks.md` | 7,243 | `3913691af6b72a2e0f4e26c571cf2862df6f12c04960699bff54df67bc355cbb` |
| `scripts/lookup_drug_smiles.py` | 9,180 | `b5ac7bb893c81458fe3bdd3706d4ea5aa58402656a7946afa3e2f01fb3170fc5` |
| `scripts/refetch_drug_smiles.py` | 3,799 | `80afe1ae11325ca8fa741a420c8aa1c77cb095028dd68ad4f2594c3a0cd212ba` |
| `scripts/resolve_multi_cid.py` | 4,596 | `a8772a2f2aa89281239f00a3fcece060f121da65a0b94e43cc4777ab1591aa71` |
| `tests/test_drug_smiles_coverage.py` | 3,110 | `aef57a806b295dc55fe3549e179dc85cd88659ad99842ceda146df9ce9a50cb8` |

Pytest: **6/6 pass** (`pytest tests/test_drug_smiles_coverage.py -v`).

---

## 3. Sci-Plex 3 dataset

**Source:** scPerturb v1.4 (Peidli et al. 2024, *Nature Methods*), DOI [10.5281/zenodo.13350497](https://doi.org/10.5281/zenodo.13350497) — the version-corrected release of Srivatsan et al. 2020 (*Science*, original GEO accession GSE139944).

### Raw shape
- **799,317 cells × 110,983 genes**, sparse CSR, `int64` raw counts (min=1, max=2152 in 200k-sample stored nonzeros; integer-valued throughout).
- `obs` has 19 columns (cell_line, perturbation, dose_value, time, plate, well, replicate, target, pathway, pathway_level_{1,2}, plus auxiliaries); `var` has `ensembl_id` + `gene_symbol` (index).
- `layers`, `obsm`, `obsp`, `varm`, `varp`, `uns` are all **empty** — no precomputed PCA / UMAP / neighbors / normalized layer / unstructured metadata.

### After filtering to the 24h training subset
| filter | n cells |
|---|---:|
| raw | 799,317 |
| − demultiplex failures (cell_line NaN) | 762,795 |
| − 72h cells (held out, see §6) | **680,685 (training pool)** |

### 24h subset condition structure (training pool)
- **3 cell lines** — A549 (162,171), K562 (173,652), MCF7 (344,862)
- **188 unique drugs + 1 control** (perturbation == `'control'`, dose_value == 0.0)
- **4 doses per drug, identical across all drugs**: 10, 100, 1000, 10000 nM
- **2,256 fully-populated `(drug, dose, cell_line)` conditions** (188 × 4 × 3, no gaps)
- Cells per condition: min 14, median 251, max 832; **14 conditions <50 cells, 2 conditions <20 cells** (cytotoxic drugs at high dose — expected biology)
- **Control breakdown:** A549 = 3,773; K562 = 3,935; MCF7 = 7,786; total **15,494 control cells in the 24h pool** (~2.3% of valid cells)

### 72h subset (held out for counterfactual evaluation)
- 82,110 cells, **A549 only**, 47 of the 188 drugs only, plus 2,084 control cells. Reserved as Tier 4 evaluation set (see `docs/DESIGN.md` §3).

---

## 4. Drug SMILES resolution

**Source:** PubChem PUG REST. Endpoint sequence: `name → CIDs → SMILES (+ ConnectivitySMILES)` with name-variant fallbacks (paren-strip, parenthetical content, salt-token strip) and a `/synonyms/` last resort.

### Coverage

| status | count | meaning |
|---|---:|---|
| `ok` | 175 | direct name match, single CID |
| `ok_via_variant` | 5 | name cleanup needed (e.g. Glesatinib retry) |
| `ok_multi_cid` | 7 | multiple CIDs returned; took first; flagged for manual review |
| `ok_multi_cid_resolved` | 1 | multi-CID auto-resolved by salt-form preference (`.` in SMILES) |
| `failed` | 0 | — |
| **total** | **188** | **100% covered** |

All 188 SMILES parse with rdkit. 65 drugs have non-trivial stereochemistry (`canonical_smiles != isomeric_smiles`).

### Multi-CID auto-resolution outcome
The 8 multi-CID drugs were re-checked against all candidate CIDs; the rule was "prefer the candidate whose isomeric SMILES does not contain `.` (the SMILES disconnection character indicating a salt complex)."

| drug | outcome | from | to |
|---|---|---|---|
| **Rigosertib (ON-01910)** | **auto-switched** | 23696523 (salt) | 59603054 (free) |
| ENMD-2076 L-(+)-Tartaric acid | flagged — all salt | 66576993 | 66576993 |
| Givinostat (ITF2357) | flagged — all salt | 9804991 | 9804991 |
| GSK-LSD1 2HCl | flagged — all salt | 91826516 | 91826516 |
| Obatoclax Mesylate (GX15-070) | flagged — all salt | 16681698 | 16681698 |
| Tranylcypromine (2-PCPA) HCl | flagged — all salt | 70183457 | 70183457 |
| Temsirolimus | flagged — 3 free-form stereoisomers | 18293306 | 18293306 |
| Veliparib (ABT-888) | flagged — 2 free-form enantiomers | 11960529 | 11960529 |

Of the 7 flagged: 5 are drugs administered as salts (the drug name carries the salt token, so keeping the salt CID is arguably the *more* faithful representation of the experimental species); 2 are stereo-ambiguous (different enantiomers / racemates listed as separate CIDs). The latter two need a manual call about which stereo form to feed the model.

### Schema gotcha worth remembering
PubChem deprecated `CanonicalSMILES` / `IsomericSMILES` property names in 2025. The new names are `SMILES` (stereo-aware) and `ConnectivitySMILES` (achiral). The API silently accepts old names in the request URL but returns new keys in the response, so a naive script returns empty SMILES strings with valid CIDs. The repo scripts use the new names; column semantics in the CSV are preserved (`canonical_smiles` = achiral, `isomeric_smiles` = stereo).

---

## 5. MAMMAL `cell_line_drug_response` task surface

This is the surface we modify in milestone 3. All paths under `~/biomed-multi-alignment/`.

> **Erratum (2026-05-14, surfaced during 3A spike):** Earlier conversation notes referenced an embedding dimension of 1024 for `ma-ted-458m`. The actual loaded checkpoint reports `t5_model.get_input_embeddings().embedding_dim = 768`. The head input dim for the CellCast regression head is therefore **768**, not 1024. See `results/3a_spike_test.md`.

| Concern | File | Lines | Current behavior | Milestone 3 change |
|---|---|---|---|---|
| Encoder prompt builder | `mammal/examples/cell_line_drug_response/task.py` | 146–152 | `<MASK> + SMILES + ranked genes + <EOS>` (no dose, no cell-line token) | Add dose encoding (token or scalar literal) |
| Encoder length budget | same file | 140–143 | 1500 tokens total, 200 reserved for SMILES, 6 for format, ~1294 for genes | Reserve ~10 more for dose |
| Head config | `mammal/model.py` | 151–163 | `num_classes` from `scalars_prediction_head.num_classes`, defaults to 1 | Set to `G` (num HVGs) via YAML |
| Head class | `fuse/dl/models/heads/common.py` | 115–155 | `ClassifierMLP(in_ch=**768**, layers=…, num_classes=…)` | No code change; only widening |
| `<MASK>` slice | `mammal/examples/cell_line_drug_response/task.py` | 228–230 | `scalars_preds[:, 0]` → `[B]` | Change to `scalars_preds[:, 0, :]` → `[B, G]` |
| Squeeze | `mammal/model.py` | 263 | `.squeeze(dim=2)` collapses `[B,S,1]→[B,S]` | No-op when `num_classes>1`; safe to leave |
| Loss | `mammal/losses.py` | 91–147 | `ScalarsPredictionsLoss` is shape-agnostic; asserts `preds.shape == targets.shape`, masks via `LABELS_SCALARS_VALID_MASK`, default MSE | No code change. Per-gene weighting would need a subclass |
| Label encoding | `mammal/examples/cell_line_drug_response/task.py` | 175–178 | `f"<@TOKENIZER-TYPE=SCALARS_LITERALS>{value}<@TOKENIZER-TYPE=AA>" + <PAD>*N` → produces `[B]` `LABELS_SCALARS_VALUES` | **Open question — see Q1** |
| Metrics | `mammal/metrics.py` | 169–217 + `task.py:62–82` | scalar-only: pcorr / spearcorr / mae / mse / rmse / r2 | Add per-gene Pearson, direction accuracy, calibration |

---

## 6. Surprises

Things I'd want a new collaborator to know about before they touch the data.

1. **Mouse genes hidden in the gene index.** `var` has 110,983 entries — 58,347 human (ENSG…) + **52,636 mouse (ENSMUSG…)**, block-structured (human 0..58346, mouse 58347..end). In a 5k-cell sample, mouse columns carry 6,931 total counts vs 10,495,588 human counts (0.066%) — effectively ambient noise from scPerturb's multi-organism harmonized index. Drop mouse genes before HVG selection.

2. **36,522 demultiplex-failure cells (4.6%).** Cells where hash-barcode → well assignment failed. They have NaN for every per-experiment field but `celltype` and `disease` carry the literal string `'None'`. ncounts distribution is similar to valid cells. Drop them — they're unlabeled and unusable for supervision.

3. **72h is severely confounded.** Sci-Plex's 72h arm is **A549 only**, **47 of 188 drugs only**, drug list skewed to epigenetic regulators. Training on both time-points jointly would alias time × cell line × drug class. Decision recorded in `DECISIONS.md` and `DESIGN.md` §2: train on 24h only; 72h is reserved for counterfactual evaluation.

4. **PubChem schema change (2025).** `CanonicalSMILES`/`IsomericSMILES` property keys → `ConnectivitySMILES`/`SMILES`. API silently returns new keys for old-name requests; naive scripts get empty strings back. See §4.

5. **Glyph corruption in one drug name.** `obs['perturbation']` contains the literal string `'Glesatinib?(MGCD265)'` with a real `?` (U+003F) where a hyphen or special character was likely the original. The string-exact join is therefore brittle — `tests/test_drug_smiles_coverage.py::test_glesatinib_glyph_preserved` guards against anyone "cleaning" the name during preprocessing.

6. **Cell-line representation is implicit in the prompt.** MAMMAL's existing task does **not** use a cell-line ID token. Cell identity is fully captured by ranked gene expression (top ~1294 genes by binned expression). MCF7/A549/K562 don't need a vocabulary entry; we just feed their (pseudobulk or per-cell) ranked expression. This is important for understanding which design choices are real choices and which are non-issues (cell-line vocabulary is a non-issue).

---

## 7. Document excerpts

### 7.1 `docs/DESIGN.md` — section headers

- §1 Project framing: predictions for handoff (four pillars: calibration / prioritization / auditability / honest scope; counterfactual prediction as first-class)
- §2 Sci-Plex time-point decision: train on 24h only
- §3 Counterfactual validation strategy (layers a–f: Tier 4 ground-truth / within-24h held-out splits / calibrated OOD uncertainty / mechanistic plausibility / cross-dataset LINCS L1000 / wet-lab handoff)

### 7.2 `docs/OPEN_QUESTIONS.md` — Q1 (label encoding)

> ## Q1 — How do we encode a G-dimensional regression target in MAMMAL's label pipeline?
>
> **Status:** open (carries into milestone 3).
>
> **Why it matters.** MAMMAL's existing scalar regression task emits exactly one scalar per sample. CellCast needs G ~ 1k–5k scalars per sample. Widening the head is one config edit (`num_classes`). The label side is not: targets are parsed by the `SCALARS_LITERALS` sub-tokenizer from a string literal, and that assumes one literal per sample.
>
> **Verified facts.** Single-scalar label is built at `task.py:175–178` via `f"<@TOKENIZER-TYPE=SCALARS_LITERALS>{value}<@TOKENIZER-TYPE=AA>" + <PAD>*N`. The loss (`mammal/losses.py:122–147`) is shape-agnostic — asserts equal shapes, masks invalid, takes mean. The mask slice at `task.py:228–230` picks position 0; for G outputs we'd read `[:, 0, :]`. So head and loss already permit a vector target; only the *label-side population* of `LABELS_SCALARS_VALUES` as a `[B, G]` tensor is unresolved.
>
> **Three options carried forward:** (1) inline-literal list (G space-separated literals in `LABELS_STR`); (2) tensor bypass (skip the string, write the tensor directly into `sample_dict[LABELS_SCALARS_VALUES]`); (3) multi-`<MASK>` readout (rejected unless forced — eats context budget and is OOD for the pretrained encoder).
>
> **Tentative lean:** option 2 (tensor bypass) — least likely to fight the framework. Option 1 is a fallback if the SCALARS tokenizer turns out to handle literal lists. **What closes this:** a 20-line spike running each option through the tokenizer + a tiny encoder forward pass, in milestone 3 Week 1, before preprocessing work.

### 7.3 `notes/mammal_singlecell_tasks.md` (full text)

```markdown
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
```

---

## 8. Ready for milestone 3

**What's settled:**
- Dataset acquired, integrity verified, provenance documented.
- 24h-only training scope locked in (`DECISIONS.md` 2026-05-14).
- 188/188 drugs have validated SMILES; 1 multi-CID auto-resolved, 7 flagged for optional manual review.
- MAMMAL task surface fully mapped — exact files / lines / shapes for every required change.
- Join integrity guarded by `pytest tests/test_drug_smiles_coverage.py` (6/6 passing).

**Decisions still owed before milestone 3 training can start:**

1. **Input cell representation — per-cell vs pseudobulk.** Do we feed each cell's own ranked gene expression to the encoder (~680k training samples, each with its own `(drug, dose, cell_line)` label), or do we pseudobulk to one expression vector per `(drug, dose, cell_line)` condition (2,256 training samples)? Per-cell preserves cell-level heterogeneity and gives a much larger training set; pseudobulk matches the original `cell_line_drug_response` design and reduces noise. Probably per-cell, but worth a short experiment.

2. **Dose encoding — token vs scalar literal.** The MAMMAL prompt has no dose slot. Options: discretize dose into 4 dose tokens (`<DOSE_10nM>`, `<DOSE_100nM>`, `<DOSE_1uM>`, `<DOSE_10uM>`) added to the vocabulary, or append the numeric dose as a `<@TOKENIZER-TYPE=SCALARS_LITERALS>` literal that flows into `ENCODER_INPUTS_SCALARS`. Token form is simpler; scalar form is more general (would support out-of-grid doses at inference time, useful for the counterfactual mode).

3. **Label encoding — tensor bypass vs inline literals.** See `OPEN_QUESTIONS.md` Q1. Resolved by the spike script described there, ideally in milestone 3 Week 1 before preprocessing work begins.

Auxiliary nice-to-decide (not blocking):

- **Gene-vocabulary cross-walk.** Sci-Plex `var` uses HGNC symbols (and a 52k mouse gene block we drop). MAMMAL's gene tokenizer was trained on some specific vocabulary. Map symbols → tokenizer vocab; quantify how many of the 58k human genes resolve. This determines the HVG-candidate pool.

- **HVG count.** The `num_HVGs` choice sets the head width and the per-sample target dimension. 2,000 is a defensible starting point; LINCS L1000 uses 978. Pick after the gene-vocab cross-walk.

- **Manual review for 7 flagged multi-CID drugs.** Optional. 5 are administered as salts (current CID is reasonable); 2 (Temsirolimus, Veliparib) are stereo-ambiguous and deserve a manual call.
