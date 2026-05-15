# CellCast — Open Questions

Questions that are blocking or shape-defining for upcoming milestones. Each entry: question, why it matters, what we already know, what's unresolved, what would close it. Closed questions move to `DECISIONS.md`.

---

## Q1 — How do we encode a G-dimensional regression target in MAMMAL's label pipeline?

**Status:** resolved (2026-05-14) — **tensor bypass at `LABELS_SCALARS_VALUES` confirmed end-to-end** by the 3A spike test (`scripts/spike_label_tensor.py`, results at `results/3a_spike_test.md`). Forward executes, loss is finite (0.077), backward flows nonzero gradients to all 4 head parameter tensors. See `DECISIONS.md` entry "Label encoding: tensor bypass at LABELS_SCALARS_VALUES (gated by 3A spike test)" dated 2026-05-14.

**Why it matters.** MAMMAL's existing scalar regression task (`cell_line_drug_response`) emits exactly one scalar per sample. CellCast needs to emit a per-HVG log-fold-change vector — `G` scalars per sample, where `G` is on the order of 1k–5k. Widening the prediction *head* is trivial (`num_classes=1 → G`, one config edit), but the **label side** is not: MAMMAL builds its scalar targets by parsing a string literal through the `SCALARS_LITERALS` sub-tokenizer, and that pipeline assumes a single literal per sample.

**What we already know (verified against source).**

- Single-scalar label construction (`mammal/examples/cell_line_drug_response/task.py:175–178`):

  ```python
  sample_dict[LABELS_STR] = (
      f"<@TOKENIZER-TYPE=SCALARS_LITERALS>{ground_truth_value}<@TOKENIZER-TYPE=AA>"
      + "".join(["<PAD>"] * (encoder_input_max_seq_len - 1))
  )
  ```

  The tokenizer parses the literal into `LABELS_SCALARS_VALUES` (a `[B]` tensor) and `LABELS_SCALARS_VALID_MASK` (a `[B]` mask).

- The loss (`mammal/losses.py:122–147`) is shape-agnostic: it asserts `preds.shape == targets.shape == valid.shape`, masks invalid positions, and takes a mean. If we feed `[B, G]` predictions and `[B, G]` targets with `[B, G]` validity, MSE works out of the box.

- The mask slice (`mammal/examples/cell_line_drug_response/task.py:228–230`) currently picks position 0 only: `scalars_preds[:, 0]`. For G outputs we read `[:, 0, :]` and the `[B, G]` tensor flows straight into the loss assert.

So the head side and the loss side both already permit a vector target. **The only remaining decision is how to populate `LABELS_SCALARS_VALUES` as a `[B, G]` tensor.**

**Three options carried forward.**

1. **Inline-literal list.** Extend the label string to emit `G` numeric literals in sequence:

   ```python
   literals = " ".join(map(str, lfc_vector))
   LABELS_STR = f"<@TOKENIZER-TYPE=SCALARS_LITERALS>{literals}<@TOKENIZER-TYPE=AA>" + "<PAD>"*…
   ```

   *Unknown:* whether the `SCALARS_LITERALS` tokenizer produces a multi-element `LABELS_SCALARS_VALUES` when given a space-separated literal list, or whether it expects exactly one literal per call. **Investigate:** spike a 5-element call through `ModularTokenizerOp` and inspect the output shape of `LABELS_SCALARS_VALUES`.

2. **Tensor bypass.** Skip `LABELS_STR` entirely; populate `LABELS_SCALARS_VALUES` and `LABELS_SCALARS_VALID_MASK` directly as `[G]`-shaped tensors in `data_preprocessing`. Keep a dummy `LABELS_STR` so the CE branch of the loss has something to operate on (or zero its weight). This is the simplest path if MAMMAL's collate function is happy with a pre-tensorized scalar field.

   *Unknown:* whether the collate / data module asserts that the scalar field came from the tokenizer, or whether a hand-written tensor passes through cleanly. **Investigate:** trace `pl_data_module.py` and `mammal/keys.py` for any required round-trip.

3. **Multi-`<MASK>` readout.** Emit `G` `<MASK>` tokens in the encoder prompt, one per gene, and read off `G` scalar predictions from `G` positions (head stays `num_classes=1`).

   *Pros:* no label-string changes; each gene gets its own readout position with full attention to the prompt. *Cons:* eats encoder context budget (G ≈ 1k tokens would consume most of the 1500-token window we currently spend on SMILES + ranked genes), and the pre-trained model has never seen multiple `<MASK>` tokens in the same prompt. Likely OOD for the encoder.

**Tentative lean.** Option 2 (tensor bypass) is the cleanest and least likely to fight the framework. Option 1 is a fallback if the tokenizer happens to support literal lists natively. Option 3 is rejected unless something forces our hand — it costs too much context and likely degrades the SMILES + gene-ranked-expression conditioning that makes the pretrained encoder useful in the first place.

**What closes this question.** A 20-line spike script that constructs a dummy `sample_dict` with a `[G=10]` target via each of options 1 and 2, runs it through the existing tokenizer + a tiny encoder forward pass, and confirms which one ends up with `preds.shape == targets.shape` and a finite loss. Run it in milestone 3 Week 1, before any data preprocessing work; the choice cascades into every downstream change.
