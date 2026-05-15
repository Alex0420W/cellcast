# 3A spike test — label tensor bypass

**Date:** 2026-05-14
**Script:** `scripts/spike_label_tensor.py`
**Status:** ✅ **PASS** — `LABELS_SCALARS_VALUES` can be pre-populated as a `[B, G]` tensor without going through the `SCALARS_LITERALS` string pipeline. Tensor bypass is the milestone-3 label-encoding path.

## What was tested

- Loaded `ibm/biomed.omics.bl.sm.ma-ted-458m` from HF Hub on CUDA.
- Attached a fresh regression head with `num_classes=5, layers=[256], dropout=0.1` (via `mammal.model.get_encoder_mlp_head`).
- Built a tiny dummy prompt matching `task.py:147` exactly (SMILES `CCO`, three gene tokens `[BRCA1][TP53][MYC]`).
- Set `LABELS_SCALARS_VALUES = tensor([0.1, -0.2, 0.3, -0.4, 0.5])` directly. Set `LABELS_SCALARS_VALID_MASK = ones(5, bool)` directly. **No `LABELS_STR` written**.
- Forward via `model.forward_encoder_only([sample_dict])`; sliced `SCALARS_PREDICTION_HEAD_LOGITS[:, 0, :]` to get `[B=1, G=5]` predictions at the `<MASK>` position; stored under a custom key `model.out.lfc_pred`.
- Loss via `ScalarsPredictionsLoss(loss_type='mse', pred_key='model.out.lfc_pred')`.
- `loss.backward()`; inspected `.grad` on every head parameter.

## Pass criteria — all met

| Criterion | Result |
|---|---|
| Forward executes without exception | ✅ |
| Loss is a finite scalar | ✅ `0.076935` |
| Loss is in plausible range (0–10) | ✅ |
| Backward executes without exception | ✅ |
| ≥1 parameter in `scalars_prediction_head` has nonzero gradient | ✅ **4/4 parameter tensors** have nonzero grad |

## Run output

```
[1/6] load model + tokenizer  (device=cuda)
[2/6] attach fresh G-wide regression head
      embedding_size=768  head params=198149
[3/6] build prompt (matches task.py:147 exactly with dummy SMILES + 3 genes)
      token seq len = 11
[4/6] populate LABELS_SCALARS_VALUES / VALID_MASK by tensor bypass (no LABELS_STR)
[5/6] forward + slice <MASK> position
      raw [B,S,G] = (1, 11, 5)
      sliced [B,G] = (1, 5)
      targets [B,G] = (1, 5)
      valid_mask [B,G] = (1, 5)
[6/6] loss + backward + grad check
      loss = 0.076935
      head params with nonzero grad: 4/4
      head grad-norm (sum over params): 1.554507
      peak GPU memory: 2443.2 MB

PASS — tensor bypass confirmed
```

## Numbers worth remembering

- **Embedding size = 768** (not 1024 as the milestone-2 surface table assumed — the `ma-ted-458m` checkpoint uses 768-dim T5 hidden states). The milestone-2 report's MAMMAL surface table is wrong on this point; head input dim for the real CellCast task is **768**.
- **Peak GPU memory for a 1-sample, seq-len-11 forward+backward = 2.4 GB.** Most of this is the loaded model weights (~458M params × 4 bytes ≈ 1.8 GB) plus a single forward's activations.
- **Head parameter count for G=5, layers=[256] = 198,149** (= 768×256 + 256 + 256×5 + 5). Scales linearly with G — for G=2000 the head would be ~700k params.
- **Head grad-norm = 1.55** on the first step with random head init + small random targets. Healthy magnitude, no vanishing/explosion concerns.

## Mini-gotcha encountered (now fixed in the script)

First attempt failed with `RuntimeError: Expected all tensors to be on the same device, but got index is on cpu, different from other tensors on cuda:0`. The `CollateDefault` invoked by `forward_encoder_only([sample_dict])` does not auto-move tensors to the model's device — the caller must put `ENCODER_INPUTS_TOKENS`, `ENCODER_INPUTS_ATTENTION_MASK`, `LABELS_SCALARS_VALUES`, `LABELS_SCALARS_VALID_MASK` on `model.device` before passing. This is the same pattern as the existing `data_preprocessing` which takes `device=` as a kwarg.

For CellCast's M3 task class: thread `device=` through `data_preprocessing` exactly like `cell_line_drug_response.task.py` does, and the issue doesn't recur.

## Implication

The decision in `DECISIONS.md` dated 2026-05-14 ("Label encoding: tensor bypass at LABELS_SCALARS_VALUES (gated by 3A spike test)") is now **active** — gate satisfied. `OPEN_QUESTIONS.md` Q1 closed.
