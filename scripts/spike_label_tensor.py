"""3A spike: confirm MAMMAL accepts a pre-filled [B, G] LABELS_SCALARS_VALUES tensor
without going through the SCALARS_LITERALS string round-trip.

Pass criteria: forward executes, loss finite, backward executes, head has nonzero grad.
"""
from __future__ import annotations

import torch
from fuse.data.tokenizers.modular_tokenizer.op import ModularTokenizerOp

from mammal.keys import (
    ENCODER_INPUTS_ATTENTION_MASK,
    ENCODER_INPUTS_STR,
    ENCODER_INPUTS_TOKENS,
    LABELS_SCALARS_VALID_MASK,
    LABELS_SCALARS_VALUES,
    SCALARS_PREDICTION_HEAD_LOGITS,
)
from mammal.losses import ScalarsPredictionsLoss
from mammal.model import Mammal, get_encoder_mlp_head

HF_ID = "ibm/biomed.omics.bl.sm.ma-ted-458m"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
G = 5
SLICED_PRED_KEY = "model.out.lfc_pred"

print(f"[1/6] load model + tokenizer  (device={DEVICE})")
model = Mammal.from_pretrained(HF_ID).to(DEVICE)
tokenizer_op = ModularTokenizerOp.from_pretrained(HF_ID)

print("[2/6] attach fresh G-wide regression head")
emb = model.t5_model.get_input_embeddings().embedding_dim
model.scalars_prediction_head = get_encoder_mlp_head(
    embedding_size=emb, layers=[256], dropout=0.1, num_classes=G,
).to(DEVICE)
n_head_params = sum(p.numel() for p in model.scalars_prediction_head.parameters())
print(f"      embedding_size={emb}  head params={n_head_params}")

print("[3/6] build prompt (matches task.py:147 exactly with dummy SMILES + 3 genes)")
prompt = (
    "<@TOKENIZER-TYPE=SMILES><MASK>"
    "<@TOKENIZER-TYPE=SMILES><MOLECULAR_ENTITY><MOLECULAR_ENTITY_SMALL_MOLECULE><SMILES_SEQUENCE>CCO"
    "<@TOKENIZER-TYPE=GENE><MOLECULAR_ENTITY><MOLECULAR_ENTITY_CELL_GENE_EXPRESSION_RANKED>"
    "[BRCA1][TP53][MYC]"
    "<EOS>"
)
sample_dict: dict = {ENCODER_INPUTS_STR: prompt}
tokenizer_op(
    sample_dict=sample_dict,
    key_in=ENCODER_INPUTS_STR,
    key_out_tokens_ids=ENCODER_INPUTS_TOKENS,
    key_out_attention_mask=ENCODER_INPUTS_ATTENTION_MASK,
    on_unknown="warn",
)
sample_dict[ENCODER_INPUTS_TOKENS] = torch.tensor(sample_dict[ENCODER_INPUTS_TOKENS], dtype=torch.long, device=DEVICE)
sample_dict[ENCODER_INPUTS_ATTENTION_MASK] = torch.tensor(sample_dict[ENCODER_INPUTS_ATTENTION_MASK], dtype=torch.long, device=DEVICE)
print(f"      token seq len = {sample_dict[ENCODER_INPUTS_TOKENS].shape[0]}")

print("[4/6] populate LABELS_SCALARS_VALUES / VALID_MASK by tensor bypass (no LABELS_STR)")
sample_dict[LABELS_SCALARS_VALUES] = torch.tensor([0.1, -0.2, 0.3, -0.4, 0.5], dtype=torch.float, device=DEVICE)
sample_dict[LABELS_SCALARS_VALID_MASK] = torch.ones(G, dtype=torch.bool, device=DEVICE)

print("[5/6] forward + slice <MASK> position")
torch.cuda.reset_peak_memory_stats(DEVICE) if DEVICE == "cuda" else None
out = model.forward_encoder_only([sample_dict])
raw_preds = out[SCALARS_PREDICTION_HEAD_LOGITS]      # [B, S, G]
sliced = raw_preds[:, 0, :]                          # [B, G] at <MASK>
out[SLICED_PRED_KEY] = sliced
print(f"      raw [B,S,G] = {tuple(raw_preds.shape)}")
print(f"      sliced [B,G] = {tuple(sliced.shape)}")
print(f"      targets [B,G] = {tuple(out[LABELS_SCALARS_VALUES].shape)}")
print(f"      valid_mask [B,G] = {tuple(out[LABELS_SCALARS_VALID_MASK].shape)}")

print("[6/6] loss + backward + grad check")
loss_fn = ScalarsPredictionsLoss(loss_type="mse", pred_key=SLICED_PRED_KEY)
loss = loss_fn(out)
assert torch.is_tensor(loss) and torch.isfinite(loss), f"loss not finite: {loss!r}"
print(f"      loss = {loss.item():.6f}")
loss.backward()

head_params = list(model.scalars_prediction_head.parameters())
nonzero_grad = sum(1 for p in head_params if p.grad is not None and p.grad.abs().sum().item() > 0)
grad_norm = sum(p.grad.norm().item() for p in head_params if p.grad is not None)
print(f"      head params with nonzero grad: {nonzero_grad}/{len(head_params)}")
print(f"      head grad-norm (sum over params): {grad_norm:.6f}")

if DEVICE == "cuda":
    peak_mb = torch.cuda.max_memory_allocated(DEVICE) / 1024**2
    print(f"      peak GPU memory: {peak_mb:.1f} MB")

assert nonzero_grad > 0, "no head gradients flowed"
print("\nPASS — tensor bypass confirmed")
