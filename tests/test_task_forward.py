"""3C-6 smoke test: forward + backward on 8 real preprocessed conditions.

Verifies:
  - Tokenizer expansion adds 4 dose tokens without shifting existing IDs.
  - New embedding rows are initialized to mean-of-existing.
  - The new prompt template (SMILES + dose token + ranked genes) tokenizes
    cleanly; the dose token round-trips through decode unchanged.
  - Forward through the loaded model + wide head produces [B, 7153] preds.
  - Tensor-bypass labels flow through ScalarsPredictionsLoss with finite loss.
  - loss.backward() flows non-zero gradients to the head.
  - One decoded sample prompt is printed for human inspection.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

ROOT = Path(os.path.expanduser("~/cellcast"))
sys.path.insert(0, str(ROOT))

from mammal.keys import ENCODER_INPUTS_TOKENS  # noqa: E402
from src.tasks.drug_response_vector import (   # noqa: E402
    DOSE_TOKENS,
    SLICED_PRED_KEY,
    build_sample_dict,
    collate_to_device,
    configure_frozen_backbone_with_trainable_dose_rows,
    expand_tokenizer_and_embeddings,
    per_gene_pearson_macro,
    per_gene_spearman_macro,
    process_model_output,
    top_k_deg_direction_accuracy,
)

PARQUET = ROOT / "data/sciplex/processed/cellcast_v0.parquet"
HVG_PATH = ROOT / "data/sciplex/processed/hvg_genes.txt"
HF_ID = "ibm/biomed.omics.bl.sm.ma-ted-458m"
BATCH = 8


@pytest.fixture(scope="module")
def env():
    """Load tokenizer, model, expand tokens, attach G-wide head, sample 8 rows."""
    from fuse.data.tokenizers.modular_tokenizer.op import ModularTokenizerOp
    from mammal.model import Mammal, get_encoder_mlp_head

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer_op = ModularTokenizerOp.from_pretrained(HF_ID)
    model = Mammal.from_pretrained(HF_ID).to(device)

    # Snapshot ids of a few representative existing tokens before expansion
    pre_ids = {t: tokenizer_op.get_token_id(t) for t in
               ["[BRCA1]", "<MASK>", "<EOS>", "<PAD>",
                "<SMILES_SEQUENCE>", "<MOLECULAR_ENTITY>"]}

    report = expand_tokenizer_and_embeddings(tokenizer_op, model)

    # Post-expansion: verify those same tokens still have the same IDs
    post_ids = {t: tokenizer_op.get_token_id(t) for t in pre_ids}

    # Attach the G-wide regression head
    hvg = HVG_PATH.read_text().splitlines()
    G = len(hvg)
    emb_dim = model.t5_model.get_input_embeddings().embedding_dim
    model.scalars_prediction_head = get_encoder_mlp_head(
        embedding_size=emb_dim, layers=[768, 768], dropout=0.1, num_classes=G,
    ).to(device)

    # Load 8 conditions from the parquet
    df = pd.read_parquet(PARQUET).sample(n=BATCH, random_state=42).reset_index(drop=True)

    # Build sample dicts (CPU-side; collate handles device)
    samples = []
    for _, r in df.iterrows():
        samples.append(build_sample_dict(
            smiles=r["smiles"],
            dose_bin=r["dose_bin"],
            ranked_genes=list(r["input_gene_ranked_list"]),
            lfc_vector=np.asarray(r["label_lfc_vector"], dtype=np.float32),
            tokenizer_op=tokenizer_op,
        ))

    return {
        "device": device,
        "tokenizer_op": tokenizer_op,
        "model": model,
        "report": report,
        "pre_ids": pre_ids,
        "post_ids": post_ids,
        "df": df,
        "samples": samples,
        "hvg": hvg,
        "G": G,
    }


def test_dose_tokens_registered(env):
    tk = env["tokenizer_op"]
    ids = {t: tk.get_token_id(t) for t in DOSE_TOKENS}
    assert all(v >= 0 for v in ids.values()), f"dose tokens have invalid ids: {ids}"
    assert len(set(ids.values())) == 4, f"dose token ids collided: {ids}"


def test_no_existing_tokens_shifted(env):
    pre, post = env["pre_ids"], env["post_ids"]
    for t in pre:
        assert pre[t] == post[t], f"token {t!r} shifted: {pre[t]} -> {post[t]}"


def test_dose_rows_are_distinct(env):
    """After differentiated init the 4 dose rows must be meaningfully distinct.

    Per DECISIONS.md 2026-05-14: init = mean + orthogonal-perturbation at 10% of
    mean's L2 norm; pairwise cosine distance should sit around 0.005 (1 - cos ~
    1 - cos(angle); with 10% perturb on ~24-norm mean the angle is ~6 degrees
    which is ~0.005 in cosine distance). We assert >1e-3 to confirm distinctness
    without being too tight on the exact magnitude."""
    model = env["model"]
    tk = env["tokenizer_op"]
    emb = model.t5_model.get_input_embeddings().weight
    rows = {t: emb[tk.get_token_id(t)].float().detach() for t in DOSE_TOKENS}
    # Pairwise L2 distance
    toks = list(rows)
    for i in range(len(toks)):
        for j in range(i + 1, len(toks)):
            l2 = (rows[toks[i]] - rows[toks[j]]).norm().item()
            assert l2 > 1e-3, f"dose rows {toks[i]}/{toks[j]} too close: L2={l2:.6f}"


def test_decoded_prompt_round_trip(env):
    """Encode prompt -> decode -> dose token visible in decoded string."""
    tk = env["tokenizer_op"]
    sample0 = env["samples"][0]
    ids = sample0[ENCODER_INPUTS_TOKENS]
    decoded = tk.decode(ids.tolist())
    # The decoded string should contain exactly one of the four dose tokens
    found = [t for t in DOSE_TOKENS if t in decoded]
    assert len(found) == 1, f"expected exactly one dose token in decoded prompt; found {found}"
    # And the SMILES + MASK tags should also be present
    for required in ["<MASK>", "<SMILES_SEQUENCE>", "<MOLECULAR_ENTITY_CELL_GENE_EXPRESSION_RANKED>"]:
        assert required in decoded, f"missing {required!r} in decoded prompt"

    # Print the decoded prompt so it shows up in pytest output -- human visual check
    df = env["df"]
    r = df.iloc[0]
    print()
    print(f"--- decoded prompt for sample 0 ---")
    print(f"  condition_id = {r['condition_id']}")
    print(f"  cell_line    = {r['cell_line']}")
    print(f"  drug_name    = {r['drug_name']}")
    print(f"  dose_bin     = {r['dose_bin']}")
    print(f"  smiles       = {r['smiles']}")
    print()
    print(decoded[:1500] + ("..." if len(decoded) > 1500 else ""))


def test_forward_backward_with_freezing(env):
    """End-to-end: freeze backbone except head+dose rows; check grad mask."""
    device = env["device"]
    model = env["model"]
    samples = env["samples"]
    tk = env["tokenizer_op"]
    G = env["G"]

    freeze_report = configure_frozen_backbone_with_trainable_dose_rows(model, tk)
    print()
    print(f"  trainable: head={freeze_report['head_trainable_params']:,}  "
          f"dose_rows={freeze_report['dose_rows_trainable_params']:,}  "
          f"total={freeze_report['total_trainable_params']:,}")
    expected_dose = 4 * 768
    assert freeze_report["dose_rows_trainable_params"] == expected_dose

    batch_dict = collate_to_device(samples, device=device)

    if device == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    # Snapshot of dose-row embeddings before backward (so we can confirm grad pulls them).
    dose_ids = freeze_report["dose_token_ids"]
    emb_w = model.t5_model.get_input_embeddings().weight
    rows_before = emb_w[dose_ids].detach().clone().float()

    out = model.forward_encoder_only(batch_dict)
    out = process_model_output(out)
    preds = out[SLICED_PRED_KEY]
    assert preds.shape == (BATCH, G), f"preds shape mismatch: {tuple(preds.shape)} vs ({BATCH}, {G})"

    from mammal.losses import ScalarsPredictionsLoss
    loss_fn = ScalarsPredictionsLoss(loss_type="mse", pred_key=SLICED_PRED_KEY)
    loss = loss_fn(out)
    assert torch.is_tensor(loss) and torch.isfinite(loss), f"non-finite loss: {loss!r}"
    print(f"  loss = {loss.item():.6f}")

    loss.backward()

    # Head gradients
    head_params = list(model.scalars_prediction_head.parameters())
    n_with_grad = sum(1 for p in head_params if p.grad is not None and p.grad.abs().sum().item() > 0)
    head_grad_norm = sum(p.grad.norm().item() for p in head_params if p.grad is not None)
    print(f"  head params with nonzero grad: {n_with_grad}/{len(head_params)}")
    print(f"  head grad-norm (sum): {head_grad_norm:.4f}")
    assert n_with_grad > 0

    # Embedding gradient mask: dose rows nonzero, rest zero
    emb_grad = emb_w.grad
    assert emb_grad is not None, "embedding weight has no grad after backward"
    dose_grad = emb_grad[dose_ids].abs().sum().item()
    rest_mask = torch.ones(emb_grad.shape[0], dtype=torch.bool)
    for tid in dose_ids:
        rest_mask[tid] = False
    rest_grad = emb_grad[rest_mask].abs().sum().item()
    print(f"  embedding grad sums: dose-rows={dose_grad:.6f}  other-rows={rest_grad:.6f}")
    assert dose_grad > 0, "dose-row gradients did not flow"
    assert rest_grad == 0.0, f"gradient leaked outside dose rows: rest_grad={rest_grad}"

    if device == "cuda":
        peak_mb = torch.cuda.max_memory_allocated(device) / 1024**2
        print(f"  peak GPU memory: {peak_mb:.1f} MB")


def test_metric_functions_dont_crash(env):
    """Run the three custom metrics on synthetic [B,G] data to confirm they execute."""
    G = env["G"]
    B = 16
    rng = np.random.default_rng(0)
    pred = [rng.standard_normal((B, G)).astype(np.float32) * 0.1]
    target = [rng.standard_normal((B, G)).astype(np.float32) * 0.1]

    r = per_gene_pearson_macro(pred, target)
    s = per_gene_spearman_macro(pred, target)
    a = top_k_deg_direction_accuracy(pred, target, k=50)
    assert np.isfinite(r) and np.isfinite(s) and np.isfinite(a)
    print()
    print(f"  per_gene_pearson_macro (random): {r:+.4f}")
    print(f"  per_gene_spearman_macro (random): {s:+.4f}")
    print(f"  top_50_deg_direction_accuracy (random): {a:.4f}  (expect ~0.5 for random)")
