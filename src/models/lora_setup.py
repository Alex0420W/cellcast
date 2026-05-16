"""LoRA wiring for M4B.2.

Programmatically attaches LoRA adapters to MAMMAL's T5 encoder ONLY (not
decoder, not embedding, not head). Handles the parameter-freeze interaction:
PEFT freezes everything by default after wrapping; we re-enable head + dose
embedding rows + LoRA params.

Public functions:
  - find_encoder_target_modules(model, suffixes) -> list[str]
  - apply_lora_to_encoder(model, rank, alpha, dropout, target_modules) -> info dict
  - freeze_for_lora(model, tokenizer_op, dose_tokens) -> info dict
"""
from __future__ import annotations

from typing import Any

import torch
from torch import nn

# T5 v1.1 encoder Linear-module name suffixes (gated FFN: wi_0/wi_1 + wo).
ENCODER_LINEAR_SUFFIXES = ("q", "k", "v", "o", "wi_0", "wi_1", "wo")


def find_encoder_target_modules(model: nn.Module,
                                suffixes: tuple[str, ...] = ENCODER_LINEAR_SUFFIXES,
                                encoder_path_prefix: str = "t5_model.encoder.block."
                                ) -> list[str]:
    """Return full dotted-path names of every Linear module inside the encoder
    whose name suffix is in `suffixes`. Decoder modules are excluded.

    Pass these full paths to peft.LoraConfig.target_modules so PEFT matches
    exactly (not by suffix) — important because the decoder has same-suffix
    modules that we must NOT touch.
    """
    out: list[str] = []
    for name, mod in model.named_modules():
        if not isinstance(mod, nn.Linear):
            continue
        if not name.startswith(encoder_path_prefix):
            continue
        last = name.rsplit(".", 1)[-1]
        if last in suffixes:
            out.append(name)
    return out


def apply_lora_to_encoder(
    model: nn.Module,
    *,
    rank: int = 32,
    alpha: int = 32,
    dropout: float = 0.1,
    suffixes: tuple[str, ...] = ENCODER_LINEAR_SUFFIXES,
) -> dict[str, Any]:
    """Inject LoRA adapters into encoder Q/K/V/O + FFN modules.

    Uses `peft.inject_adapter_in_model` rather than `get_peft_model` so the
    original model object is mutated in place (no PeftModel wrapper) and the
    Mammal forward path continues to work without changes. Wraps with
    get_peft_model only for the side effect of installing config; the inject
    path is what actually adds the adapters.

    Returns a report dict with target module list, expected vs actual LoRA
    param count, and LoRA-only param count summary.
    """
    from peft import LoraConfig
    from peft.tuners.lora import LoraModel

    target_modules = find_encoder_target_modules(model, suffixes)
    if not target_modules:
        raise RuntimeError("no LoRA target modules found in encoder")

    # Per-module expected params: rank * (in + out)
    expected = 0
    for name in target_modules:
        mod = model.get_submodule(name)
        expected += rank * (mod.in_features + mod.out_features)

    cfg = LoraConfig(
        r=rank, lora_alpha=alpha, lora_dropout=dropout,
        target_modules=target_modules, bias="none",
        task_type=None,           # FEATURE_EXTRACTION-equivalent (no PEFT task wrapper)
    )

    # Inject adapter modules in place. This calls `LoraModel`'s replacement
    # routine, which swaps each target nn.Linear for a `lora.Linear` that
    # wraps the original frozen weights and adds A/B low-rank matrices.
    LoraModel(model, cfg, adapter_name="default")

    # Count actual LoRA params
    actual_lora = 0
    for n, p in model.named_parameters():
        if "lora_" in n:
            actual_lora += p.numel()

    return {
        "n_target_modules": len(target_modules),
        "target_modules_sample": target_modules[:6] + ["..."],
        "rank": rank, "alpha": alpha, "dropout": dropout,
        "lora_params_expected": expected,
        "lora_params_actual": actual_lora,
        "lora_param_match": (actual_lora == expected),
    }


def freeze_for_lora(
    model: nn.Module,
    tokenizer_op,
    dose_tokens: tuple[str, ...],
) -> dict[str, Any]:
    """Three-way trainability after LoRA injection:
       - LoRA adapter params (lora_A, lora_B): trainable
       - Head (scalars_prediction_head.*): trainable
       - Embedding weight (only the 4 dose-token rows via gradient hook): trainable
       - Everything else: frozen

    Mirrors the existing M3 freeze logic but accounts for LoRA params.
    """
    dose_ids = sorted(tokenizer_op.get_token_id(t) for t in dose_tokens)

    # Step 1: freeze everything (including LoRA — we'll re-enable selectively)
    for p in model.parameters():
        p.requires_grad_(False)

    # Step 2: re-enable LoRA params
    n_lora_enabled = 0
    for n, p in model.named_parameters():
        if "lora_" in n:
            p.requires_grad_(True)
            n_lora_enabled += p.numel()

    # Step 3: re-enable head
    if model.scalars_prediction_head is None:
        raise RuntimeError("model.scalars_prediction_head is not attached")
    n_head_enabled = 0
    for p in model.scalars_prediction_head.parameters():
        p.requires_grad_(True)
        n_head_enabled += p.numel()

    # Step 4: re-enable embedding weight + register dose-row gradient mask hook
    emb = model.t5_model.get_input_embeddings()
    emb.weight.requires_grad_(True)
    dose_id_tensor = torch.tensor(dose_ids, dtype=torch.long)

    def _mask_grad(grad: torch.Tensor) -> torch.Tensor:
        masked = torch.zeros_like(grad)
        masked.index_copy_(0, dose_id_tensor.to(grad.device),
                           grad.index_select(0, dose_id_tensor.to(grad.device)))
        return masked

    handle = emb.weight.register_hook(_mask_grad)
    if not hasattr(model, "_cellcast_hooks"):
        model._cellcast_hooks = []
    model._cellcast_hooks.append(handle)

    n_dose_rows = len(dose_ids) * emb.embedding_dim

    # Sanity totals. NOTE: torch's requires_grad counter overcounts the
    # embedding table (all 99k_vocab × 768 = 76M params have requires_grad=True
    # to let the hook see gradients, but the hook zeros all rows except the
    # 4 dose IDs). Report both so the discrepancy is explicit.
    n_total = sum(p.numel() for p in model.parameters())
    n_requires_grad_naive = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_effective = n_head_enabled + n_lora_enabled + n_dose_rows

    return {
        "dose_token_ids": dose_ids,
        "embedding_dim": emb.embedding_dim,
        "head_trainable_params": n_head_enabled,
        "lora_trainable_params": n_lora_enabled,
        "dose_rows_trainable_params": n_dose_rows,
        "effective_trainable_params": n_effective,
        "requires_grad_naive_count": n_requires_grad_naive,
        "requires_grad_overcount_explained_by": (
            f"embedding has requires_grad=True ({emb.weight.numel():,}) but hook "
            f"masks all rows except dose_ids; effective trainable rows = {len(dose_ids)}"
        ),
        "total_params": n_total,
        "effective_trainable_fraction": n_effective / n_total,
        "gradient_mask": f"backward hook on input-embedding masks rows outside {dose_ids}",
    }


def lora_param_l2_norm(model: nn.Module) -> dict[str, float]:
    """Aggregate LoRA-A and LoRA-B L2 norms across the whole model.
    Useful for the per-epoch sanity log: at init lora_B is zero (so B norm
    starts at 0); A is Gaussian-init with small std. Both should grow during
    training if LoRA is learning.
    """
    a_sq = 0.0
    b_sq = 0.0
    for n, p in model.named_parameters():
        if "lora_A" in n and p.requires_grad:
            a_sq += float(p.detach().pow(2).sum())
        elif "lora_B" in n and p.requires_grad:
            b_sq += float(p.detach().pow(2).sum())
    return {
        "lora_A_l2": a_sq ** 0.5,
        "lora_B_l2": b_sq ** 0.5,
        "lora_total_l2": (a_sq + b_sq) ** 0.5,
    }
