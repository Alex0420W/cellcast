"""Tests for M4B.2 LoRA setup.

Three load-bearing properties:
  1. LoRA modules attach only to the 84 ENCODER target modules (12 blocks ×
     {q, k, v, o, wi_0, wi_1, wo}), never to the decoder, the embedding, or
     the head.
  2. Frozen non-LoRA Linear weights are unchanged after a backward+step
     (only LoRA A/B and head + dose rows can move).
  3. LoRA param count matches the analytical formula 2 * rank * (in + out)
     per target module.

These tests use a tiny synthetic T5-like fixture so they don't require
loading the 458M-param Mammal checkpoint. The lora_setup helpers operate
on any nn.Module with the right submodule names, so the fixture is
sufficient to exercise the wiring.
"""
from __future__ import annotations

import os, sys
from pathlib import Path

import pytest
import torch
from torch import nn

sys.path.insert(0, os.path.expanduser("~/cellcast"))

from src.models.lora_setup import (
    ENCODER_LINEAR_SUFFIXES, apply_lora_to_encoder, find_encoder_target_modules,
)


# ---- Tiny synthetic T5-like fixture ---------------------------------------- #
class _SelfAttn(nn.Module):
    def __init__(self, d=16):
        super().__init__()
        self.q = nn.Linear(d, d, bias=False)
        self.k = nn.Linear(d, d, bias=False)
        self.v = nn.Linear(d, d, bias=False)
        self.o = nn.Linear(d, d, bias=False)


class _GatedFFN(nn.Module):
    def __init__(self, d=16, d_ff=32):
        super().__init__()
        self.wi_0 = nn.Linear(d, d_ff, bias=False)
        self.wi_1 = nn.Linear(d, d_ff, bias=False)
        self.wo = nn.Linear(d_ff, d, bias=False)


class _T5LayerSelfAttention(nn.Module):
    def __init__(self, d=16):
        super().__init__()
        self.SelfAttention = _SelfAttn(d)


class _T5LayerFF(nn.Module):
    def __init__(self, d=16, d_ff=32):
        super().__init__()
        self.DenseReluDense = _GatedFFN(d, d_ff)


class _T5Block(nn.Module):
    def __init__(self, d=16, d_ff=32):
        super().__init__()
        # Match the real T5 layer ordering: layer[0] = attention, layer[1] = FFN
        self.layer = nn.ModuleList([_T5LayerSelfAttention(d), _T5LayerFF(d, d_ff)])


class _Encoder(nn.Module):
    def __init__(self, n_blocks=3, d=16, d_ff=32):
        super().__init__()
        self.block = nn.ModuleList([_T5Block(d, d_ff) for _ in range(n_blocks)])


class _Decoder(nn.Module):
    """Decoder has the SAME module suffixes (q/k/v/o/wi_0/wi_1/wo). The
    encoder-only filter must skip these."""
    def __init__(self, n_blocks=2, d=16, d_ff=32):
        super().__init__()
        self.block = nn.ModuleList([_T5Block(d, d_ff) for _ in range(n_blocks)])


class _T5Model(nn.Module):
    def __init__(self, d=16, d_ff=32, n_enc=3, n_dec=2):
        super().__init__()
        self.encoder = _Encoder(n_enc, d, d_ff)
        self.decoder = _Decoder(n_dec, d, d_ff)


class _MammalLike(nn.Module):
    """Mammal-like wrapper: has `.t5_model` and a head named like the real
    one. This is the minimal interface lora_setup operates on."""
    def __init__(self, d=16, d_ff=32, num_classes=11, n_enc=3, n_dec=2):
        super().__init__()
        self.t5_model = _T5Model(d, d_ff, n_enc, n_dec)
        # head-shaped module — name matches the real `scalars_prediction_head`
        self.scalars_prediction_head = nn.Sequential(
            nn.Linear(d, d, bias=False), nn.ReLU(), nn.Linear(d, num_classes, bias=False),
        )

    def forward(self, x):
        # exercise both encoder and decoder + head so backward sees them
        h = x
        for blk in self.t5_model.encoder.block:
            q = blk.layer[0].SelfAttention.q(h)
            h = blk.layer[1].DenseReluDense.wo(
                blk.layer[1].DenseReluDense.wi_0(q + h)
            )
        for blk in self.t5_model.decoder.block:
            h = h + blk.layer[0].SelfAttention.q(h)
        return self.scalars_prediction_head(h)


# ---- Test 1: encoder-only targeting --------------------------------------- #
def test_find_encoder_target_modules_skips_decoder_and_head():
    m = _MammalLike(d=16, d_ff=32, n_enc=3, n_dec=2)
    targets = find_encoder_target_modules(m, suffixes=ENCODER_LINEAR_SUFFIXES)
    # Expect 3 blocks × 7 modules per block = 21 encoder targets
    assert len(targets) == 21, f"expected 21 encoder targets, got {len(targets)}"
    assert all(t.startswith("t5_model.encoder.block.") for t in targets), \
        f"some target outside encoder: {[t for t in targets if not t.startswith('t5_model.encoder.block.')]}"
    assert not any("decoder" in t for t in targets), \
        f"decoder leak: {[t for t in targets if 'decoder' in t]}"
    assert not any("scalars_prediction_head" in t for t in targets), \
        f"head leak: {[t for t in targets if 'scalars_prediction_head' in t]}"
    # Verify each suffix represented in the right count
    by_suffix: dict[str, int] = {}
    for t in targets:
        s = t.rsplit(".", 1)[-1]
        by_suffix[s] = by_suffix.get(s, 0) + 1
    for s in ENCODER_LINEAR_SUFFIXES:
        assert by_suffix.get(s) == 3, f"{s}: expected 3 (one per block), got {by_suffix.get(s)}"


# ---- Test 2: LoRA param count formula ------------------------------------- #
def test_lora_param_count_matches_formula():
    m = _MammalLike(d=16, d_ff=32, n_enc=3)
    rank = 8
    report = apply_lora_to_encoder(m, rank=rank, alpha=rank, dropout=0.0)
    # Per attn module (q/k/v/o): rank * (16 + 16) = 32 * rank
    # Per wi_0/wi_1: rank * (16 + 32) = 48 * rank
    # Per wo: rank * (32 + 16) = 48 * rank
    # Per block: 4 * 32 * rank + 2 * 48 * rank + 1 * 48 * rank = 272 * rank
    # 3 blocks → 816 * rank
    expected = 816 * rank
    assert report["lora_params_expected"] == expected, (
        f"formula expected {expected}, report says {report['lora_params_expected']}"
    )
    assert report["lora_params_actual"] == expected, (
        f"actual {report['lora_params_actual']} != expected {expected}"
    )
    assert report["lora_param_match"] is True
    assert report["n_target_modules"] == 21
    # Sanity: lora_A and lora_B exist on every target module
    found_a = sum(1 for n, _ in m.named_parameters() if "lora_A" in n)
    found_b = sum(1 for n, _ in m.named_parameters() if "lora_B" in n)
    assert found_a == 21 and found_b == 21


# ---- Test 3: frozen non-LoRA weights are unchanged after backward+step ---- #
def test_frozen_weights_unchanged_after_step():
    torch.manual_seed(0)
    m = _MammalLike(d=16, d_ff=32, n_enc=2, n_dec=1, num_classes=4)

    # Apply LoRA + freeze (mirror what train_residual_lora.py does, minus the
    # MAMMAL-specific embedding hook which doesn't apply to our fixture).
    apply_lora_to_encoder(m, rank=4, alpha=4, dropout=0.0)
    for p in m.parameters():
        p.requires_grad_(False)
    for n, p in m.named_parameters():
        if "lora_" in n:
            p.requires_grad_(True)
    for p in m.scalars_prediction_head.parameters():
        p.requires_grad_(True)

    # Capture snapshots of: (a) a frozen non-LoRA encoder Linear weight,
    # (b) a frozen decoder Linear weight, (c) the trainable head weight.
    enc_q_base = m.t5_model.encoder.block[0].layer[0].SelfAttention.q.base_layer.weight
    dec_q = m.t5_model.decoder.block[0].layer[0].SelfAttention.q
    head_w = m.scalars_prediction_head[0].weight
    snap_enc_q = enc_q_base.detach().clone()
    snap_dec_q = dec_q.weight.detach().clone()
    snap_head  = head_w.detach().clone()

    # Single forward + backward + optimizer step on synthetic input
    opt = torch.optim.AdamW([p for p in m.parameters() if p.requires_grad], lr=1e-2)
    x = torch.randn(3, 16)
    pred = m(x)
    loss = pred.pow(2).mean()
    loss.backward()
    opt.step()

    # Frozen encoder base weight: unchanged
    assert torch.allclose(enc_q_base, snap_enc_q, atol=0), \
        "frozen encoder Linear base_layer.weight changed after step"
    # Frozen decoder weight: unchanged (PEFT shouldn't touch it; freeze keeps it)
    assert torch.allclose(dec_q.weight, snap_dec_q, atol=0), \
        "frozen decoder Linear weight changed after step (LoRA target may have leaked into decoder)"
    # Head weight: changed (was trainable)
    assert not torch.allclose(head_w, snap_head, atol=0), \
        "head weight did not change after step despite being trainable"


# ---- Test 4 (bonus): LoRA-A starts nonzero, LoRA-B starts at zero --------- #
def test_lora_initialization_lora_B_at_zero_lora_A_gaussian():
    m = _MammalLike(d=32, d_ff=64, n_enc=2)
    apply_lora_to_encoder(m, rank=4, alpha=4, dropout=0.0)
    for n, p in m.named_parameters():
        if "lora_A" in n:
            # Gaussian init: should have non-trivial norm
            assert p.detach().norm() > 1e-3, f"lora_A weight has near-zero norm at init: {n}"
        elif "lora_B" in n:
            # B init at zero per LoRA paper
            assert torch.allclose(p.detach(), torch.zeros_like(p)), \
                f"lora_B weight not zero at init: {n}"


# ---- Test 5 (bonus): suffix matcher doesn't over-match -------------------- #
def test_suffix_matching_is_exact_not_substring():
    """A module named 'queries' (ending in 's') must NOT match suffix 'q'.
    Our matcher splits on '.' and checks the last component, so this should
    be enforced."""
    class _Distractor(nn.Module):
        def __init__(self, d=8):
            super().__init__()
            self.queries = nn.Linear(d, d, bias=False)  # name ends with 's', not 'q'
    class _M(nn.Module):
        def __init__(self):
            super().__init__()
            self.t5_model = nn.Module()
            self.t5_model.encoder = nn.Module()
            self.t5_model.encoder.block = nn.ModuleList([
                nn.Module(),
            ])
            self.t5_model.encoder.block[0].layer = nn.ModuleList([
                _Distractor(),
            ])
    m = _M()
    targets = find_encoder_target_modules(m, suffixes=("q",))
    assert targets == [], f"suffix matcher over-matched 'queries' as 'q': {targets}"
