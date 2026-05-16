"""M4B.2 — train CellCast with LoRA on T5 encoder + residual target.

Mirrors scripts/train_residual.py with these additions:
  - PEFT LoRA adapters injected into the 84 T5 encoder Linear modules
    (12 blocks × {q, k, v, o, wi_0, wi_1, wo})
  - Two optimizer parameter groups: head+dose at lr=1e-4, LoRA at lr=5e-4
  - Head weights initialized from the M3 best checkpoint (NOT random init)
    so we start from the configuration that already captures per-stratum
    mean structure, and let LoRA + residual target add drug signal on top
  - Per-epoch LoRA-only weight-norm logging (sanity that LoRA is learning,
    not sitting at init)
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

ROOT = Path(os.path.expanduser("~/cellcast"))
sys.path.insert(0, str(ROOT))

from scripts.train import (  # noqa: E402
    CellCastDataset, _cellcast_collate, HF_ID, HVG, PARQUET, SPLITS,
)
from scripts.train_residual import (  # noqa: E402
    CellCastResidualDataModule, CellCastResidualModule,
)
from src.models.lora_setup import (  # noqa: E402
    apply_lora_to_encoder, freeze_for_lora, lora_param_l2_norm,
)
from src.tasks.drug_response_residual import (  # noqa: E402
    DOSE_TOKENS, load_or_expand_tokenizer,
)

RUN_NAME = "cellcast_v0_residual_lora32"
RUN_DIR = ROOT / "runs" / RUN_NAME
RUN_DIR.mkdir(parents=True, exist_ok=True)

M3_CKPT = ROOT / "runs/cellcast_v0/checkpoints/best-7-816-pcorr=0.1282.ckpt"
LORA_RANK = 32
LORA_ALPHA = 32
LORA_DROPOUT = 0.1
LR_HEAD_AND_DOSE = 1e-4
LR_LORA = 5e-4

SEED = 1234


# --------------------------------------------------------------------------- #
# Lightning module — extends CellCastResidualModule with LoRA setup + per-     #
# epoch LoRA-norm logging + dual-LR optimizer.                                 #
# --------------------------------------------------------------------------- #
class CellCastResidualLoRAModule(CellCastResidualModule):
    """Adds LoRA setup, LoRA-norm logging, and dual-LR optimizer to
    CellCastResidualModule. Otherwise identical (residual target,
    full-LFC val pcorr reconstruction)."""

    def __init__(self, *args, lora_rank: int = LORA_RANK,
                 lora_alpha: int = LORA_ALPHA, lora_dropout: float = LORA_DROPOUT,
                 lr_head_and_dose: float = LR_HEAD_AND_DOSE,
                 lr_lora: float = LR_LORA, **kwargs):
        # Save hparams BEFORE calling super (super calls save_hyperparameters)
        super().__init__(*args, **kwargs)
        # Append LoRA-specific hparams
        self.hparams["lora_rank"] = lora_rank
        self.hparams["lora_alpha"] = lora_alpha
        self.hparams["lora_dropout"] = lora_dropout
        self.hparams["lr_head_and_dose"] = lr_head_and_dose
        self.hparams["lr_lora"] = lr_lora
        self._lora_report = None

    def setup(self, stage: str | None = None) -> None:
        # 1) attach tokenizer (if not already)
        if self.tokenizer_op is None and getattr(self.trainer, "datamodule", None) is not None:
            self.tokenizer_op = self.trainer.datamodule.tokenizer_op
        if self.tokenizer_op is None:
            raise RuntimeError("tokenizer_op missing on LightningModule setup")

        # 2) Load M3-trained head + dose-row weights into the bare Mammal+head
        #    (BEFORE LoRA injection, since the ckpt doesn't have lora_* keys).
        if M3_CKPT.exists() and not getattr(self, "_loaded_m3", False):
            if self.global_rank == 0:
                print(f"[setup] loading M3 checkpoint: {M3_CKPT.name}")
            ckpt = torch.load(str(M3_CKPT), map_location="cpu", weights_only=False)
            # The CellCastModule state_dict has keys like "model.t5_model.shared.weight",
            # "model.scalars_prediction_head.classifier.0.weight", etc.
            # Our self.model is the Mammal — so prefix is just "model.<rest>" -> "<rest>"
            new_sd = {}
            for k, v in ckpt["state_dict"].items():
                if k.startswith("model."):
                    new_sd[k[len("model."):]] = v
            # Allow strict=True since both this module and M3 use the same Mammal+head shape
            self.model.load_state_dict(new_sd, strict=True)
            self._loaded_m3 = True

        # 3) Inject LoRA adapters into encoder Q/K/V/O + FFN modules.
        self._lora_report = apply_lora_to_encoder(
            self.model, rank=self.hparams.lora_rank,
            alpha=self.hparams.lora_alpha, dropout=self.hparams.lora_dropout,
        )

        # 4) Freeze with LoRA-aware logic: head + dose rows + LoRA all train;
        #    everything else (T5 base weights, decoder, layer norms, etc.) frozen.
        self._freeze_report = freeze_for_lora(self.model, self.tokenizer_op, DOSE_TOKENS)

        if self.global_rank == 0:
            print(f"[setup] LoRA report: targets={self._lora_report['n_target_modules']} "
                  f"lora_params={self._lora_report['lora_params_actual']:,}", flush=True)
            print(f"[setup] freeze report: head={self._freeze_report['head_trainable_params']:,} "
                  f"lora={self._freeze_report['lora_trainable_params']:,} "
                  f"dose_rows={self._freeze_report['dose_rows_trainable_params']:,} "
                  f"effective_trainable={self._freeze_report['effective_trainable_params']:,} "
                  f"total={self._freeze_report['total_params']:,} "
                  f"({self._freeze_report['effective_trainable_fraction']*100:.2f}% trainable)",
                  flush=True)

    def on_train_epoch_start(self) -> None:
        norms = lora_param_l2_norm(self.model)
        for k, v in norms.items():
            self.log(f"train/{k}", float(v), on_step=False, on_epoch=True)
        if self.global_rank == 0:
            print(f"[lora_norms epoch={self.current_epoch}] "
                  f"A_L2={norms['lora_A_l2']:.4f}  B_L2={norms['lora_B_l2']:.4f}  "
                  f"total_L2={norms['lora_total_l2']:.4f}", flush=True)

    def configure_optimizers(self):
        # Build two parameter groups:
        #   (a) head + dose rows  → lr = lr_head_and_dose
        #   (b) lora_A + lora_B   → lr = lr_lora
        head_dose_params, lora_params, other = [], [], []
        for n, p in self.model.named_parameters():
            if not p.requires_grad:
                continue
            if "lora_" in n:
                lora_params.append(p)
            elif "scalars_prediction_head" in n:
                head_dose_params.append(p)
            elif n.endswith("shared.weight"):  # input embedding (dose-row hook)
                head_dose_params.append(p)
            else:
                other.append((n, p))
        assert not other, f"unexpected trainable param outside groups: {[n for n, _ in other][:5]}"

        if self.global_rank == 0:
            print(f"[opt] head+dose params group: {sum(p.numel() for p in head_dose_params):,}  "
                  f"lora params group: {sum(p.numel() for p in lora_params):,}", flush=True)
        opt = torch.optim.AdamW(
            [
                {"params": head_dose_params, "lr": self.hparams.lr_head_and_dose,
                 "weight_decay": self.hparams.weight_decay},
                {"params": lora_params,      "lr": self.hparams.lr_lora,
                 "weight_decay": self.hparams.weight_decay},
            ]
        )

        def lr_lambda(step: int) -> float:
            if step < self.hparams.warmup_steps:
                return step / max(1, self.hparams.warmup_steps)
            progress = (step - self.hparams.warmup_steps) / max(
                1, self.hparams.total_steps - self.hparams.warmup_steps
            )
            return 0.1 + 0.9 * 0.5 * (1 + np.cos(np.pi * min(1.0, progress)))

        sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
        return {
            "optimizer": opt,
            "lr_scheduler": {"scheduler": sched, "interval": "step", "frequency": 1},
        }


# --------------------------------------------------------------------------- #
# Entry point                                                                  #
# --------------------------------------------------------------------------- #
def main():
    pl.seed_everything(SEED, workers=True)

    n_hvg = len([ln for ln in HVG.read_text().splitlines() if ln.strip()])
    print(f"[main] n_HVG={n_hvg}")

    batch_size = int(os.environ.get("CELLCAST_BATCH", "16"))
    epochs = int(os.environ.get("CELLCAST_EPOCHS", "8"))
    warmup_steps = int(os.environ.get("CELLCAST_WARMUP", "25"))
    print(f"[main] batch_size={batch_size}  epochs={epochs}  warmup_steps={warmup_steps}")

    dm = CellCastResidualDataModule(batch_size=batch_size, num_workers=2, seed=SEED)
    dm.setup()

    steps_per_epoch = (len(dm.train_df) + batch_size - 1) // batch_size
    total_steps = steps_per_epoch * epochs
    print(f"[main] steps_per_epoch={steps_per_epoch}  total_steps={total_steps}")

    lm = CellCastResidualLoRAModule(
        num_HVGs=n_hvg,
        head_layers=(768, 768),
        head_dropout=0.1,
        lr=LR_HEAD_AND_DOSE,                # legacy field; not used by configure_optimizers
        weight_decay=0.01,
        warmup_steps=warmup_steps,
        total_steps=total_steps,
        tokenizer_op=dm.tokenizer_op,
        lora_rank=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        lr_head_and_dose=LR_HEAD_AND_DOSE,
        lr_lora=LR_LORA,
    )

    ckpt_cb = ModelCheckpoint(
        dirpath=str(RUN_DIR / "checkpoints"),
        filename="best-{epoch}-{step}-pcorr={val/pcorr_macro:.4f}",
        monitor="val/pcorr_macro",
        mode="max",
        save_top_k=1,
        save_last=True,
        auto_insert_metric_name=False,
    )
    lr_cb = LearningRateMonitor(logging_interval="step")
    logger = TensorBoardLogger(save_dir=str(RUN_DIR / "tb"), name="", version="")

    trainer = pl.Trainer(
        max_epochs=epochs,
        accelerator="auto",
        devices=1,
        precision="bf16-mixed",
        log_every_n_steps=1,
        gradient_clip_val=1.0,
        callbacks=[ckpt_cb, lr_cb],
        logger=logger,
        num_sanity_val_steps=2,
        default_root_dir=str(RUN_DIR),
    )

    t0 = time.time()
    trainer.fit(lm, datamodule=dm)
    elapsed = time.time() - t0
    print(f"[main] wall_clock={elapsed/60:.1f} min")

    final_lora_norms = lora_param_l2_norm(lm.model)
    summary = {
        "run_name": RUN_NAME,
        "n_HVG": n_hvg,
        "batch_size": batch_size,
        "epochs": epochs,
        "warmup_steps": warmup_steps,
        "lora_rank": LORA_RANK, "lora_alpha": LORA_ALPHA, "lora_dropout": LORA_DROPOUT,
        "lr_head_and_dose": LR_HEAD_AND_DOSE, "lr_lora": LR_LORA,
        "init_from_m3_ckpt": str(M3_CKPT),
        "best_ckpt_path": ckpt_cb.best_model_path,
        "best_val_pcorr_macro_full_LFC": (
            float(ckpt_cb.best_model_score)
            if ckpt_cb.best_model_score is not None else None
        ),
        "wall_clock_min": round(elapsed / 60, 2),
        "monitor_metric": "val/pcorr_macro (FULL-LFC reconstruction)",
        "lora_report": lm._lora_report,
        "freeze_report": {k: (v if isinstance(v, (int, float, str, bool)) else str(v))
                          for k, v in (lm._freeze_report or {}).items()},
        "final_lora_norms": final_lora_norms,
        "n_train_conditions": int(len(dm.train_df)),
        "n_val_conditions": int(len(dm.val_df)),
    }
    (RUN_DIR / "train_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
