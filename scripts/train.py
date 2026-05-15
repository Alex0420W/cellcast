"""CellCast v0 training run.

End-to-end first run on 24h Sci-Plex pseudobulk:
  - Frozen MAMMAL backbone + trainable G-wide regression head + 4 trainable dose rows
  - bf16 mixed precision
  - 5 epochs, batch 16, AdamW LR=1e-4 wd=0.01, cosine + 200-step warmup
  - 90% of train-split drugs for training, 10% for in-training validation
  - Best checkpoint by validation macro per-gene Pearson
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
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger

ROOT = Path(os.path.expanduser("~/cellcast"))
sys.path.insert(0, str(ROOT))

from src.tasks.drug_response_vector import (  # noqa: E402
    DOSE_TOKENS,
    SLICED_PRED_KEY,
    build_sample_dict,
    configure_frozen_backbone_with_trainable_dose_rows,
    expand_tokenizer_and_embeddings,
    load_or_expand_tokenizer,
    per_gene_pearson_macro,
    per_gene_spearman_macro,
    process_model_output,
    top_k_deg_direction_accuracy,
)

HF_ID = "ibm/biomed.omics.bl.sm.ma-ted-458m"
PARQUET = ROOT / "data/sciplex/processed/cellcast_v0.parquet"
HVG = ROOT / "data/sciplex/processed/hvg_genes.txt"
SPLITS = ROOT / "data/sciplex/processed/splits.json"

RUN_NAME = "cellcast_v0"
RUN_DIR = ROOT / "runs" / RUN_NAME
RUN_DIR.mkdir(parents=True, exist_ok=True)

SEED = 1234
INTERNAL_VAL_FRAC = 0.10


# ---------------------------------------------------------------------------- #
# Dataset / DataModule                                                         #
# ---------------------------------------------------------------------------- #
class CellCastDataset(torch.utils.data.Dataset):
    def __init__(self, df: pd.DataFrame, tokenizer_op, encoder_max_len: int = 1500):
        self.df = df.reset_index(drop=True)
        self.tokenizer_op = tokenizer_op
        self.encoder_max_len = encoder_max_len

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict:
        r = self.df.iloc[idx]
        sd = build_sample_dict(
            smiles=r["smiles"],
            dose_bin=r["dose_bin"],
            ranked_genes=list(r["input_gene_ranked_list"]),
            lfc_vector=np.asarray(r["label_lfc_vector"], dtype=np.float32),
            tokenizer_op=self.tokenizer_op,
            encoder_input_max_seq_len=self.encoder_max_len,
        )
        # Metadata for per-cell-line metrics; stay strings/floats (collate -> lists)
        sd["meta_cell_line"] = r["cell_line"]
        sd["meta_drug_name"] = r["drug_name"]
        sd["meta_dose_nM"] = float(r["dose_nM"])
        sd["meta_condition_id"] = r["condition_id"]
        return sd


def _cellcast_collate(samples: list[dict]) -> dict:
    from fuse.data.utils.collates import CollateDefault
    return CollateDefault()(samples)


class CellCastDataModule(pl.LightningDataModule):
    def __init__(self, batch_size: int = 16, num_workers: int = 2, seed: int = SEED):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.seed = seed
        self.tokenizer_op = None
        self.train_df = None
        self.val_df = None

    def setup(self, stage: str | None = None) -> None:
        df = pd.read_parquet(PARQUET)
        split = json.loads(SPLITS.read_text())
        train_full = df[df["drug_name"].isin(split["train_drugs"])].reset_index(drop=True)
        rng = np.random.default_rng(self.seed)
        perm = rng.permutation(len(train_full))
        n_val = int(round(len(train_full) * INTERNAL_VAL_FRAC))
        val_idx = perm[:n_val]
        tr_idx = perm[n_val:]
        self.val_df = train_full.iloc[val_idx].reset_index(drop=True)
        self.train_df = train_full.iloc[tr_idx].reset_index(drop=True)
        print(f"[data] train n_conditions={len(self.train_df)}  val n_conditions={len(self.val_df)}")

        self.tokenizer_op, _ = load_or_expand_tokenizer()

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            CellCastDataset(self.train_df, self.tokenizer_op),
            batch_size=self.batch_size, shuffle=True,
            collate_fn=_cellcast_collate, num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            CellCastDataset(self.val_df, self.tokenizer_op),
            batch_size=self.batch_size, shuffle=False,
            collate_fn=_cellcast_collate, num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
        )


# ---------------------------------------------------------------------------- #
# Lightning module                                                             #
# ---------------------------------------------------------------------------- #
class CellCastModule(pl.LightningModule):
    def __init__(
        self,
        *,
        num_HVGs: int,
        head_layers: tuple[int, ...] = (768, 768),
        head_dropout: float = 0.1,
        lr: float = 1e-4,
        weight_decay: float = 0.01,
        warmup_steps: int = 200,
        total_steps: int = 1000,
        tokenizer_op=None,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["tokenizer_op"])
        from mammal.model import Mammal, get_encoder_mlp_head
        model = Mammal.from_pretrained(HF_ID)
        emb_dim = model.t5_model.get_input_embeddings().embedding_dim
        model.scalars_prediction_head = get_encoder_mlp_head(
            embedding_size=emb_dim, layers=list(head_layers),
            dropout=head_dropout, num_classes=num_HVGs,
        )
        self.model = model
        self.tokenizer_op = tokenizer_op
        self._freeze_report = None
        self._val_preds: list[torch.Tensor] = []
        self._val_targets: list[torch.Tensor] = []
        self._val_cls: list[str] = []
        self.num_HVGs = num_HVGs

    def setup(self, stage: str | None = None) -> None:
        if self.tokenizer_op is None and getattr(self.trainer, "datamodule", None) is not None:
            self.tokenizer_op = self.trainer.datamodule.tokenizer_op
        if self.tokenizer_op is None:
            raise RuntimeError("tokenizer_op missing on LightningModule setup")
        self._freeze_report = configure_frozen_backbone_with_trainable_dose_rows(
            self.model, self.tokenizer_op
        )
        if self.global_rank == 0:
            print(f"[setup] trainable: head={self._freeze_report['head_trainable_params']:,}"
                  f"  dose_rows={self._freeze_report['dose_rows_trainable_params']:,}"
                  f"  total={self._freeze_report['total_trainable_params']:,}")

    def forward(self, batch: dict) -> dict:
        out = self.model.forward_encoder_only(batch)
        return process_model_output(out)

    def _compute_loss(self, batch: dict) -> tuple[torch.Tensor, dict]:
        from mammal.losses import ScalarsPredictionsLoss
        out = self.forward(batch)
        loss_fn = ScalarsPredictionsLoss(loss_type="mse", pred_key=SLICED_PRED_KEY)
        loss = loss_fn(out)
        return loss, out

    def training_step(self, batch, batch_idx):
        loss, _ = self._compute_loss(batch)
        self.log("train/loss", loss, on_step=True, on_epoch=False, prog_bar=True, batch_size=len(batch["meta_cell_line"]))
        return loss

    def validation_step(self, batch, batch_idx):
        loss, out = self._compute_loss(batch)
        preds = out[SLICED_PRED_KEY].detach().float().cpu()
        targets = batch["data.labels.scalars.values"].detach().float().cpu()
        if preds.dim() == 1:  # single-sample edge case
            preds = preds.unsqueeze(0)
        if targets.dim() == 1:
            targets = targets.unsqueeze(0)
        self._val_preds.append(preds)
        self._val_targets.append(targets)
        self._val_cls.extend(list(batch["meta_cell_line"]))
        self.log("val/loss_step", loss, on_step=False, on_epoch=True, batch_size=preds.shape[0])
        return loss

    def on_validation_epoch_end(self) -> None:
        if not self._val_preds:
            return
        P = torch.cat(self._val_preds, dim=0).numpy()
        T = torch.cat(self._val_targets, dim=0).numpy()
        cls = np.array(self._val_cls)

        def metrics(P_, T_) -> dict:
            return {
                "pcorr_macro": per_gene_pearson_macro([P_], [T_]),
                "spearcorr_macro": per_gene_spearman_macro([P_], [T_]),
                "top50_dir_acc": top_k_deg_direction_accuracy([P_], [T_], k=50),
                "mse": float(((P_ - T_) ** 2).mean()),
            }

        m_all = metrics(P, T)
        for k, v in m_all.items():
            self.log(f"val/{k}", float(v), prog_bar=(k == "pcorr_macro"))
        for cl in np.unique(cls):
            mask = (cls == cl)
            if mask.sum() < 2:
                continue
            m = metrics(P[mask], T[mask])
            for k, v in m.items():
                self.log(f"val/{cl}/{k}", float(v))

        if self.global_rank == 0:
            line = "  ".join(f"{k}={v:+.4f}" for k, v in m_all.items())
            print(f"[val epoch={self.current_epoch}] {line}")

        self._val_preds.clear()
        self._val_targets.clear()
        self._val_cls.clear()

    def configure_optimizers(self):
        trainable = [p for p in self.parameters() if p.requires_grad]
        opt = torch.optim.AdamW(trainable, lr=self.hparams.lr,
                                weight_decay=self.hparams.weight_decay)
        # Cosine annealing with linear warmup
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


# ---------------------------------------------------------------------------- #
# Entry point                                                                  #
# ---------------------------------------------------------------------------- #
def main():
    pl.seed_everything(SEED, workers=True)

    # Read HVG count to size the head
    n_hvg = len([ln for ln in HVG.read_text().splitlines() if ln.strip()])
    print(f"[main] n_HVG={n_hvg}")

    batch_size = int(os.environ.get("CELLCAST_BATCH", "16"))
    epochs = int(os.environ.get("CELLCAST_EPOCHS", "8"))
    warmup_steps = int(os.environ.get("CELLCAST_WARMUP", "25"))
    print(f"[main] batch_size={batch_size}  epochs={epochs}  warmup_steps={warmup_steps}")

    dm = CellCastDataModule(batch_size=batch_size, num_workers=2, seed=SEED)
    dm.setup()  # eager setup so we can compute total_steps

    steps_per_epoch = (len(dm.train_df) + batch_size - 1) // batch_size
    total_steps = steps_per_epoch * epochs
    print(f"[main] steps_per_epoch={steps_per_epoch}  total_steps={total_steps}")

    lm = CellCastModule(
        num_HVGs=n_hvg,
        head_layers=(768, 768),
        head_dropout=0.1,
        lr=1e-4,
        weight_decay=0.01,
        warmup_steps=warmup_steps,
        total_steps=total_steps,
        tokenizer_op=dm.tokenizer_op,
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

    summary = {
        "run_name": RUN_NAME,
        "n_HVG": n_hvg,
        "batch_size": batch_size,
        "epochs": epochs,
        "best_ckpt_path": ckpt_cb.best_model_path,
        "best_val_pcorr_macro": float(ckpt_cb.best_model_score) if ckpt_cb.best_model_score is not None else None,
        "wall_clock_min": round(elapsed / 60, 2),
    }
    (RUN_DIR / "train_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
