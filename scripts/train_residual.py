"""M4B.1 — train CellCast v0 with the residual-to-stratum-mean target.

Mirrors scripts/train.py with one structural change: the dataset's labels are
RESIDUALS (full LFC - per-(cell_line, dose) stratum mean computed on inner-
train drugs only). Forward path, loss type, head shape, optimizer, schedule,
batch size, epochs, and freeze configuration are identical to M3.

Validation logs two pcorr metrics:
  - val/pcorr_macro       — RECONSTRUCTED full-LFC pcorr (residual_pred + stratum_mean
                            vs true_LFC). This is the M3-comparable metric and what
                            the ModelCheckpoint monitor watches. Matches test-time
                            evaluation semantics.
  - val/pcorr_macro_resid — pcorr of raw model output vs residual target. This is
                            what the model directly optimizes against; it's the
                            cleaner "is the model learning anything" diagnostic.
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

# Reuse the existing CellCastModule (forward path is identical) and dataset shell.
from scripts.train import (  # noqa: E402
    CellCastDataset, CellCastModule, _cellcast_collate, HF_ID, HVG, PARQUET, SPLITS,
)
from src.tasks.drug_response_residual import (  # noqa: E402
    SLICED_PRED_KEY,
    StratumMean,
    attach_residual_labels,
    per_gene_pearson_macro,
    per_gene_spearman_macro,
    process_model_output,
    top_k_deg_direction_accuracy,
    load_or_expand_tokenizer,
)

RUN_NAME = "cellcast_v0_residual"
RUN_DIR = ROOT / "runs" / RUN_NAME
RUN_DIR.mkdir(parents=True, exist_ok=True)

SEED = 1234
INTERNAL_VAL_FRAC = 0.10


# --------------------------------------------------------------------------- #
# DataModule that swaps in residual labels                                    #
# --------------------------------------------------------------------------- #
class CellCastResidualDataModule(pl.LightningDataModule):
    def __init__(self, batch_size: int = 16, num_workers: int = 2, seed: int = SEED):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.seed = seed
        self.tokenizer_op = None
        self.train_df = None
        self.val_df = None
        self.stratum_mean: StratumMean | None = None

    def setup(self, stage: str | None = None) -> None:
        df = pd.read_parquet(PARQUET)
        split = json.loads(SPLITS.read_text())
        train_full = df[df["drug_name"].isin(split["train_drugs"])].reset_index(drop=True)

        rng = np.random.default_rng(self.seed)
        perm = rng.permutation(len(train_full))
        n_val = int(round(len(train_full) * INTERNAL_VAL_FRAC))
        val_idx = perm[:n_val]
        tr_idx = perm[n_val:]
        val_df_full = train_full.iloc[val_idx].reset_index(drop=True)
        train_df_full = train_full.iloc[tr_idx].reset_index(drop=True)

        # Stratum mean fit on INNER-train (no leakage from val drugs)
        self.stratum_mean = StratumMean.fit(train_df_full)

        # Replace label_lfc_vector with residual on both inner-train and val
        self.train_df = attach_residual_labels(train_df_full, self.stratum_mean)
        self.val_df = attach_residual_labels(val_df_full, self.stratum_mean)

        print(f"[data] inner_train n_conditions={len(self.train_df)}  "
              f"val n_conditions={len(self.val_df)}", flush=True)
        print(f"[data] residual std on inner_train: {np.stack([np.asarray(v) for v in self.train_df['label_lfc_vector']]).std():.5f}  "
              f"max|residual|: {np.stack([np.asarray(v) for v in self.train_df['label_lfc_vector']]).__abs__().max():.4f}",
              flush=True)
        print(f"[data] full_lfc std on inner_train: {np.stack([np.asarray(v) for v in df[df.drug_name.isin(split['train_drugs'])]['label_lfc_vector']]).std():.5f}",
              flush=True)
        print(f"[data] {len(self.stratum_mean.means)} strata fit on {len(set(train_df_full.drug_name))} inner-train drugs",
              flush=True)

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


# --------------------------------------------------------------------------- #
# Lightning module: subclass to override on_validation_epoch_end so we log     #
# both full-LFC reconstruction pcorr and raw residual pcorr.                  #
# --------------------------------------------------------------------------- #
class CellCastResidualModule(CellCastModule):
    """Override val-epoch logging to compute full-LFC pcorr via reconstruction.

    Reconstruction strategy: stratum_mean is looked up per-row from the
    DataModule's StratumMean using each row's (cell_line, dose_nM) metadata
    captured during validation_step. Robust to partial val passes (Lightning's
    sanity check runs only num_sanity_val_steps batches).
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._val_doses: list[float] = []

    def validation_step(self, batch, batch_idx):
        # Capture dose alongside cell_line so we can rebuild stratum_mean per
        # row in on_validation_epoch_end.
        self._val_doses.extend([float(d) for d in batch["meta_dose_nM"]])
        return super().validation_step(batch, batch_idx)

    def on_validation_epoch_end(self) -> None:
        if not self._val_preds:
            return
        P_resid = torch.cat(self._val_preds, dim=0).numpy()       # [N, G] residual pred
        T_resid = torch.cat(self._val_targets, dim=0).numpy()      # [N, G] residual target
        cls = np.array(self._val_cls)
        doses = self._val_doses

        # Per-row stratum_mean lookup using the DataModule's fitted means.
        # T_full = T_resid + sm because the residual target was constructed as
        # full_lfc - sm; reconstruction is exact (modulo float32 round-trip).
        dm = self.trainer.datamodule
        sm_per_row = dm.stratum_mean.lookup_for_rows(cls, doses).astype(np.float32)
        assert sm_per_row.shape[0] == P_resid.shape[0], (
            f"row mismatch: pred={P_resid.shape[0]} sm_lookup={sm_per_row.shape[0]} "
            f"cls={len(cls)} doses={len(doses)}"
        )
        P_full = P_resid + sm_per_row
        T_full = T_resid + sm_per_row

        def metrics(P_, T_) -> dict:
            return {
                "pcorr_macro": per_gene_pearson_macro([P_], [T_]),
                "spearcorr_macro": per_gene_spearman_macro([P_], [T_]),
                "top50_dir_acc": top_k_deg_direction_accuracy([P_], [T_], k=50),
                "mse": float(((P_ - T_) ** 2).mean()),
            }

        # Primary metrics — on FULL LFC. Monitor key 'val/pcorr_macro' lives here.
        m_full = metrics(P_full, T_full)
        for k, v in m_full.items():
            self.log(f"val/{k}", float(v), prog_bar=(k == "pcorr_macro"))

        # Diagnostic — pcorr on the residual target (what the model directly fits).
        m_resid = metrics(P_resid, T_resid)
        for k, v in m_resid.items():
            self.log(f"val/{k}_resid", float(v))

        # Per-cell-line full-LFC metrics (the M4B primary axis).
        for cl in np.unique(cls):
            mask = (cls == cl)
            if mask.sum() < 2:
                continue
            m = metrics(P_full[mask], T_full[mask])
            for k, v in m.items():
                self.log(f"val/{cl}/{k}", float(v))

        if self.global_rank == 0:
            full_line  = "  ".join(f"{k}={v:+.4f}" for k, v in m_full.items())
            resid_line = "  ".join(f"{k}_resid={v:+.4f}" for k, v in m_resid.items())
            print(f"[val ep={self.current_epoch}] FULL  {full_line}")
            print(f"[val ep={self.current_epoch}] RESID {resid_line}")

        self._val_preds.clear()
        self._val_targets.clear()
        self._val_cls.clear()
        self._val_doses.clear()


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

    lm = CellCastResidualModule(
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
        "warmup_steps": warmup_steps,
        "best_ckpt_path": ckpt_cb.best_model_path,
        "best_val_pcorr_macro_full_LFC": (
            float(ckpt_cb.best_model_score)
            if ckpt_cb.best_model_score is not None else None
        ),
        "wall_clock_min": round(elapsed / 60, 2),
        "monitor_metric": "val/pcorr_macro (FULL-LFC reconstruction)",
        "n_inner_train_drugs": int(len(set(dm.train_df.drug_name))),
        "n_val_drugs": int(len(set(dm.val_df.drug_name))),
        "n_train_conditions": int(len(dm.train_df)),
        "n_val_conditions": int(len(dm.val_df)),
    }
    (RUN_DIR / "train_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
