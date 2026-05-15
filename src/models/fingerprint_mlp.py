"""P6 — Morgan-fingerprint MLP baseline for the M4B framing question.

A small MLP that ingests (Morgan FP + cell-line one-hot + dose one-hot) and
predicts a per-HVG LFC vector. Trained on the residual-to-stratum-mean
target so it can't win by re-learning the per-stratum mean. Compared
against StratifiedMeanBaseline (M3) and CellCast v0 (M3).

Tests whether ANY drug-aware model can beat the per-stratum mean,
independent of MAMMAL.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
import torch
from torch import nn

CELL_LINES = ("A549", "K562", "MCF7")
DOSE_NMS = (10.0, 100.0, 1000.0, 10000.0)
MORGAN_NBITS = 2048
MORGAN_RADIUS = 2
INPUT_DIM = MORGAN_NBITS + len(CELL_LINES) + len(DOSE_NMS)  # 2055


# ---- Fingerprint computation ------------------------------------------------ #
def morgan_fp(smiles: str, *, radius: int = MORGAN_RADIUS,
              nbits: int = MORGAN_NBITS) -> np.ndarray:
    """Return a [nbits] uint8 0/1 vector for the molecule. Raises on parse failure."""
    from rdkit import Chem
    from rdkit.Chem import AllChem
    from rdkit.DataStructs import ConvertToNumpyArray
    m = Chem.MolFromSmiles(smiles)
    if m is None:
        raise ValueError(f"unparseable SMILES: {smiles!r}")
    fp = AllChem.GetMorganFingerprintAsBitVect(m, radius=radius, nBits=nbits)
    arr = np.zeros(nbits, dtype=np.uint8)
    ConvertToNumpyArray(fp, arr)
    return arr


def build_drug_fp_table(drug_smiles_csv_path: str | None = None) -> dict[str, np.ndarray]:
    """drug_name -> [nbits] uint8 fingerprint, computed from isomeric_smiles."""
    import os
    from pathlib import Path
    if drug_smiles_csv_path is None:
        drug_smiles_csv_path = os.path.expanduser("~/cellcast/data/sciplex/drug_smiles.csv")
    df = pd.read_csv(drug_smiles_csv_path)
    if "isomeric_smiles" not in df.columns or "drug_name" not in df.columns:
        raise RuntimeError(f"drug_smiles.csv missing required cols: {df.columns.tolist()}")
    out: dict[str, np.ndarray] = {}
    for _, r in df.iterrows():
        if pd.isna(r["isomeric_smiles"]):
            continue
        out[r["drug_name"]] = morgan_fp(r["isomeric_smiles"])
    return out


# ---- Feature assembly ------------------------------------------------------- #
def encode_features(
    drug_names: Iterable[str],
    cell_lines: Iterable[str],
    doses_nM: Iterable[float],
    fp_table: dict[str, np.ndarray],
) -> np.ndarray:
    """Stack [Morgan FP | cell-line one-hot | dose-bin one-hot] per row."""
    drug_names = list(drug_names)
    cell_lines = list(cell_lines)
    doses_nM = [float(d) for d in doses_nM]
    n = len(drug_names)
    X = np.zeros((n, INPUT_DIM), dtype=np.float32)
    cl_idx = {c: i for i, c in enumerate(CELL_LINES)}
    dose_idx = {d: i for i, d in enumerate(DOSE_NMS)}
    for i in range(n):
        d = drug_names[i]
        if d not in fp_table:
            raise KeyError(f"drug {d!r} missing from fp_table")
        X[i, :MORGAN_NBITS] = fp_table[d]
        X[i, MORGAN_NBITS + cl_idx[cell_lines[i]]] = 1.0
        X[i, MORGAN_NBITS + len(CELL_LINES) + dose_idx[doses_nM[i]]] = 1.0
    return X


# ---- Stratum-mean target (no leakage) --------------------------------------- #
@dataclass
class StratumMean:
    """Per-(cell_line, dose) mean LFC vector. Fit on TRAIN drugs only."""
    means: dict[tuple[str, float], np.ndarray]   # 12 entries (3 cl × 4 dose)
    G: int

    @classmethod
    def fit(cls, train_df: pd.DataFrame) -> "StratumMean":
        means: dict[tuple[str, float], np.ndarray] = {}
        for (cl, dose), group in train_df.groupby(["cell_line", "dose_nM"]):
            mat = np.stack([np.asarray(v, dtype=np.float32)
                            for v in group["label_lfc_vector"]])
            means[(cl, float(dose))] = mat.mean(axis=0)
        Gs = {v.shape[0] for v in means.values()}
        assert len(Gs) == 1, f"inconsistent G across strata: {Gs}"
        return cls(means=means, G=Gs.pop())

    def lookup_for_rows(self, cell_lines: Iterable[str],
                        doses_nM: Iterable[float]) -> np.ndarray:
        """Return [N, G] stack of stratum means for each row."""
        cls_l = list(cell_lines)
        doses_l = [float(d) for d in doses_nM]
        return np.stack([self.means[(cls_l[i], doses_l[i])] for i in range(len(cls_l))])

    def residual_target(self, df: pd.DataFrame) -> np.ndarray:
        """target = true_LFC - stratum_mean (broadcast lookup)."""
        true = np.stack([np.asarray(v, dtype=np.float32) for v in df["label_lfc_vector"]])
        sm = self.lookup_for_rows(df["cell_line"], df["dose_nM"])
        return true - sm

    def reconstruct(self, residual_pred: np.ndarray, df: pd.DataFrame) -> np.ndarray:
        """Add stratum_mean back to a residual prediction to recover full LFC."""
        sm = self.lookup_for_rows(df["cell_line"], df["dose_nM"])
        return residual_pred + sm


# ---- Model ------------------------------------------------------------------ #
class FingerprintMLP(nn.Module):
    """3-layer MLP: input_dim -> 1024 -> 1024 -> num_HVGs.

    Same head shape as CellCast (768->768->768->num_HVGs has similar parameter
    count given the input dim, and matches the spirit of "fair head comparison").
    Dropout 0.2 between hidden layers.
    """

    def __init__(self, input_dim: int = INPUT_DIM,
                 hidden: tuple[int, ...] = (1024, 1024),
                 num_HVGs: int = 7153,
                 dropout: float = 0.2) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        prev = input_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers += [nn.Linear(prev, num_HVGs)]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---- Training (in-process, returns artifact dict) --------------------------- #
def train_fingerprint_mlp(
    *,
    X_train: np.ndarray, Y_train: np.ndarray,   # [N, 2055], [N, G]
    X_val: np.ndarray,   Y_val: np.ndarray,
    val_for_pcorr_cls: np.ndarray | None = None,  # for per-cl logging if desired
    num_HVGs: int,
    epochs: int = 50,
    batch_size: int = 64,
    lr: float = 1e-3,
    weight_decay: float = 0.01,
    warmup_frac: float = 0.05,
    patience: int = 5,
    device: str | None = None,
    seed: int = 1234,
    verbose: bool = True,
) -> dict:
    """Train the FingerprintMLP. Returns dict of artifacts (best state_dict,
    train/val curves, best epoch, time)."""
    import time
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    g = torch.Generator().manual_seed(seed)

    model = FingerprintMLP(input_dim=X_train.shape[1], num_HVGs=num_HVGs).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    n_train = X_train.shape[0]
    steps_per_epoch = max(1, (n_train + batch_size - 1) // batch_size)
    total_steps = epochs * steps_per_epoch
    warmup_steps = max(1, int(warmup_frac * total_steps))

    def lr_at(step: int) -> float:
        if step < warmup_steps:
            return float(step) / float(warmup_steps)
        # cosine to 0
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + np.cos(np.pi * progress))

    Xt_train = torch.from_numpy(X_train).float()
    Yt_train = torch.from_numpy(Y_train).float()
    Xt_val = torch.from_numpy(X_val).float().to(device)
    Yt_val = torch.from_numpy(Y_val).float().to(device)

    train_loss_hist: list[float] = []
    val_loss_hist: list[float] = []
    val_pcorr_hist: list[float] = []
    best_val_pcorr: float = -np.inf
    best_state: dict[str, torch.Tensor] | None = None
    best_epoch: int = -1
    bad_epochs: int = 0
    step = 0
    t0 = time.time()

    for epoch in range(epochs):
        # Shuffle indices
        perm = torch.randperm(n_train, generator=g).numpy()
        epoch_losses: list[float] = []
        model.train()
        for s in range(0, n_train, batch_size):
            idx = perm[s:s + batch_size]
            xb = Xt_train[idx].to(device)
            yb = Yt_train[idx].to(device)
            for pg in opt.param_groups:
                pg["lr"] = lr * lr_at(step)
            opt.zero_grad(set_to_none=True)
            pred = model(xb)
            loss = nn.functional.mse_loss(pred, yb)
            loss.backward()
            opt.step()
            epoch_losses.append(float(loss.detach().cpu()))
            step += 1
        train_loss_hist.append(float(np.mean(epoch_losses)))

        # Val
        model.eval()
        with torch.no_grad():
            vpred = model(Xt_val)
            vloss = float(nn.functional.mse_loss(vpred, Yt_val).cpu())
            # Per-gene macro Pearson on val
            P = vpred.float().cpu().numpy()
            T = Yt_val.float().cpu().numpy()
            Pm = P - P.mean(axis=0, keepdims=True)
            Tm = T - T.mean(axis=0, keepdims=True)
            Pstd = Pm.std(axis=0); Tstd = Tm.std(axis=0)
            valid = (Pstd > 1e-8) & (Tstd > 1e-8)
            num = (Pm * Tm).mean(axis=0)
            denom = np.where((Pstd * Tstd) == 0, 1.0, Pstd * Tstd)
            r = np.where(valid, num / denom, np.nan)
            vpcorr = float(np.nanmean(r))
        val_loss_hist.append(vloss)
        val_pcorr_hist.append(vpcorr)

        improved = vpcorr > best_val_pcorr + 1e-6
        if improved:
            best_val_pcorr = vpcorr
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch + 1
            bad_epochs = 0
        else:
            bad_epochs += 1

        if verbose:
            print(f"[ep {epoch+1:>2d}/{epochs}] "
                  f"train_loss={train_loss_hist[-1]:.5f}  "
                  f"val_loss={vloss:.5f}  val_pcorr_resid={vpcorr:+.4f}  "
                  f"best={best_val_pcorr:+.4f}@{best_epoch}  "
                  f"lr={opt.param_groups[0]['lr']:.2e}  bad={bad_epochs}",
                  flush=True)

        if bad_epochs >= patience:
            if verbose:
                print(f"[early-stop] val_pcorr_resid plateau at epoch {epoch+1} "
                      f"(no improvement for {patience} epochs)", flush=True)
            break

    elapsed = time.time() - t0
    return {
        "best_state_dict": best_state,
        "best_val_pcorr_resid": best_val_pcorr,
        "best_epoch": best_epoch,
        "train_loss_hist": train_loss_hist,
        "val_loss_hist": val_loss_hist,
        "val_pcorr_resid_hist": val_pcorr_hist,
        "epochs_run": len(train_loss_hist),
        "wall_clock_s": elapsed,
        "warmup_steps": warmup_steps,
        "total_steps_planned": total_steps,
    }
