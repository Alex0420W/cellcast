"""Non-learned baselines for the M3 first-run comparison.

StratifiedMeanBaseline: for each (cell_line, dose) stratum, predict the mean
LFC vector across training drugs in that stratum. Strongest "drug-agnostic"
baseline — beating it means the model learned something about the drug.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

PARQUET = os.path.expanduser("~/cellcast/data/sciplex/processed/cellcast_v0.parquet")
SPLITS_JSON = os.path.expanduser("~/cellcast/data/sciplex/processed/splits.json")
BASELINE_NPZ = os.path.expanduser("~/cellcast/results/baseline_predictions.npz")


class StratifiedMeanBaseline:
    """Predict the per-(cell_line, dose) mean LFC vector across training drugs."""

    def __init__(self) -> None:
        self.stratum_mean: dict[tuple[str, float], np.ndarray] = {}
        self.G: int | None = None

    def fit(self, train_df: pd.DataFrame) -> "StratifiedMeanBaseline":
        for (cl, dose), group in train_df.groupby(["cell_line", "dose_nM"]):
            mat = np.stack([np.asarray(v, dtype=np.float32)
                            for v in group["label_lfc_vector"]])  # [n_train_drugs_in_stratum, G]
            self.stratum_mean[(cl, float(dose))] = mat.mean(axis=0)
        Gs = {v.shape[0] for v in self.stratum_mean.values()}
        assert len(Gs) == 1, f"inconsistent G across strata: {Gs}"
        self.G = Gs.pop()
        return self

    def predict(self, test_df: pd.DataFrame) -> np.ndarray:
        if self.G is None:
            raise RuntimeError("fit() before predict()")
        preds = np.empty((len(test_df), self.G), dtype=np.float32)
        for i, (_, r) in enumerate(test_df.iterrows()):
            preds[i] = self.stratum_mean[(r["cell_line"], float(r["dose_nM"]))]
        return preds


def fit_and_save() -> dict:
    import json
    df = pd.read_parquet(PARQUET)
    split = json.loads(Path(SPLITS_JSON).read_text())
    train_df = df[df["drug_name"].isin(split["train_drugs"])].reset_index(drop=True)
    test_df = df[df["drug_name"].isin(split["test_drugs"])].reset_index(drop=True)

    bl = StratifiedMeanBaseline().fit(train_df)
    preds = bl.predict(test_df)
    targets = np.stack([np.asarray(v, dtype=np.float32)
                        for v in test_df["label_lfc_vector"]])

    Path(BASELINE_NPZ).parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        BASELINE_NPZ,
        preds=preds,
        targets=targets,
        condition_ids=test_df["condition_id"].to_numpy(),
        cell_lines=test_df["cell_line"].to_numpy(),
        drug_names=test_df["drug_name"].to_numpy(),
        dose_nM=test_df["dose_nM"].to_numpy(),
    )
    return {
        "n_train_conditions": len(train_df),
        "n_test_conditions": len(test_df),
        "n_strata": len(bl.stratum_mean),
        "G": bl.G,
        "out": BASELINE_NPZ,
    }


if __name__ == "__main__":
    r = fit_and_save()
    print(r)
