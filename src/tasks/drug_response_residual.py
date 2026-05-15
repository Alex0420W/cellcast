"""DrugResponseResidualTask — M4B.1 fork of drug_response_vector.

Same model, prompt, dose tokens, head shape, and forward path. The only
structural change is the label tensor:

  full LFC                                (drug_response_vector)
    becomes
  residual = true LFC - stratum_mean      (drug_response_residual)

where `stratum_mean` is the per-(cell_line, dose) mean of full LFC computed
on TRAIN drugs only (no leakage). Loss is MSE on the residual.

At inference time the full-LFC prediction is reconstructed as:
    full_pred = model_residual_pred + stratum_mean

Reconstruction lives in the eval script — keeps training code dumb about
what the head is producing. See scripts/evaluate_residual.py.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

# Re-export everything the train/eval scripts need from the parent task.
# The forward path, tokenizer expansion, and freezing logic are identical.
from src.tasks.drug_response_vector import (  # noqa: F401
    CELLCAST_TOKENIZER_PATH,
    DOSE_TOKENS,
    SLICED_PRED_KEY,
    build_sample_dict,
    configure_frozen_backbone_with_trainable_dose_rows,
    expand_tokenizer_and_embeddings,
    init_dose_token_embeddings,
    load_or_expand_tokenizer,
    per_gene_pearson_macro,
    per_gene_spearman_macro,
    process_model_output,
    top_k_deg_direction_accuracy,
)


# ---- Stratum-mean target (no leakage) --------------------------------------- #
@dataclass
class StratumMean:
    """Per-(cell_line, dose) mean LFC vector. Fit on TRAIN drugs only.

    Identical computation to src.models.fingerprint_mlp.StratumMean and to
    src.models.baselines.StratifiedMeanBaseline. Duplicated here so the
    residual-task code is self-contained.
    """
    means: dict[tuple[str, float], np.ndarray]   # (cl, dose) -> [G]
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

    def lookup(self, cell_line: str, dose_nM: float) -> np.ndarray:
        return self.means[(cell_line, float(dose_nM))]

    def lookup_for_rows(self, cell_lines, doses_nM) -> np.ndarray:
        cls_l = list(cell_lines)
        doses_l = [float(d) for d in doses_nM]
        if not cls_l:
            return np.zeros((0, self.G), dtype=np.float32)
        return np.stack([self.means[(cls_l[i], doses_l[i])] for i in range(len(cls_l))])

    def residual_for_rows(self, df: pd.DataFrame) -> np.ndarray:
        """Compute residual = true_LFC - stratum_mean per row of df."""
        if len(df) == 0:
            return np.zeros((0, self.G), dtype=np.float32)
        true = np.stack([np.asarray(v, dtype=np.float32) for v in df["label_lfc_vector"]])
        sm = self.lookup_for_rows(df["cell_line"], df["dose_nM"])
        return true - sm

    def reconstruct(self, residual_pred: np.ndarray, df: pd.DataFrame) -> np.ndarray:
        """Add stratum_mean back to a residual prediction to recover full LFC."""
        sm = self.lookup_for_rows(df["cell_line"], df["dose_nM"])
        return residual_pred + sm


def attach_residual_labels(df: pd.DataFrame, sm: StratumMean) -> pd.DataFrame:
    """Return a copy of df with `label_lfc_vector` replaced by the residual.

    The shape and dtype of the column are preserved (numpy arrays of length G).
    Other columns (smiles, dose_bin, ranked_genes, condition_id, etc.) are
    untouched, so the existing CellCastDataset can consume the result without
    modification — it'll write the residual tensor into LABELS_SCALARS_VALUES.
    """
    out = df.copy()
    residual = sm.residual_for_rows(df).astype(np.float32)
    # store as object-dtype column of 1-D arrays (matches the existing parquet
    # convention for `label_lfc_vector`)
    out["label_lfc_vector"] = list(residual)
    out["full_lfc_vector"] = list(np.stack([np.asarray(v, dtype=np.float32)
                                            for v in df["label_lfc_vector"]]))
    return out
