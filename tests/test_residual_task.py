"""Tests for the M4B.1 residual task module.

Three load-bearing properties:
  1. Residual computation: residual = full_lfc - stratum_mean has the right
     shape and the residual on TRAINING rows reproduces (full - per-stratum-mean)
     exactly per row.
  2. No leakage: stratum_mean fit on TRAIN-only does not move when test drugs
     are also present in the input df (i.e. the fit ignores anything not labeled
     train).
  3. Reconstruction round-trip: reconstruct(residual_for_rows(df)) == full_lfc
     exactly on every row (the inverse operation cancels exactly).

Plus two bonus tests for table-stakes behavior (attach_residual_labels
preserves shape; full_lfc_vector column is preserved unchanged).
"""
from __future__ import annotations

import os, sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.expanduser("~/cellcast"))

from src.tasks.drug_response_residual import StratumMean, attach_residual_labels


# ---- Fixture: tiny synthetic LFC table -------------------------------------- #
CELL_LINES = ("A549", "K562", "MCF7")
DOSE_NMS = (10.0, 100.0, 1000.0, 10000.0)


def _toy_df(drugs: list[str], n_genes: int = 24, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for d in drugs:
        for cl in CELL_LINES:
            for dose in DOSE_NMS:
                rows.append({
                    "drug_name": d,
                    "cell_line": cl,
                    "dose_nM": dose,
                    "label_lfc_vector": rng.normal(0.0, 1.0, size=n_genes).astype(np.float32),
                    "condition_id": f"{d}_{cl}_{int(dose)}",
                    "smiles": "CCO",
                    "dose_bin": f"DOSE_{int(dose)}nM",
                    "input_gene_ranked_list": [f"G{i}" for i in range(n_genes)],
                })
    return pd.DataFrame(rows)


# ---- Test 1: residual computation has correct shape + per-row arithmetic --- #
def test_residual_shape_and_per_row_arithmetic():
    drugs = ["DrugA", "DrugB", "DrugC"]
    df = _toy_df(drugs, n_genes=20, seed=1)
    sm = StratumMean.fit(df)

    residual = sm.residual_for_rows(df)
    assert residual.shape == (len(df), 20), f"unexpected residual shape: {residual.shape}"

    # Per-row check: residual[i] should equal true_lfc[i] - stratum_mean[(cl[i], dose[i])]
    for i in range(len(df)):
        r = df.iloc[i]
        expected = (np.asarray(r["label_lfc_vector"], dtype=np.float32)
                    - sm.means[(r["cell_line"], float(r["dose_nM"]))])
        np.testing.assert_allclose(residual[i], expected, rtol=1e-6, atol=1e-6,
                                   err_msg=f"row {i} residual mismatch")


# ---- Test 2: no leakage (test-drug rows ignored when fitting on train_df) -- #
def test_stratum_mean_train_only_no_leakage():
    train_drugs = ["DrugA", "DrugB", "DrugC"]
    test_drugs = ["DrugX", "DrugY"]   # held out
    full = _toy_df(train_drugs + test_drugs, n_genes=16, seed=42)

    train_df = full[full["drug_name"].isin(train_drugs)].reset_index(drop=True)
    sm = StratumMean.fit(train_df)

    # Manually recompute means using only train_df and confirm exact match.
    for (cl, dose), grp in train_df.groupby(["cell_line", "dose_nM"]):
        expected = np.stack([np.asarray(v, dtype=np.float32)
                             for v in grp["label_lfc_vector"]]).mean(axis=0)
        np.testing.assert_allclose(sm.means[(cl, float(dose))], expected,
                                   err_msg=f"stratum mean drift at ({cl}, {dose})")

    # Recompute including test_df and confirm the means DIFFER at every stratum
    # (otherwise the train-only constraint is undetectable on this fixture).
    for (cl, dose), grp in full.groupby(["cell_line", "dose_nM"]):
        contaminated = np.stack([np.asarray(v, dtype=np.float32)
                                 for v in grp["label_lfc_vector"]]).mean(axis=0)
        assert not np.allclose(sm.means[(cl, float(dose))], contaminated), (
            f"contaminated mean equals train-only mean at ({cl}, {dose}); "
            "fixture too sparse to detect leakage"
        )


# ---- Test 3: reconstruction round-trip ------------------------------------- #
def test_reconstruct_round_trip_exact_on_train_rows():
    drugs = ["DrugA", "DrugB", "DrugC", "DrugD"]
    df = _toy_df(drugs, n_genes=32, seed=7)
    sm = StratumMean.fit(df)

    true_lfc = np.stack([np.asarray(v, dtype=np.float32) for v in df["label_lfc_vector"]])
    residual = sm.residual_for_rows(df)
    reconstructed = sm.reconstruct(residual, df)

    assert reconstructed.shape == true_lfc.shape, "reconstruct changed shape"
    np.testing.assert_allclose(reconstructed, true_lfc, atol=1e-6,
                               err_msg="reconstruct(residual_for_rows(df), df) != true_LFC")


# ---- Test 4 (bonus): attach_residual_labels preserves the schema ----------- #
def test_attach_residual_labels_preserves_columns_and_swaps_label():
    drugs = ["DrugA", "DrugB"]
    df = _toy_df(drugs, n_genes=10, seed=3)
    sm = StratumMean.fit(df)

    out = attach_residual_labels(df, sm)
    assert len(out) == len(df), "row count changed"
    # Original columns still present
    for col in ("drug_name", "cell_line", "dose_nM", "smiles", "dose_bin",
                "input_gene_ranked_list", "condition_id"):
        assert col in out.columns
    # New full_lfc_vector column matches the original label_lfc_vector
    for i in range(len(df)):
        np.testing.assert_array_equal(
            np.asarray(out.iloc[i]["full_lfc_vector"], dtype=np.float32),
            np.asarray(df.iloc[i]["label_lfc_vector"], dtype=np.float32),
        )
    # label_lfc_vector is now the residual: equal to (full - stratum_mean)
    for i in range(len(out)):
        r_orig = df.iloc[i]
        expected_resid = (np.asarray(r_orig["label_lfc_vector"], dtype=np.float32)
                          - sm.means[(r_orig["cell_line"], float(r_orig["dose_nM"]))])
        np.testing.assert_allclose(
            np.asarray(out.iloc[i]["label_lfc_vector"], dtype=np.float32),
            expected_resid, atol=1e-6,
        )


# ---- Test 5 (bonus): empty val df still works ------------------------------ #
def test_residual_on_empty_df_returns_empty():
    drugs = ["DrugA"]
    df = _toy_df(drugs, n_genes=8, seed=5)
    sm = StratumMean.fit(df)
    empty = df.iloc[:0]
    out = sm.residual_for_rows(empty)
    assert out.shape == (0, 8)
