"""Tests for the P6 Morgan-fingerprint MLP baseline.

Specifically tests the three properties that matter for the fairness of the
M3-vs-P6 comparison:
  1. Forward shape: model output is [B, num_HVGs].
  2. No leakage: stratum means are derived solely from train_df; if we pass
     test_df rows through residual_target(test_df), the result reflects only
     train-derived means (test drugs do not contribute to the means).
  3. Reconstruction round-trip: full LFC = residual + stratum_mean is the
     identity map for residual=true_LFC-mean (ie. reconstruct(residual_target(df), df) == true_LFC).
"""
from __future__ import annotations

import os, sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

sys.path.insert(0, os.path.expanduser("~/cellcast"))

from src.models.fingerprint_mlp import (
    INPUT_DIM, MORGAN_NBITS, CELL_LINES, DOSE_NMS,
    FingerprintMLP, StratumMean, encode_features, morgan_fp,
)


# ---- Fixtures: a tiny synthetic LFC table ----------------------------------- #
def _toy_df(drugs: list[str], seed: int = 0) -> pd.DataFrame:
    """Build a [n_drugs * 3 * 4 = 12 * n_drugs] toy LFC table on a fake gene panel."""
    rng = np.random.default_rng(seed)
    G = 16
    rows = []
    for d in drugs:
        for cl in CELL_LINES:
            for dose in DOSE_NMS:
                rows.append({
                    "drug_name": d,
                    "cell_line": cl,
                    "dose_nM": dose,
                    "label_lfc_vector": rng.normal(0.0, 1.0, size=G).astype(np.float32),
                    "condition_id": f"{d}_{cl}_{int(dose)}",
                })
    return pd.DataFrame(rows)


# ---- Test 1: forward shape -------------------------------------------------- #
def test_forward_output_shape_matches_num_HVGs():
    G = 1234
    model = FingerprintMLP(input_dim=INPUT_DIM, num_HVGs=G).eval()
    x = torch.zeros(7, INPUT_DIM)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (7, G), f"unexpected forward shape: {tuple(out.shape)}"


# ---- Test 2: no leakage (test drugs do not influence stratum means) -------- #
def test_stratum_mean_no_leakage_from_test_drugs():
    train_drugs = ["DrugA", "DrugB", "DrugC"]
    test_drugs = ["DrugX", "DrugY"]  # held out
    full = _toy_df(train_drugs + test_drugs, seed=42)

    train_df = full[full["drug_name"].isin(train_drugs)].reset_index(drop=True)
    test_df = full[full["drug_name"].isin(test_drugs)].reset_index(drop=True)

    sm = StratumMean.fit(train_df)

    # Recompute the stratum means manually using ONLY train_df and confirm equality.
    for (cl, dose), expected_group in train_df.groupby(["cell_line", "dose_nM"]):
        expected_mean = np.stack([np.asarray(v, dtype=np.float32)
                                  for v in expected_group["label_lfc_vector"]]).mean(axis=0)
        np.testing.assert_allclose(sm.means[(cl, float(dose))], expected_mean,
                                   err_msg=f"stratum mean drift at ({cl},{dose})")

    # Recompute means including test_df and confirm they DIFFER from sm.means
    # at every stratum (otherwise the train-only constraint is undetectable).
    for (cl, dose), grp in full.groupby(["cell_line", "dose_nM"]):
        contaminated = np.stack([np.asarray(v, dtype=np.float32)
                                 for v in grp["label_lfc_vector"]]).mean(axis=0)
        # Should differ from train-only mean
        assert not np.allclose(sm.means[(cl, float(dose))], contaminated), (
            f"contaminated mean equals train-only mean at ({cl},{dose}); "
            "either our toy data accidentally collided or leakage isn't detectable"
        )

    # Residual target on test_df uses ONLY train-derived means.
    residuals = sm.residual_target(test_df)
    assert residuals.shape == (len(test_df), sm.G)


# ---- Test 3: reconstruction round-trip ------------------------------------- #
def test_reconstruct_round_trip_equals_true_lfc():
    train_drugs = ["DrugA", "DrugB", "DrugC", "DrugD"]
    test_drugs = ["DrugX"]
    full = _toy_df(train_drugs + test_drugs, seed=7)
    train_df = full[full["drug_name"].isin(train_drugs)].reset_index(drop=True)
    test_df = full[full["drug_name"].isin(test_drugs)].reset_index(drop=True)
    sm = StratumMean.fit(train_df)

    true_lfc = np.stack([np.asarray(v, dtype=np.float32) for v in test_df["label_lfc_vector"]])
    residual = sm.residual_target(test_df)
    reconstructed = sm.reconstruct(residual, test_df)

    assert reconstructed.shape == true_lfc.shape, "reconstruct changed shape"
    np.testing.assert_allclose(reconstructed, true_lfc, atol=1e-6,
                               err_msg="reconstruct(residual_target(df), df) != true_LFC")


# ---- Test 4 (bonus): fingerprint determinism + shape ----------------------- #
def test_morgan_fp_deterministic_and_correct_shape():
    fp1 = morgan_fp("CCO")  # ethanol
    fp2 = morgan_fp("CCO")
    np.testing.assert_array_equal(fp1, fp2)
    assert fp1.shape == (MORGAN_NBITS,)
    assert fp1.dtype == np.uint8
    assert int(fp1.sum()) > 0  # ethanol has *some* bits set


# ---- Test 5 (bonus): feature encoder one-hot positions --------------------- #
def test_encode_features_onehot_layout():
    fp_table = {"DrugA": np.zeros(MORGAN_NBITS, dtype=np.uint8)}
    fp_table["DrugA"][0] = 1
    fp_table["DrugA"][2047] = 1
    X = encode_features(["DrugA"], ["K562"], [1000.0], fp_table)
    assert X.shape == (1, INPUT_DIM)
    # Morgan slice
    assert X[0, 0] == 1.0 and X[0, 2047] == 1.0
    # K562 is index 1 in CELL_LINES = ('A549', 'K562', 'MCF7')
    assert X[0, MORGAN_NBITS + 1] == 1.0
    assert X[0, MORGAN_NBITS + 0] == 0.0
    assert X[0, MORGAN_NBITS + 2] == 0.0
    # 1000 nM is index 2 in DOSE_NMS = (10, 100, 1000, 10000)
    base = MORGAN_NBITS + len(CELL_LINES)
    assert X[0, base + 2] == 1.0
    assert X[0, base + 0] == 0.0 and X[0, base + 1] == 0.0 and X[0, base + 3] == 0.0
