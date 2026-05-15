"""Tests for the 3B preprocessing output."""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROC = Path(os.path.expanduser("~/cellcast/data/sciplex/processed"))
PARQUET = PROC / "cellcast_v0.parquet"
HVG = PROC / "hvg_genes.txt"

DOSE_BINS = {"DOSE_10nM", "DOSE_100nM", "DOSE_1000nM", "DOSE_10000nM"}


@pytest.fixture(scope="module")
def df() -> pd.DataFrame:
    assert PARQUET.exists(), f"parquet not found at {PARQUET}"
    return pd.read_parquet(PARQUET)


@pytest.fixture(scope="module")
def hvg_list() -> list[str]:
    assert HVG.exists(), f"HVG file missing at {HVG}"
    with open(HVG) as f:
        lines = [ln.rstrip("\n") for ln in f if ln.strip()]
    return lines


def test_parquet_row_count(df):
    assert len(df) == 2256, f"expected 2256 rows, got {len(df)}"


def test_no_nan_in_lfc(df):
    bad = []
    for i, v in enumerate(df["label_lfc_vector"]):
        arr = np.asarray(v)
        if np.isnan(arr).any():
            bad.append(i)
    assert not bad, f"NaN in label_lfc_vector for {len(bad)} rows: {bad[:5]}"


def test_no_inf_in_lfc(df):
    bad = []
    for i, v in enumerate(df["label_lfc_vector"]):
        arr = np.asarray(v)
        if not np.isfinite(arr).all():
            bad.append(i)
    assert not bad, f"non-finite values in label_lfc_vector: {bad[:5]}"


def test_dose_bin_in_valid_set(df):
    bad = set(df["dose_bin"].unique()) - DOSE_BINS
    assert not bad, f"unexpected dose_bin values: {bad}"


def test_dose_bin_matches_dose_nM(df):
    """Sanity: the bin string and the numeric dose agree."""
    expected = {10.0: "DOSE_10nM", 100.0: "DOSE_100nM",
                1000.0: "DOSE_1000nM", 10000.0: "DOSE_10000nM"}
    for _, r in df.iterrows():
        assert expected[r["dose_nM"]] == r["dose_bin"], (
            f"dose mismatch for {r['condition_id']}: dose_nM={r['dose_nM']} dose_bin={r['dose_bin']}"
        )


def test_every_smiles_rdkit_parses(df):
    from rdkit import Chem
    from rdkit import RDLogger
    RDLogger.DisableLog("rdApp.*")
    bad = []
    for _, r in df.iterrows():
        if Chem.MolFromSmiles(r["smiles"]) is None:
            bad.append((r["drug_name"], r["smiles"][:60]))
    assert not bad, f"{len(bad)} unparseable SMILES: {bad[:5]}"


def test_input_gene_ranked_list_length(df):
    bad = []
    for i, v in enumerate(df["input_gene_ranked_list"]):
        n = len(list(v))
        if n != 1294:
            bad.append((i, n))
    assert not bad, f"rows with wrong gene-rank length: {bad[:5]}"


def test_input_gene_ranked_list_per_cell_line_consistency(df):
    """Within a cell line every row should share the same ranked list (it's a function of the cell line's control)."""
    for cl, group in df.groupby("cell_line"):
        signatures = {tuple(v) for v in group["input_gene_ranked_list"]}
        assert len(signatures) == 1, f"{cl} has {len(signatures)} distinct gene-rank lists across rows"


def test_hvg_file_exists(hvg_list):
    assert len(hvg_list) > 0, "HVG file is empty"


def test_hvg_alphabetical(hvg_list):
    assert hvg_list == sorted(hvg_list), "HVG list is not alphabetical"


def test_lfc_vector_length_matches_hvg(df, hvg_list):
    expected = len(hvg_list)
    bad = []
    for i, v in enumerate(df["label_lfc_vector"]):
        n = len(np.asarray(v))
        if n != expected:
            bad.append((i, n))
    assert not bad, f"lfc length != n_HVG ({expected}) for rows: {bad[:5]}"


def test_drug_coverage_unchanged():
    """The M2 invariant: every non-control drug in obs[perturbation] has a SMILES row in drug_smiles.csv."""
    import scanpy as sc
    import csv
    a = sc.read_h5ad(os.path.expanduser("~/data/sciplex/SrivatsanTrapnell2020_sciplex3.h5ad"), backed="r")
    obs_drugs = set(a.obs["perturbation"].astype(str).unique()) - {"control", "nan"}
    csv_path = os.path.expanduser("~/cellcast/data/sciplex/drug_smiles.csv")
    csv_drugs = {r["drug_name"] for r in csv.DictReader(open(csv_path))}
    missing = obs_drugs - csv_drugs
    assert not missing, f"drug-coverage regression: missing {sorted(missing)[:5]}"


def test_parquet_columns_present(df):
    needed = {"condition_id", "cell_line", "drug_name", "dose_nM", "dose_bin",
              "smiles", "input_gene_ranked_list", "label_lfc_vector", "n_cells_aggregated"}
    missing = needed - set(df.columns)
    assert not missing, f"missing columns: {missing}"


def test_n_cells_aggregated_positive(df):
    assert (df["n_cells_aggregated"] > 0).all()


def test_condition_id_unique(df):
    n_dup = df["condition_id"].duplicated().sum()
    assert n_dup == 0, f"{n_dup} duplicate condition_ids"
