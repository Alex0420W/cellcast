"""Join-integrity test: every unique non-control drug name in Sci-Plex obs['perturbation']
must appear as a row in data/sciplex/drug_smiles.csv with a parseable SMILES.

Guards against silent drug-loss during milestone 3 preprocessing — if a drug name
fails to join (string-exact, including quirks like 'Glesatinib?(MGCD265)' which
contains a literal U+003F glyph from scPerturb harmonization), cells under that
drug would either be dropped or get no SMILES condition.
"""
from __future__ import annotations

import csv
import os

import pytest

H5AD = os.path.expanduser("~/data/sciplex/SrivatsanTrapnell2020_sciplex3.h5ad")
CSV = os.path.expanduser("~/cellcast/data/sciplex/drug_smiles.csv")


@pytest.fixture(scope="module")
def obs_drugs() -> set[str]:
    import scanpy as sc
    a = sc.read_h5ad(H5AD, backed="r")
    pert = a.obs["perturbation"].astype(str)
    return set(pert.unique()) - {"control", "nan"}


@pytest.fixture(scope="module")
def csv_rows() -> list[dict]:
    with open(CSV, newline="") as f:
        return list(csv.DictReader(f))


def test_csv_exists_and_nonempty(csv_rows):
    assert len(csv_rows) > 0, f"{CSV} is empty"


def test_every_obs_drug_has_csv_row(obs_drugs, csv_rows):
    csv_names = {r["drug_name"] for r in csv_rows}
    missing = obs_drugs - csv_names
    assert not missing, (
        f"{len(missing)} drugs in obs['perturbation'] have no row in drug_smiles.csv:\n"
        f"  {sorted(missing)[:20]}"
    )


def test_no_extra_drugs_in_csv(obs_drugs, csv_rows):
    csv_names = {r["drug_name"] for r in csv_rows}
    extra = csv_names - obs_drugs
    assert not extra, (
        f"{len(extra)} drug rows in drug_smiles.csv don't exist in obs['perturbation']:\n"
        f"  {sorted(extra)[:20]}"
    )


def test_every_row_has_nonempty_smiles(csv_rows):
    bad = [r["drug_name"] for r in csv_rows if not r["isomeric_smiles"]]
    assert not bad, f"{len(bad)} rows have empty isomeric_smiles: {bad[:10]}"


def test_glesatinib_glyph_preserved(obs_drugs, csv_rows):
    """The scPerturb name 'Glesatinib?(MGCD265)' contains a literal U+003F.
    If anyone 'cleans' the name during preprocessing the join will break silently."""
    target = "Glesatinib?(MGCD265)"
    assert target in obs_drugs, (
        f"obs no longer contains the glyph-corrupted name {target!r}; "
        f"check that the AnnData wasn't re-harmonized"
    )
    csv_names = {r["drug_name"] for r in csv_rows}
    assert target in csv_names, (
        f"drug_smiles.csv is missing the exact name {target!r}; "
        f"do not strip the '?' character from drug names"
    )


def test_all_smiles_parse_with_rdkit(csv_rows):
    """Belt-and-braces: ensure every SMILES is parseable so a downstream rdkit
    canonicalization or fingerprint step can't fail per-row."""
    from rdkit import Chem
    from rdkit import RDLogger
    RDLogger.DisableLog("rdApp.*")

    bad = []
    for r in csv_rows:
        if Chem.MolFromSmiles(r["isomeric_smiles"]) is None:
            bad.append((r["drug_name"], r["isomeric_smiles"][:60]))
    assert not bad, f"{len(bad)} unparseable SMILES: {bad[:5]}"
