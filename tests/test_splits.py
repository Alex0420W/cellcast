"""Tests for the drug-level train/test split."""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

ROOT = Path(os.path.expanduser("~/cellcast"))
sys.path.insert(0, str(ROOT))

from src.splits import SEED, load_split, make_split  # noqa: E402


@pytest.fixture(scope="module")
def saved_split():
    return load_split()


def test_split_reproducible(saved_split):
    """Re-running make_split with the same seed must produce the saved split exactly."""
    fresh = make_split(seed=SEED)
    assert fresh["train_drugs"] == saved_split["train_drugs"]
    assert fresh["test_drugs"] == saved_split["test_drugs"]


def test_drugs_disjoint(saved_split):
    overlap = set(saved_split["train_drugs"]) & set(saved_split["test_drugs"])
    assert not overlap, f"train/test drug overlap: {sorted(overlap)[:5]}"


def test_drug_counts(saved_split):
    assert saved_split["n_drugs_total"] == 188
    assert saved_split["n_train_drugs"] + saved_split["n_test_drugs"] == 188
    # Conditions: 12 per drug
    assert saved_split["n_train_conditions"] == saved_split["n_train_drugs"] * 12
    assert saved_split["n_test_conditions"] == saved_split["n_test_drugs"] * 12


def test_every_pathway_in_both_splits(saved_split):
    d2p = saved_split["drug_to_pathway"]
    train_pathways = {d2p[d] for d in saved_split["train_drugs"]}
    test_pathways = {d2p[d] for d in saved_split["test_drugs"]}
    all_pathways = set(d2p.values())
    missing_train = all_pathways - train_pathways
    missing_test = all_pathways - test_pathways
    assert not missing_train, f"pathways missing from train: {sorted(missing_train)}"
    assert not missing_test, f"pathways missing from test: {sorted(missing_test)}"


def test_test_drug_count_in_range(saved_split):
    # 20% of 188 ~ 37.6; per-pathway rounding pushes it slightly. Allow 35–42.
    n = saved_split["n_test_drugs"]
    assert 35 <= n <= 42, f"test drug count {n} outside expected range"


def test_control_not_in_splits(saved_split):
    assert "control" not in saved_split["train_drugs"]
    assert "control" not in saved_split["test_drugs"]
