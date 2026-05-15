"""Train/test split for CellCast — drug-level holdout, stratified by pathway_level_1.

20% of the 188 drugs go to test, stratified so every pathway_level_1 is represented
in both train and test. Control conditions are excluded from both splits (controls
are the LFC baseline, not a target).
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import scanpy as sc

H5AD = os.path.expanduser("~/data/sciplex/SrivatsanTrapnell2020_sciplex3.h5ad")
SPLITS_PATH = Path(os.path.expanduser("~/cellcast/data/sciplex/processed/splits.json"))
TEST_FRAC = 0.20
SEED = 42


def drug_to_pathway_map() -> dict[str, str]:
    """Read obs from the Sci-Plex h5ad and return {drug_name -> pathway_level_1}.

    Each drug is annotated with exactly one pathway in scPerturb v1.4.
    """
    a = sc.read_h5ad(H5AD, backed="r")
    obs = a.obs[a.obs["cell_line"].notna()]
    mapping: dict[str, str] = {}
    for drug, sub in obs.groupby("perturbation", observed=True):
        drug = str(drug)
        if drug == "control":
            continue
        pathways = sub["pathway_level_1"].astype(str).unique().tolist()
        # The first value drives stratification; pathway is annotated identically
        # across all rows for a given drug.
        mapping[drug] = pathways[0]
    return mapping


def make_split(seed: int = SEED, test_frac: float = TEST_FRAC) -> dict:
    """Stratified holdout enforcing >=1 drug per pathway in BOTH splits.

    sklearn's StratifiedShuffleSplit doesn't guarantee per-class min counts
    on small classes (e.g. pathways with 2 drugs at 20% test rate).
    """
    drug_to_path = drug_to_pathway_map()
    rng = np.random.default_rng(seed)

    by_pathway: dict[str, list[str]] = {}
    for d, p in drug_to_path.items():
        by_pathway.setdefault(p, []).append(d)
    for p in by_pathway:
        by_pathway[p].sort()

    train_drugs: list[str] = []
    test_drugs: list[str] = []
    for p in sorted(by_pathway):
        bucket = by_pathway[p]
        n = len(bucket)
        # Round-half-up to whole drugs; clamp so both splits have >=1.
        n_test = int(round(n * test_frac))
        n_test = max(1, n_test)
        n_test = min(n_test, n - 1)        # always keep >=1 in train
        idx = rng.permutation(n)
        bucket_perm = [bucket[i] for i in idx]
        test_drugs.extend(bucket_perm[:n_test])
        train_drugs.extend(bucket_perm[n_test:])

    train_drugs.sort()
    test_drugs.sort()
    # 4 doses × 3 cell lines = 12 conditions per drug
    return {
        "seed": seed,
        "test_frac": test_frac,
        "stratification": "pathway_level_1, with >=1 drug per pathway in both splits",
        "n_drugs_total": len(drug_to_path),
        "n_train_drugs": len(train_drugs),
        "n_test_drugs": len(test_drugs),
        "n_train_conditions": len(train_drugs) * 12,
        "n_test_conditions": len(test_drugs) * 12,
        "train_drugs": train_drugs,
        "test_drugs": test_drugs,
        "drug_to_pathway": drug_to_path,
    }


def save_split(split: dict, path: Path = SPLITS_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(split, indent=2))


def load_split(path: Path = SPLITS_PATH) -> dict:
    return json.loads(path.read_text())


if __name__ == "__main__":
    split = make_split()
    save_split(split)
    print(f"wrote {SPLITS_PATH}")
    print(f"  seed={split['seed']}  test_frac={split['test_frac']}")
    print(f"  drugs: total={split['n_drugs_total']}  train={split['n_train_drugs']}  test={split['n_test_drugs']}")
    print(f"  conditions: train={split['n_train_conditions']}  test={split['n_test_conditions']}")
    # Per-pathway breakdown
    from collections import Counter
    train_pw = Counter(split["drug_to_pathway"][d] for d in split["train_drugs"])
    test_pw = Counter(split["drug_to_pathway"][d] for d in split["test_drugs"])
    print(f"\n  per-pathway split (train / test):")
    for p in sorted(set(train_pw) | set(test_pw)):
        print(f"    {p:>40s}: {train_pw.get(p, 0):>3} / {test_pw.get(p, 0):>3}")
