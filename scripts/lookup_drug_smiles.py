"""
PubChem PUG REST lookup for Sci-Plex drug names -> SMILES.

For each unique non-control value of obs['perturbation']:
  1. name -> CIDs (PUG REST /compound/name/{name}/cids/JSON)
  2. CID -> CanonicalSMILES + IsomericSMILES (PUG REST /compound/cid/{cid}/property/.../JSON)
  3. On miss, try (a) parenthetical-stripped name, (b) parenthetical content, (c) /synonyms/JSON fallback.

Rate-limited to <= 5 req/sec (PubChem unauthenticated limit). Writes to
~/cellcast/data/sciplex/drug_smiles.csv with one row per drug.
"""
from __future__ import annotations

import csv
import os
import re
import sys
import time
from dataclasses import dataclass, asdict
from urllib.parse import quote

import requests
import scanpy as sc

PUG = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"
MIN_INTERVAL = 0.21   # ~4.8 req/sec, leaves margin under 5/sec
TIMEOUT = 15

H5AD = os.path.expanduser("~/data/sciplex/SrivatsanTrapnell2020_sciplex3.h5ad")
OUT_CSV = os.path.expanduser("~/cellcast/data/sciplex/drug_smiles.csv")

_last_call = [0.0]


def _throttle() -> None:
    elapsed = time.time() - _last_call[0]
    if elapsed < MIN_INTERVAL:
        time.sleep(MIN_INTERVAL - elapsed)
    _last_call[0] = time.time()


def _get(url: str, *, expect_404_ok: bool = True) -> tuple[int, dict | None]:
    """GET with throttle; returns (status, json-or-None)."""
    _throttle()
    try:
        r = requests.get(url, timeout=TIMEOUT)
    except requests.RequestException as e:
        return -1, {"error": str(e)}
    if r.status_code == 200:
        try:
            return 200, r.json()
        except ValueError:
            return 200, None
    if r.status_code == 404 and expect_404_ok:
        return 404, None
    return r.status_code, None


def name_to_cids(name: str) -> list[int]:
    url = f"{PUG}/compound/name/{quote(name, safe='')}/cids/JSON"
    code, data = _get(url)
    if code != 200 or not data:
        return []
    return list(data.get("IdentifierList", {}).get("CID", []))


def cid_to_props(cid: int) -> dict | None:
    # PubChem 2025+ renames properties: IsomericSMILES -> SMILES (with stereo),
    # CanonicalSMILES -> ConnectivitySMILES (no stereo). Request both new names.
    url = f"{PUG}/compound/cid/{cid}/property/SMILES,ConnectivitySMILES,MolecularFormula,MolecularWeight/JSON"
    code, data = _get(url)
    if code != 200 or not data:
        return None
    props = data.get("PropertyTable", {}).get("Properties", [])
    return props[0] if props else None


def synonym_to_cids(name: str) -> list[int]:
    """Last-resort: ask PubChem's substance-name table via the autocomplete-like 'synonyms' query.
    Uses /compound/name/{name}/synonyms first to confirm a match exists; if 404, returns []."""
    url = f"{PUG}/compound/name/{quote(name, safe='')}/synonyms/JSON"
    code, data = _get(url)
    if code != 200 or not data:
        return []
    info = data.get("InformationList", {}).get("Information", [])
    cids: list[int] = []
    for item in info:
        cid = item.get("CID")
        if cid is not None:
            cids.append(cid)
    return cids


# Cleaning patterns for retries
PAREN_RE = re.compile(r"\s*\([^)]*\)\s*")
TRAILING_SALT_RE = re.compile(
    r"\s+("
    r"HCl|hydrochloride|Sodium|sodium|Potassium|potassium|Calcium|calcium|"
    r"Mesylate|mesylate|Maleate|maleate|Tosylate|tosylate|Acetate|acetate|"
    r"Bromide|bromide|Sulfate|sulfate|Phosphate|phosphate|Citrate|citrate|"
    r"Dihydrochloride|dihydrochloride|Trihydrate|trihydrate"
    r")\s*$"
)


def _candidates_for(name: str) -> list[str]:
    """Generate ordered name variants to try."""
    seen: set[str] = set()
    out: list[str] = []

    def add(s: str) -> None:
        s = s.strip()
        if s and s not in seen:
            seen.add(s)
            out.append(s)

    add(name)
    # strip parentheticals
    stripped = PAREN_RE.sub(" ", name).strip()
    add(stripped)
    # parenthetical content (often a code name like 'INCB018424')
    m = re.findall(r"\(([^)]*)\)", name)
    for inner in m:
        add(inner.strip())
    # strip trailing salt token (after stripping parens)
    for base in (stripped, name):
        salt_stripped = TRAILING_SALT_RE.sub("", base).strip()
        add(salt_stripped)
    return out


@dataclass
class Row:
    drug_name: str
    pubchem_cid: str          # primary CID we picked (or "")
    canonical_smiles: str
    isomeric_smiles: str
    lookup_status: str        # ok / ok_via_variant / ok_multi_cid / failed
    notes: str                # diagnostic detail


def lookup(name: str) -> Row:
    cands = _candidates_for(name)
    last_err = ""
    for i, cand in enumerate(cands):
        cids = name_to_cids(cand)
        if not cids:
            # try synonyms endpoint as a softer fallback for this candidate
            cids = synonym_to_cids(cand) if i == 0 else []
        if not cids:
            last_err = f"no CID for variant '{cand}'"
            continue

        primary = cids[0]
        props = cid_to_props(primary)
        if not props:
            last_err = f"CID {primary} matched for '{cand}' but property fetch returned nothing"
            continue

        # New PubChem schema: 'SMILES' is the stereo-aware canonical, 'ConnectivitySMILES' is the achiral form.
        # Map to our CSV columns to preserve the old semantics:
        #   canonical_smiles  <- ConnectivitySMILES (achiral)
        #   isomeric_smiles   <- SMILES (full info, what we feed to the model)
        isomeric = props.get("SMILES", "") or props.get("IsomericSMILES", "") or ""
        canonical = (
            props.get("ConnectivitySMILES", "")
            or props.get("CanonicalSMILES", "")
            or isomeric  # fall back to isomeric if PubChem only returns the full form
        )

        # Build status + note
        multi = len(cids) > 1
        if i == 0 and not multi:
            status = "ok"
            note = ""
        elif i == 0 and multi:
            status = "ok_multi_cid"
            note = f"matched {len(cids)} CIDs; took first. all={cids[:10]}"
        else:
            status = "ok_via_variant"
            note = f"matched via variant '{cand}'"
            if multi:
                status = "ok_multi_cid"
                note += f"; multi-CID, took first. all={cids[:10]}"
        return Row(
            drug_name=name,
            pubchem_cid=str(primary),
            canonical_smiles=canonical,
            isomeric_smiles=isomeric,
            lookup_status=status,
            notes=note,
        )

    return Row(
        drug_name=name,
        pubchem_cid="",
        canonical_smiles="",
        isomeric_smiles="",
        lookup_status="failed",
        notes=f"tried {len(cands)} variants; last_err={last_err}",
    )


def main() -> int:
    print(f"[1/4] loading drug names from {H5AD}", flush=True)
    a = sc.read_h5ad(H5AD, backed="r")
    pert = a.obs["perturbation"].astype(str)
    pert = pert[pert.notna() & (pert != "nan")]
    unique = sorted(set(pert.unique()) - {"control"})
    print(f"  unique non-control drug names: {len(unique)}")

    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)

    # Resume support: skip drugs already in CSV
    existing: dict[str, Row] = {}
    if os.path.exists(OUT_CSV):
        with open(OUT_CSV, newline="") as f:
            for r in csv.DictReader(f):
                existing[r["drug_name"]] = Row(**{k: r.get(k, "") for k in
                                                  ("drug_name","pubchem_cid","canonical_smiles","isomeric_smiles","lookup_status","notes")})
        print(f"  resume: {len(existing)} cached rows in {OUT_CSV}")

    todo = [n for n in unique if n not in existing]
    print(f"[2/4] querying PubChem for {len(todo)} drugs (rate <= 5/sec)", flush=True)

    rows: list[Row] = list(existing.values())
    t0 = time.time()
    for k, name in enumerate(todo, 1):
        r = lookup(name)
        rows.append(r)
        if k % 10 == 0 or k == len(todo):
            elapsed = time.time() - t0
            rate = k / elapsed if elapsed else 0
            print(f"  {k}/{len(todo)}  ({rate:.1f} req/s effective)  last='{name}' -> {r.lookup_status}", flush=True)

    rows.sort(key=lambda r: r.drug_name.lower())
    fields = ["drug_name", "pubchem_cid", "canonical_smiles", "isomeric_smiles", "lookup_status", "notes"]
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(asdict(r))
    print(f"[3/4] wrote {OUT_CSV}  ({len(rows)} rows)")

    # Summary
    from collections import Counter
    counts = Counter(r.lookup_status for r in rows)
    print("\n[4/4] summary")
    for k, v in sorted(counts.items()):
        print(f"  {k}: {v}")
    failed = [r.drug_name for r in rows if r.lookup_status == "failed"]
    multi = [r.drug_name for r in rows if r.lookup_status == "ok_multi_cid"]
    print(f"\nfailed ({len(failed)}): {failed}")
    print(f"\nmulti-CID ({len(multi)}):")
    for r in rows:
        if r.lookup_status == "ok_multi_cid":
            print(f"  {r.drug_name}  cid={r.pubchem_cid}  note={r.notes}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
