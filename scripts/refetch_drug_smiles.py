"""One-off fixer: refetch SMILES for rows in drug_smiles.csv that have empty smiles
but a non-empty CID. Also retries the originally-failed 'Glesatinib?(MGCD265)' row
using the stripped name 'Glesatinib'.

Run after editing lookup_drug_smiles.py to use the new PubChem property names
(SMILES, ConnectivitySMILES).
"""
from __future__ import annotations

import csv
import os
import sys
import time

# Import the (now-corrected) helpers from the main script
sys.path.insert(0, os.path.dirname(__file__))
from lookup_drug_smiles import cid_to_props, name_to_cids  # noqa: E402

OUT_CSV = os.path.expanduser("~/cellcast/data/sciplex/drug_smiles.csv")


def main() -> int:
    rows = list(csv.DictReader(open(OUT_CSV)))
    print(f"loaded {len(rows)} rows from {OUT_CSV}")

    to_refetch = [r for r in rows if r["pubchem_cid"] and not r["isomeric_smiles"]]
    to_retry_failed = [r for r in rows if r["lookup_status"] == "failed"]
    print(f"  refetch CID->SMILES for {len(to_refetch)} rows")
    print(f"  retry failed name lookups: {len(to_retry_failed)} rows")

    t0 = time.time()
    for k, r in enumerate(to_refetch, 1):
        cid = int(r["pubchem_cid"])
        props = cid_to_props(cid)
        if not props:
            r["notes"] = (r["notes"] + " | refetch_failed").strip(" |")
            continue
        iso = props.get("SMILES", "") or ""
        canon = props.get("ConnectivitySMILES", "") or iso
        r["isomeric_smiles"] = iso
        r["canonical_smiles"] = canon
        if k % 20 == 0 or k == len(to_refetch):
            rate = k / (time.time() - t0)
            print(f"  refetched {k}/{len(to_refetch)}  ({rate:.1f} req/s)", flush=True)

    # Retry failed-by-name rows
    for r in to_retry_failed:
        name = r["drug_name"]
        # Aggressive cleanup for known scperturb glyph corruption
        clean = (name.replace("?", " ")
                     .replace(" ", " ")
                     .replace("  ", " ")
                     .strip())
        # Use only the first word (drug name proper) if cleanup didn't help
        candidates = [clean, clean.split("(")[0].strip(), clean.split()[0]]
        seen = set()
        for cand in candidates:
            if not cand or cand in seen:
                continue
            seen.add(cand)
            cids = name_to_cids(cand)
            if cids:
                props = cid_to_props(cids[0])
                if props:
                    iso = props.get("SMILES", "") or ""
                    canon = props.get("ConnectivitySMILES", "") or iso
                    r["pubchem_cid"] = str(cids[0])
                    r["isomeric_smiles"] = iso
                    r["canonical_smiles"] = canon
                    r["lookup_status"] = "ok_via_variant"
                    r["notes"] = f"retry-after-fix: matched via variant '{cand}'"
                    print(f"  retried {name!r} -> CID {cids[0]} (variant {cand!r})")
                    break
        else:
            print(f"  still failed: {name!r}")

    # Write back
    fields = ["drug_name", "pubchem_cid", "canonical_smiles", "isomeric_smiles", "lookup_status", "notes"]
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})

    # Summary
    from collections import Counter
    counts = Counter(r["lookup_status"] for r in rows)
    empty_iso = sum(1 for r in rows if not r["isomeric_smiles"])
    empty_canon = sum(1 for r in rows if not r["canonical_smiles"])
    print(f"\nresulting status counts: {dict(counts)}")
    print(f"rows with empty isomeric_smiles:  {empty_iso}")
    print(f"rows with empty canonical_smiles: {empty_canon}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
