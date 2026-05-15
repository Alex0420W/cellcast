"""Resolve PubChem multi-CID rows in drug_smiles.csv automatically.

For each row with lookup_status == 'ok_multi_cid', fetch SMILES for every
candidate CID listed in the notes column, and prefer the candidate whose
isomeric SMILES does *not* contain '.' (the SMILES disconnection character
indicating a salt complex / mixture).

Rules:
  - exactly one candidate has no '.': switch to that candidate (if different).
  - all candidates have '.', or none do: leave current choice; flag in notes.
"""
from __future__ import annotations

import ast
import csv
import os
import re
import sys

sys.path.insert(0, os.path.dirname(__file__))
from lookup_drug_smiles import cid_to_props  # noqa: E402

OUT_CSV = os.path.expanduser("~/cellcast/data/sciplex/drug_smiles.csv")

# Notes look like: "matched 2 CIDs; took first. all=[66576993, 24776050]"
ALL_RE = re.compile(r"all=(\[[^\]]+\])")


def parse_all_cids(notes: str) -> list[int]:
    m = ALL_RE.search(notes)
    if not m:
        return []
    try:
        return list(ast.literal_eval(m.group(1)))
    except (ValueError, SyntaxError):
        return []


def main() -> int:
    rows = list(csv.DictReader(open(OUT_CSV)))
    changes: list[dict] = []

    for r in rows:
        if r["lookup_status"] != "ok_multi_cid":
            continue

        cids = parse_all_cids(r["notes"])
        if not cids:
            continue

        candidates: list[tuple[int, str, str]] = []  # (cid, iso, canon)
        for c in cids:
            props = cid_to_props(c)
            if not props:
                continue
            iso = props.get("SMILES", "") or ""
            canon = props.get("ConnectivitySMILES", "") or iso
            candidates.append((c, iso, canon))

        no_salt = [t for t in candidates if "." not in t[1]]
        with_salt = [t for t in candidates if "." in t[1]]

        original_cid = r["pubchem_cid"]
        change_record = {
            "drug_name": r["drug_name"],
            "before_cid": original_cid,
            "candidates": [(c, iso[:60], "salt" if "." in iso else "free")
                           for c, iso, _ in candidates],
        }

        if len(no_salt) == 1:
            new_cid, new_iso, new_canon = no_salt[0]
            if str(new_cid) != original_cid:
                r["pubchem_cid"] = str(new_cid)
                r["isomeric_smiles"] = new_iso
                r["canonical_smiles"] = new_canon
                r["lookup_status"] = "ok_multi_cid_resolved"
                r["notes"] = (
                    f"auto-resolved: picked CID {new_cid} (no '.' = free form). "
                    f"all={cids}, with_salt_cids={[t[0] for t in with_salt]}"
                )
                change_record["after_cid"] = str(new_cid)
                change_record["action"] = "switched"
            else:
                r["lookup_status"] = "ok_multi_cid_resolved"
                r["notes"] = (
                    f"auto-resolved: original CID {original_cid} is already the free form. "
                    f"all={cids}"
                )
                change_record["after_cid"] = original_cid
                change_record["action"] = "kept (already free)"
        else:
            # 0 or >=2 no-salt candidates -> ambiguous
            reason = ("all candidates contain '.'" if len(no_salt) == 0
                      else f"{len(no_salt)} candidates with no '.'")
            r["notes"] = (
                f"manual review needed: {reason}. kept CID {original_cid}. "
                f"all={cids}, no_salt_cids={[t[0] for t in no_salt]}, "
                f"with_salt_cids={[t[0] for t in with_salt]}"
            )
            change_record["after_cid"] = original_cid
            change_record["action"] = f"flagged: {reason}"

        changes.append(change_record)

    fields = ["drug_name", "pubchem_cid", "canonical_smiles", "isomeric_smiles",
              "lookup_status", "notes"]
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})

    print(f"processed {len(changes)} multi-CID rows\n")
    for c in changes:
        print(f"{c['drug_name']}")
        print(f"  action: {c['action']}")
        print(f"  before_cid={c['before_cid']}  after_cid={c['after_cid']}")
        for cid, iso_snip, flag in c["candidates"]:
            marker = "<<" if str(cid) == c["after_cid"] else "  "
            print(f"  {marker} CID {cid:>10}  [{flag}]  {iso_snip}")
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
