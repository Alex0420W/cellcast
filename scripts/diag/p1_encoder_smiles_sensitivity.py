"""P1 — Encoder output sensitivity to SMILES.

5 chemically diverse held-out test drugs × 3 cell lines × dose=1000nM.
For each forward, mean-pool the encoder last_hidden_state over four spans:
  (a) full   — all valid (non-pad) positions
  (b) smiles — SMILES content positions only
  (c) mask   — single MASK position (no pooling)
  (d) gene   — gene rank positions only
Compute 5x5 cosine-distance matrices for each (cell_line, span) combination.

Pattern interpretation per the 4A plan:
  - (b) high & (c) low  -> drug signal at SMILES tokens but not propagating to MASK (L2)
  - (b) low             -> encoder collapses drug signal at SMILES level (L1)
  - (b) and (c) both high -> signal reaches the head; head is the problem (L3)
"""
from __future__ import annotations
import json, os, sys, time
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0, os.path.expanduser("~/cellcast"))

from scripts.diag._lib import (
    OUT_ROOT, P1_DRUGS, build_one_sample, cosine_distance_matrix,
    find_spans, forward_with_hidden, heatmap, load_model, load_test_df,
    span_meanpool,
)

OUT = OUT_ROOT / "p1"
CELL_LINES = ("K562", "A549", "MCF7")
DOSE_NM = 1000


def short_label(d: str) -> str:
    """Compact display label for plot axes."""
    short = {
        "Lomustine ": "Lomustine",
        "Quercetin": "Quercetin",
        "Dasatinib": "Dasatinib",
        "Bisindolylmaleimide IX (Ro 31-8220 Mesylate)": "Bisindolylmal-IX",
        "2-Methoxyestradiol (2-MeOE2)": "2-MeOE2",
    }
    return short.get(d, d.split(" ")[0][:14])


def main():
    t0 = time.time()
    print("[P1] loading model + data ...", flush=True)
    L = load_model()
    df = load_test_df()
    print(f"  device={L.device}  test_df rows={len(df)}", flush=True)

    rows = []
    for cl in CELL_LINES:
        for d in P1_DRUGS:
            sub = df[(df.drug_name == d) & (df.cell_line == cl) & (df.dose_nM == DOSE_NM)]
            if len(sub) == 0:
                raise RuntimeError(f"no row for ({d!r}, {cl}, {DOSE_NM}nM)")
            rows.append((d, cl, sub.iloc[0]))
    print(f"  built sample list: {len(rows)} forwards", flush=True)

    span_meta = {}              # one-time per cell line
    pooled = {                  # cell_line -> span_kind -> [n_drugs, D]
        cl: {k: [] for k in ("full", "smiles", "mask", "gene")}
        for cl in CELL_LINES
    }

    print("[P1] running forwards ...", flush=True)
    for i, (d, cl, r) in enumerate(rows):
        s = build_one_sample(
            smiles=r["smiles"], dose_bin=r["dose_bin"],
            ranked_genes=list(r["input_gene_ranked_list"]),
            lfc_vector=np.asarray(r["label_lfc_vector"], dtype=np.float32),
            tokenizer_op=L.tokenizer_op,
        )
        out = forward_with_hidden(L, [s])
        h = out.last_hidden[0].cpu()        # [S, D]
        tok = out.token_ids[0].cpu()
        am = out.attention_mask[0].cpu()
        spans = find_spans(tok, L.special_token_ids, am)

        full = h[:spans.valid_len, :].mean(dim=0).numpy()
        smi  = span_meanpool(h, spans.smiles_start, spans.smiles_end).numpy()
        msk  = h[spans.mask, :].numpy()
        gen  = span_meanpool(h, spans.gene_start, spans.gene_end).numpy()
        pooled[cl]["full"].append(full)
        pooled[cl]["smiles"].append(smi)
        pooled[cl]["mask"].append(msk)
        pooled[cl]["gene"].append(gen)

        if cl not in span_meta:
            span_meta[cl] = {
                "smiles_len": spans.smiles_end - spans.smiles_start,
                "gene_len":   spans.gene_end - spans.gene_start,
                "valid_len":  spans.valid_len,
                "dose_pos":   spans.dose_pos,
            }
        print(f"  [{i+1:2d}/15] {cl} {short_label(d):<18s} smi_len={spans.smiles_end-spans.smiles_start:2d} valid={spans.valid_len}",
              flush=True)

    # Compute distance matrices
    matrices = {}
    for cl in CELL_LINES:
        matrices[cl] = {}
        for k in ("full", "smiles", "mask", "gene"):
            arr = np.stack(pooled[cl][k])  # [5, D]
            matrices[cl][k] = cosine_distance_matrix(arr)

    labels = [short_label(d) for d in P1_DRUGS]

    # Save raw NPZ
    npz_payload = {}
    for cl in CELL_LINES:
        for k, mat in matrices[cl].items():
            npz_payload[f"{cl}__{k}__cosdist"] = mat
        for k in ("full", "smiles", "mask", "gene"):
            npz_payload[f"{cl}__{k}__vec"] = np.stack(pooled[cl][k])
    npz_payload["drug_labels"] = np.array(labels)
    npz_payload["cell_lines"] = np.array(list(CELL_LINES))
    np.savez(OUT / "p1_distances.npz", **npz_payload)
    print(f"  wrote {OUT / 'p1_distances.npz'}")

    # Plot heatmaps — one per (cell_line, span)
    for cl in CELL_LINES:
        # Find a per-cell-line common scale to make the four panels comparable
        max_v = max(matrices[cl][k].max() for k in ("full", "smiles", "mask", "gene"))
        for k in ("full", "smiles", "mask", "gene"):
            heatmap(
                matrices[cl][k], labels=labels,
                title=f"P1 cosine dist  cell={cl}  span={k}  dose={DOSE_NM}nM\n"
                      f"smi_len={span_meta[cl]['smiles_len']} gene_len={span_meta[cl]['gene_len']}",
                out_path=OUT / f"p1_cosdist_{cl}_{k}.png",
                vmin=0.0, vmax=max_v,
            )

    # Compact summary JSON
    summary = {
        "drug_labels": labels,
        "drug_full_names": list(P1_DRUGS),
        "cell_lines": list(CELL_LINES),
        "dose_nM": DOSE_NM,
        "span_meta": span_meta,
        "stats": {},  # cell_line -> span -> {mean_offdiag, max, min}
    }
    for cl in CELL_LINES:
        summary["stats"][cl] = {}
        for k, mat in matrices[cl].items():
            n = mat.shape[0]
            offdiag = mat[np.triu_indices(n, k=1)]
            summary["stats"][cl][k] = {
                "mean_offdiag": float(offdiag.mean()),
                "max":          float(offdiag.max()),
                "min":          float(offdiag.min()),
            }
    (OUT / "p1_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"  wrote {OUT / 'p1_summary.json'}")

    # Print a punchline
    print("\n[P1] Summary (mean off-diagonal cosine distance, by cell line × span):")
    print(f"  {'cell':>5s}  {'full':>8s}  {'smiles':>8s}  {'mask':>8s}  {'gene':>8s}")
    for cl in CELL_LINES:
        s = summary["stats"][cl]
        print(f"  {cl:>5s}  {s['full']['mean_offdiag']:>8.4f}  {s['smiles']['mean_offdiag']:>8.4f}  "
              f"{s['mask']['mean_offdiag']:>8.4f}  {s['gene']['mean_offdiag']:>8.4f}")

    print(f"\n[P1] DONE  {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
