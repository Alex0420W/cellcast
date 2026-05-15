"""P3 — Dose token influence.

1) For (Dasatinib, K562) at all 4 doses, compute pairwise cosine distance
   between the MASK hidden states. If the dose token does anything, these
   should be more different than what the dose-token-swap experiment shows.
2) Dose-token swap: feed the dose=1000nM input but with the dose-token
   replaced by the other 3 dose tokens (in-prompt swap, gene rank list and
   SMILES unchanged). If swapping the token doesn't change the prediction,
   the dose embedding is being ignored.
"""
from __future__ import annotations
import json, os, sys, time
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0, os.path.expanduser("~/cellcast"))

from scripts.diag._lib import (
    DOSE_TOKENS, OUT_ROOT, build_one_sample, cosine_distance_matrix,
    find_spans, forward_with_hidden, heatmap, load_model, load_full_df,
)

OUT = OUT_ROOT / "p3"
DRUG = "Dasatinib"
CELL = "K562"


def main():
    t0 = time.time()
    print("[P3] loading model + data ...", flush=True)
    L = load_model()
    df = load_full_df()
    sub = df[(df.drug_name == DRUG) & (df.cell_line == CELL)].sort_values("dose_nM")
    if len(sub) != 4:
        raise RuntimeError(f"expected 4 dose conditions for {DRUG}+{CELL}, got {len(sub)}")
    print(f"  {DRUG} + {CELL}: dose_nM = {sub.dose_nM.tolist()}  dose_bin = {sub.dose_bin.tolist()}",
          flush=True)

    # ---- Part 1: real conditions, real dose tokens ----
    print("[P3.1] real-dose forwards ...", flush=True)
    real_mask = []
    real_pred = []
    for _, r in sub.iterrows():
        s = build_one_sample(
            smiles=r["smiles"], dose_bin=r["dose_bin"],
            ranked_genes=list(r["input_gene_ranked_list"]),
            lfc_vector=np.asarray(r["label_lfc_vector"], dtype=np.float32),
            tokenizer_op=L.tokenizer_op,
        )
        out = forward_with_hidden(L, [s])
        h = out.last_hidden[0].cpu()
        spans = find_spans(out.token_ids[0].cpu(), L.special_token_ids,
                           out.attention_mask[0].cpu())
        real_mask.append(h[spans.mask].numpy())
        real_pred.append(out.pred[0].cpu().numpy())
    real_mask = np.stack(real_mask)
    real_pred = np.stack(real_pred)
    real_cosdist = cosine_distance_matrix(real_mask)
    real_pred_l2 = np.zeros((4, 4))
    for i in range(4):
        for j in range(4):
            real_pred_l2[i, j] = float(np.linalg.norm(real_pred[i] - real_pred[j]))

    dose_labels = ["10nM", "100nM", "1000nM", "10000nM"]

    # ---- Part 2: dose-token swap ----
    # Anchor: dose_nM=1000 row. Force the prompt to use each of the 4 dose tokens
    # in turn (geneses + smiles + lfc unchanged).
    anchor = sub[sub.dose_nM == 1000].iloc[0]
    print(f"[P3.2] dose-token-swap forwards (anchor=1000nM, swap dose_token only) ...", flush=True)
    swap_mask = []
    swap_pred = []
    for tok in DOSE_TOKENS:  # ('<DOSE_10nM>', '<DOSE_100nM>', '<DOSE_1000nM>', '<DOSE_10000nM>')
        s = build_one_sample(
            smiles=anchor["smiles"], dose_bin=tok,  # swap here only
            ranked_genes=list(anchor["input_gene_ranked_list"]),
            lfc_vector=np.asarray(anchor["label_lfc_vector"], dtype=np.float32),
            tokenizer_op=L.tokenizer_op,
        )
        out = forward_with_hidden(L, [s])
        h = out.last_hidden[0].cpu()
        spans = find_spans(out.token_ids[0].cpu(), L.special_token_ids,
                           out.attention_mask[0].cpu())
        swap_mask.append(h[spans.mask].numpy())
        swap_pred.append(out.pred[0].cpu().numpy())
    swap_mask = np.stack(swap_mask)
    swap_pred = np.stack(swap_pred)
    swap_cosdist = cosine_distance_matrix(swap_mask)
    swap_pred_l2 = np.zeros((4, 4))
    for i in range(4):
        for j in range(4):
            swap_pred_l2[i, j] = float(np.linalg.norm(swap_pred[i] - swap_pred[j]))

    # ---- Plots ----
    real_max = max(real_cosdist.max(), 1e-4)
    swap_max = max(swap_cosdist.max(), 1e-4)
    common_max = max(real_max, swap_max)
    heatmap(real_cosdist, dose_labels,
            title=f"P3 real-data MASK cosdist  drug={DRUG} cell={CELL}",
            out_path=OUT / "p3_real_mask_cosdist.png",
            vmin=0.0, vmax=common_max)
    heatmap(swap_cosdist, dose_labels,
            title=f"P3 dose-swap MASK cosdist  drug={DRUG} cell={CELL}\n"
                  f"(genes+smiles fixed at 1000nM input; only dose token varies)",
            out_path=OUT / "p3_swap_mask_cosdist.png",
            vmin=0.0, vmax=common_max)

    real_pred_max = max(real_pred_l2.max(), 1e-4)
    swap_pred_max = max(swap_pred_l2.max(), 1e-4)
    common_pred_max = max(real_pred_max, swap_pred_max)
    heatmap(real_pred_l2, dose_labels,
            title=f"P3 real-data PREDICTION L2 dist  drug={DRUG} cell={CELL}",
            out_path=OUT / "p3_real_pred_l2.png",
            vmin=0.0, vmax=common_pred_max)
    heatmap(swap_pred_l2, dose_labels,
            title=f"P3 dose-swap PREDICTION L2 dist  drug={DRUG} cell={CELL}\n"
                  f"(only dose token varies)",
            out_path=OUT / "p3_swap_pred_l2.png",
            vmin=0.0, vmax=common_pred_max)

    # ---- Numerical summary ----
    n = 4
    triu = np.triu_indices(n, k=1)
    summary = {
        "drug": DRUG, "cell_line": CELL, "doses_nM": [10, 100, 1000, 10000],
        "real_mask_cosdist_offdiag_mean": float(real_cosdist[triu].mean()),
        "real_mask_cosdist_offdiag_max":  float(real_cosdist[triu].max()),
        "swap_mask_cosdist_offdiag_mean": float(swap_cosdist[triu].mean()),
        "swap_mask_cosdist_offdiag_max":  float(swap_cosdist[triu].max()),
        "real_pred_l2_offdiag_mean":      float(real_pred_l2[triu].mean()),
        "swap_pred_l2_offdiag_mean":      float(swap_pred_l2[triu].mean()),
        "real_pred_l2_offdiag_max":       float(real_pred_l2[triu].max()),
        "swap_pred_l2_offdiag_max":       float(swap_pred_l2[triu].max()),
        "real_mask_cosdist_matrix": real_cosdist.tolist(),
        "swap_mask_cosdist_matrix": swap_cosdist.tolist(),
        "real_pred_l2_matrix":      real_pred_l2.tolist(),
        "swap_pred_l2_matrix":      swap_pred_l2.tolist(),
    }
    (OUT / "p3_summary.json").write_text(json.dumps(summary, indent=2))
    np.savez(OUT / "p3_arrays.npz",
             real_mask=real_mask, swap_mask=swap_mask,
             real_pred=real_pred, swap_pred=swap_pred,
             dose_labels=np.array(dose_labels))

    print("\n[P3] Summary:")
    print(f"  REAL data (genes change w/ dose, dose-token changes w/ dose):")
    print(f"    MASK cosdist   off-diag mean = {summary['real_mask_cosdist_offdiag_mean']:.4f}  max = {summary['real_mask_cosdist_offdiag_max']:.4f}")
    print(f"    pred  L2 dist  off-diag mean = {summary['real_pred_l2_offdiag_mean']:.4f}  max = {summary['real_pred_l2_offdiag_mean']:.4f}")
    print(f"  SWAP only (genes+smiles fixed, only dose-token swapped):")
    print(f"    MASK cosdist   off-diag mean = {summary['swap_mask_cosdist_offdiag_mean']:.4f}  max = {summary['swap_mask_cosdist_offdiag_max']:.4f}")
    print(f"    pred  L2 dist  off-diag mean = {summary['swap_pred_l2_offdiag_mean']:.4f}  max = {summary['swap_pred_l2_offdiag_max']:.4f}")
    print(f"\n[P3] DONE  {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
