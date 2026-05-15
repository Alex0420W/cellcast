"""P4 — SMILES ablation.

50 random held-out test conditions. For each, run inference twice:
  (a) Normal: real SMILES in the prompt
  (b) Ablated: replace the entire SMILES content with a dummy SMILES
              ("CCO" = ethanol, smallest sensible molecule the SMILES
              tokenizer accepts cleanly).
Compare overall pcorr_macro and top-50 dir_acc on both runs.

If ablated ~= normal, the model wasn't using SMILES.
If ablated meaningfully worse, the model was using SMILES (just not enough
to beat the baseline).
"""
from __future__ import annotations
import json, os, sys, time
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0, os.path.expanduser("~/cellcast"))

from scripts.diag._lib import (
    OUT_ROOT, build_one_sample, forward_with_hidden, load_model, load_test_df,
)
from scripts.evaluate import macro_pearson, macro_spearman, top_k_dir_acc

OUT = OUT_ROOT / "p4"
N_SAMPLES = 50
DUMMY_SMILES = "CCO"          # ethanol; canonical valid SMILES
SEED = 1234
BATCH = 16


def predict_in_batches(L, samples_by_kind: dict[str, list[dict]]) -> dict[str, np.ndarray]:
    """Run forward in batches per kind. Returns kind -> [N, G] preds."""
    out = {}
    for kind, samples in samples_by_kind.items():
        preds = []
        for s_idx in range(0, len(samples), BATCH):
            chunk = samples[s_idx:s_idx + BATCH]
            fout = forward_with_hidden(L, chunk)
            preds.append(fout.pred.cpu().numpy())
        out[kind] = np.concatenate(preds, axis=0)
    return out


def main():
    t0 = time.time()
    print("[P4] loading model + test data ...", flush=True)
    L = load_model()
    df = load_test_df()
    rng = np.random.default_rng(SEED)
    idx = rng.choice(len(df), size=N_SAMPLES, replace=False)
    sub = df.iloc[idx].reset_index(drop=True)
    print(f"  sampled {N_SAMPLES} test conditions  seed={SEED}", flush=True)

    targets = np.stack([np.asarray(v, dtype=np.float32) for v in sub["label_lfc_vector"]])

    # Build sample lists
    print("[P4] building sample lists ...", flush=True)
    real_samples = []
    abl_samples = []
    for _, r in sub.iterrows():
        real_samples.append(build_one_sample(
            smiles=r["smiles"], dose_bin=r["dose_bin"],
            ranked_genes=list(r["input_gene_ranked_list"]),
            lfc_vector=np.asarray(r["label_lfc_vector"], dtype=np.float32),
            tokenizer_op=L.tokenizer_op,
        ))
        abl_samples.append(build_one_sample(
            smiles=DUMMY_SMILES, dose_bin=r["dose_bin"],
            ranked_genes=list(r["input_gene_ranked_list"]),
            lfc_vector=np.asarray(r["label_lfc_vector"], dtype=np.float32),
            tokenizer_op=L.tokenizer_op,
        ))

    print("[P4] running forwards ...", flush=True)
    preds = predict_in_batches(L, {"real": real_samples, "ablated": abl_samples})

    # Metrics
    metrics = {}
    for k in ("real", "ablated"):
        P = preds[k]
        metrics[k] = {
            "pcorr_macro":     macro_pearson(P, targets),
            "spearcorr_macro": macro_spearman(P, targets),
            "top50_dir_acc":   top_k_dir_acc(P, targets, k=50),
            "mse":             float(((P - targets) ** 2).mean()),
        }

    # Compare: distribution of (real_pred - abl_pred) across samples
    diff = preds["real"] - preds["ablated"]
    per_sample_diff_l2 = np.linalg.norm(diff, axis=1)  # [N]
    per_sample_real_norm = np.linalg.norm(preds["real"], axis=1)
    rel_change = per_sample_diff_l2 / (per_sample_real_norm + 1e-12)

    summary = {
        "n_samples": int(N_SAMPLES),
        "seed": int(SEED),
        "dummy_smiles": DUMMY_SMILES,
        "metrics": metrics,
        "delta": {
            "pcorr_macro":     metrics["real"]["pcorr_macro"] - metrics["ablated"]["pcorr_macro"],
            "spearcorr_macro": metrics["real"]["spearcorr_macro"] - metrics["ablated"]["spearcorr_macro"],
            "top50_dir_acc":   metrics["real"]["top50_dir_acc"] - metrics["ablated"]["top50_dir_acc"],
            "mse":             metrics["real"]["mse"] - metrics["ablated"]["mse"],
        },
        "per_sample_pred_diff_l2": {
            "mean":  float(per_sample_diff_l2.mean()),
            "max":   float(per_sample_diff_l2.max()),
            "min":   float(per_sample_diff_l2.min()),
            "std":   float(per_sample_diff_l2.std()),
        },
        "per_sample_relative_change": {
            "mean": float(rel_change.mean()),
            "max":  float(rel_change.max()),
            "median": float(np.median(rel_change)),
        },
    }
    (OUT / "p4_summary.json").write_text(json.dumps(summary, indent=2))
    np.savez(OUT / "p4_preds.npz",
             real=preds["real"], ablated=preds["ablated"], targets=targets,
             condition_ids=sub["condition_id"].to_numpy(),
             cell_lines=sub["cell_line"].to_numpy(),
             drug_names=sub["drug_name"].to_numpy(),
             dose_nM=sub["dose_nM"].to_numpy())

    # Plot histogram of per-sample diff and bar of metrics
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), dpi=150)
    axes[0].hist(per_sample_diff_l2, bins=20, edgecolor="black", alpha=0.8)
    axes[0].set_xlabel("||real_pred - ablated_pred||_2 per sample")
    axes[0].set_ylabel("count")
    axes[0].set_title(f"P4 prediction difference under SMILES ablation\n"
                      f"n={N_SAMPLES}, dummy='{DUMMY_SMILES}'\n"
                      f"mean diff L2 = {per_sample_diff_l2.mean():.4f}  rel = {rel_change.mean():.2%}")
    axes[0].grid(alpha=0.3)

    metric_names = ["pcorr_macro", "spearcorr_macro", "top50_dir_acc", "mse"]
    x = np.arange(len(metric_names))
    width = 0.35
    real_vals = [metrics["real"][m] for m in metric_names]
    abl_vals = [metrics["ablated"][m] for m in metric_names]
    axes[1].bar(x - width / 2, real_vals, width, label="real SMILES")
    axes[1].bar(x + width / 2, abl_vals, width, label="ablated (CCO)")
    for i, m in enumerate(metric_names):
        axes[1].text(i, max(real_vals[i], abl_vals[i]) + 0.005,
                     f"Δ={real_vals[i] - abl_vals[i]:+.4f}", ha="center", fontsize=8)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(metric_names, rotation=20, fontsize=8)
    axes[1].set_title(f"P4 SMILES-real vs SMILES-ablated metrics  n={N_SAMPLES}")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3, axis="y")

    fig.tight_layout()
    fig.savefig(OUT / "p4_ablation.png")
    plt.close(fig)
    print(f"  wrote {OUT / 'p4_ablation.png'}")
    print(f"  wrote {OUT / 'p4_summary.json'}")
    print(f"\n[P4] Headline:")
    for m in metric_names:
        print(f"  {m:<18s}  real={metrics['real'][m]:+.4f}  ablated={metrics['ablated'][m]:+.4f}  Δ={summary['delta'][m]:+.4f}")
    print(f"  per-sample pred-diff L2 mean = {per_sample_diff_l2.mean():.4f}  (max {per_sample_diff_l2.max():.4f})")
    print(f"  per-sample relative change   mean = {rel_change.mean():.2%}  median = {np.median(rel_change):.2%}")
    print(f"\n[P4] DONE  {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
