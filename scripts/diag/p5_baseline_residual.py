"""P5 — Baseline-residual ceiling.

For each test condition: residual = true_LFC - baseline_prediction.
Report variance(residual) / variance(true_LFC).

If <10%, there's almost nothing for any model to learn beyond the per-stratum
mean. If >30%, there's substantial drug-specific signal we should be able
to capture.

The variance ratio is computed three ways:
  (1) flattened (treat all gene*condition entries as one population)
  (2) per-condition (variance across genes, averaged over conditions)
  (3) per-gene (variance across conditions, averaged over genes)

Each tells us a slightly different thing about where the headroom lives.
"""
from __future__ import annotations
import json, os, sys, time
from pathlib import Path
import numpy as np
sys.path.insert(0, os.path.expanduser("~/cellcast"))

from scripts.diag._lib import OUT_ROOT, BASELINE, PRED_NPZ

OUT = OUT_ROOT / "p5"


def variance_ratios(residual: np.ndarray, target: np.ndarray) -> dict:
    """residual, target: [N, G]."""
    flat_var_t = float(target.var())
    flat_var_r = float(residual.var())
    out = {
        "flat_var_target":   flat_var_t,
        "flat_var_residual": flat_var_r,
        "flat_var_ratio":    float(flat_var_r / flat_var_t),

        "per_condition_var_target":   float(target.var(axis=1).mean()),
        "per_condition_var_residual": float(residual.var(axis=1).mean()),
        "per_condition_var_ratio":    float((residual.var(axis=1) / (target.var(axis=1) + 1e-12)).mean()),

        "per_gene_var_target":   float(target.var(axis=0).mean()),
        "per_gene_var_residual": float(residual.var(axis=0).mean()),
        "per_gene_var_ratio":    float((residual.var(axis=0) / (target.var(axis=0) + 1e-12)).mean()),
    }
    return out


def main():
    t0 = time.time()
    print("[P5] loading baseline + targets ...", flush=True)
    bl = np.load(BASELINE, allow_pickle=True)
    pred = np.load(PRED_NPZ, allow_pickle=True)
    # Use the prediction-npz targets as canonical (already aligned to test_df)
    targets = pred["targets"].astype(np.float32)             # [N, G]
    cl = pred["cell_lines"]; doses = pred["dose_nM"]
    drug = pred["drug_names"]; cond = pred["condition_ids"]

    bl_ids = list(bl["condition_ids"])
    df_ids = list(cond)
    if bl_ids != df_ids:
        order = {c: i for i, c in enumerate(bl_ids)}
        idx = np.array([order[c] for c in df_ids])
        bl_preds = bl["preds"][idx]
    else:
        bl_preds = bl["preds"]
    bl_preds = bl_preds.astype(np.float32)

    residual = targets - bl_preds
    print(f"  N={targets.shape[0]} conditions  G={targets.shape[1]} genes", flush=True)

    overall = variance_ratios(residual, targets)
    print(f"\n[P5] Overall variance ratios:")
    for k, v in overall.items():
        print(f"  {k:<32s} {v:.6f}")

    # Per-cell-line breakdown
    per_cl = {}
    for c in sorted(set(cl)):
        mask = (cl == c)
        per_cl[c] = variance_ratios(residual[mask], targets[mask])
    print(f"\n[P5] Per-cell-line variance ratios (flat):")
    for c, vr in per_cl.items():
        print(f"  {c}: target_var={vr['flat_var_target']:.5f}  resid_var={vr['flat_var_residual']:.5f}  "
              f"ratio={vr['flat_var_ratio']:.4f}  ({vr['flat_var_ratio'] * 100:.2f}%)")

    # Per-dose breakdown
    per_dose = {}
    for d in sorted(set(doses)):
        mask = (doses == d)
        per_dose[float(d)] = variance_ratios(residual[mask], targets[mask])
    print(f"\n[P5] Per-dose variance ratios (flat):")
    for d, vr in per_dose.items():
        print(f"  {d:>9.0f}nM: target_var={vr['flat_var_target']:.5f}  ratio={vr['flat_var_ratio']:.4f}")

    # Save NPZ + JSON
    summary = {
        "n_conditions": int(targets.shape[0]),
        "n_genes": int(targets.shape[1]),
        "overall": overall,
        "per_cell_line": per_cl,
        "per_dose": per_dose,
        "interpretation": {
            "ceiling_low_threshold": 0.10,
            "ceiling_high_threshold": 0.30,
            "ratio": overall["per_condition_var_ratio"],
            "headroom_remaining_after_baseline": float(1.0 - overall["per_condition_var_ratio"]),
        },
    }
    (OUT / "p5_summary.json").write_text(json.dumps(summary, indent=2))
    np.savez(OUT / "p5_residual.npz",
             targets=targets, baseline_preds=bl_preds, residual=residual,
             cell_lines=cl, dose_nM=doses, drug_names=drug,
             condition_ids=cond,
             per_condition_target_var=targets.var(axis=1),
             per_condition_residual_var=residual.var(axis=1))
    print(f"  wrote {OUT / 'p5_summary.json'}")
    print(f"  wrote {OUT / 'p5_residual.npz'}")

    # ---- Plots ----
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), dpi=150)
    pc_var_t = targets.var(axis=1)
    pc_var_r = residual.var(axis=1)
    axes[0].scatter(pc_var_t, pc_var_r, s=8, alpha=0.5)
    lo = float(min(pc_var_t.min(), pc_var_r.min()))
    hi = float(max(pc_var_t.max(), pc_var_r.max()))
    axes[0].plot([lo, hi], [lo, hi], ls="--", c="red", label="y=x (residual=target)")
    axes[0].plot([lo, hi], [lo / 10, hi / 10], ls=":", c="grey", label="y=x/10  (10% ceiling)")
    axes[0].set_xlabel("per-condition variance of true LFC")
    axes[0].set_ylabel("per-condition variance of residual")
    axes[0].set_title(f"P5 per-condition variance scatter  N={targets.shape[0]}\n"
                      f"mean ratio = {overall['per_condition_var_ratio']:.3f}")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3)

    # Histogram of per-condition variance ratio
    pc_ratio = pc_var_r / (pc_var_t + 1e-12)
    axes[1].hist(pc_ratio, bins=40, edgecolor="black", alpha=0.8)
    axes[1].axvline(0.10, ls=":", c="green", label="10% ceiling (low)")
    axes[1].axvline(0.30, ls=":", c="orange", label="30% ceiling (high)")
    axes[1].axvline(pc_ratio.mean(), ls="--", c="red", label=f"mean={pc_ratio.mean():.3f}")
    axes[1].set_xlabel("per-condition residual_var / target_var")
    axes[1].set_ylabel("count")
    axes[1].set_title("P5 per-condition residual ratio histogram")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(OUT / "p5_variance_scatter.png")
    plt.close(fig)
    print(f"  wrote {OUT / 'p5_variance_scatter.png'}")

    print(f"\n[P5] Headline:")
    print(f"  flat residual / target variance ratio:           {overall['flat_var_ratio']:.4f}  ({overall['flat_var_ratio']*100:.2f}%)")
    print(f"  per-condition mean ratio:                        {overall['per_condition_var_ratio']:.4f}  ({overall['per_condition_var_ratio']*100:.2f}%)")
    print(f"  per-gene mean ratio:                             {overall['per_gene_var_ratio']:.4f}  ({overall['per_gene_var_ratio']*100:.2f}%)")
    print(f"\n[P5] DONE  {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
