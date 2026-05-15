"""Generate the M3 training-curve figures from TB events."""
from __future__ import annotations

import glob
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

sns.set_theme(style="whitegrid", context="notebook")
plt.rcParams["figure.dpi"] = 110

OUT = Path(os.path.expanduser("~/cellcast/results/m3_figures"))
OUT.mkdir(parents=True, exist_ok=True)

ev = sorted(glob.glob(os.path.expanduser("~/cellcast/runs/cellcast_v0/tb/**/events*"), recursive=True))
ea = EventAccumulator(ev[-1]); ea.Reload()

def vals(tag):
    return np.array([e.step for e in ea.Scalars(tag)]), np.array([e.value for e in ea.Scalars(tag)])

# Train loss (per step) — smooth with EMA
ts, tl = vals("train/loss")
ema = []
last = float(tl[0])
for v in tl:
    last = 0.05 * float(v) + 0.95 * last
    ema.append(last)
ema = np.array(ema)

fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(ts, tl, color="#bbbbbb", linewidth=0.6, label="raw")
ax.plot(ts, ema, color="#1f77b4", linewidth=2, label="EMA (α=0.05)")
ax.axvline(25, color="red", linestyle="--", linewidth=1, label="warmup end")
ax.set_xlabel("step")
ax.set_ylabel("train MSE loss")
ax.set_title("CellCast v0 — training loss (8 epochs, batch 16, frozen backbone)")
ax.legend()
plt.tight_layout(); plt.savefig(OUT / "train_loss.png"); plt.close()

# Val metrics per epoch
metrics = {
    "pcorr_macro":      "per-gene Pearson (macro)",
    "spearcorr_macro":  "per-gene Spearman (macro)",
    "top50_dir_acc":    "top-50 DEG direction accuracy",
    "mse":              "MSE",
}
fig, axes = plt.subplots(2, 2, figsize=(11, 7))
for ax, (m, title) in zip(axes.flat, metrics.items()):
    s, v = vals(f"val/{m}")
    ax.plot(s, v, marker="o", color="#1f77b4", linewidth=2, label="overall")
    for cl, color in zip(("A549", "K562", "MCF7"), ("#2ca02c", "#ff7f0e", "#d62728")):
        s_, v_ = vals(f"val/{cl}/{m}")
        ax.plot(s_, v_, marker="o", color=color, linewidth=1.2, alpha=0.8, label=cl)
    ax.set_xlabel("step (= epoch × 102)")
    ax.set_title(title)
    ax.legend(fontsize=8)
fig.suptitle("CellCast v0 — per-epoch validation metrics (internal 10% split)", y=1.02)
plt.tight_layout(); plt.savefig(OUT / "val_metrics.png"); plt.close()

# Comparison bar chart (CellCast vs baseline overall + per CL)
import json
m = json.loads(Path(os.path.expanduser("~/cellcast/results/3d_metrics.json")).read_text())
groups = ["overall"] + sorted(m["per_cell_line"].keys())
cellcast_v = [m["overall"]["cellcast"]["pcorr_macro"]] + [m["per_cell_line"][g]["cellcast"]["pcorr_macro"] for g in sorted(m["per_cell_line"].keys())]
baseline_v = [m["overall"]["baseline"]["pcorr_macro"]] + [m["per_cell_line"][g]["baseline"]["pcorr_macro"] for g in sorted(m["per_cell_line"].keys())]

fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
ax = axes[0]
x = np.arange(len(groups))
ax.bar(x - 0.18, cellcast_v, width=0.36, label="CellCast", color="#1f77b4")
ax.bar(x + 0.18, baseline_v, width=0.36, label="StratifiedMean", color="#999999")
ax.set_xticks(x); ax.set_xticklabels(groups)
ax.set_ylabel("macro Pearson")
ax.set_title("pcorr_macro — CellCast vs baseline (held-out test)")
ax.legend()
ax.axhline(0, color="black", linewidth=0.5)

ax = axes[1]
cellcast_v = [m["overall"]["cellcast"]["top50_dir_acc"]] + [m["per_cell_line"][g]["cellcast"]["top50_dir_acc"] for g in sorted(m["per_cell_line"].keys())]
baseline_v = [m["overall"]["baseline"]["top50_dir_acc"]] + [m["per_cell_line"][g]["baseline"]["top50_dir_acc"] for g in sorted(m["per_cell_line"].keys())]
ax.bar(x - 0.18, cellcast_v, width=0.36, label="CellCast", color="#1f77b4")
ax.bar(x + 0.18, baseline_v, width=0.36, label="StratifiedMean", color="#999999")
ax.set_xticks(x); ax.set_xticklabels(groups)
ax.set_ylabel("top-50 DEG dir acc")
ax.set_title("top50_dir_acc — CellCast vs baseline (held-out test)")
ax.axhline(0.5, color="red", linestyle="--", linewidth=1, label="chance (0.5)")
ax.legend()
plt.tight_layout(); plt.savefig(OUT / "test_comparison.png"); plt.close()

print(f"wrote figures to {OUT}/")
for p in sorted(OUT.iterdir()):
    print(f"  {p.name}")
