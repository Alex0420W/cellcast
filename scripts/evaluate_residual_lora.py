"""M4B.2 — evaluate the trained CellCast LoRA-residual model on test set.

Loads the best M4B.2 checkpoint (which contains LoRA-adapted encoder + head),
runs forward (residual prediction), reconstructs full LFC =
residual_pred + stratum_mean_full_train_drugs, and reports metrics against
StratifiedMeanBaseline, CellCast v0 (M3), CellCast v0-residual (M4B.1),
and P6 FP-MLP.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(os.path.expanduser("~/cellcast"))
sys.path.insert(0, str(ROOT))

from scripts.train_residual_lora import CellCastResidualLoRAModule  # noqa: E402
from src.models.lora_setup import apply_lora_to_encoder, freeze_for_lora  # noqa: E402
from src.tasks.drug_response_residual import (  # noqa: E402
    DOSE_TOKENS, SLICED_PRED_KEY, StratumMean,
    build_sample_dict, load_or_expand_tokenizer, process_model_output,
)
from scripts.evaluate import macro_pearson, macro_spearman, top_k_dir_acc  # noqa: E402

PARQUET = ROOT / "data/sciplex/processed/cellcast_v0.parquet"
SPLITS_JSON = ROOT / "data/sciplex/processed/splits.json"
HVG_PATH = ROOT / "data/sciplex/processed/hvg_genes.txt"
RUN_DIR = ROOT / "runs/cellcast_v0_residual_lora32"
BASELINE_NPZ = ROOT / "results/baseline_predictions.npz"
CC_M3_NPZ = ROOT / "results/cellcast_v0_predictions.npz"
M4B1_NPZ = ROOT / "results/cellcast_residual_predictions.npz"  # may be regen'd if absent
P6_NPZ = ROOT / "results/p6_predictions.npz"                   # may be regen'd if absent

OUT_NPZ = ROOT / "results/cellcast_residual_lora_predictions.npz"
OUT_JSON = ROOT / "results/m4b2_metrics.json"
BATCH = 32


def metrics_block(P: np.ndarray, T: np.ndarray) -> dict:
    return {
        "pcorr_macro":     macro_pearson(P, T),
        "spearcorr_macro": macro_spearman(P, T),
        "top50_dir_acc":   top_k_dir_acc(P, T, k=50),
        "mse":             float(((P - T) ** 2).mean()),
        "n_conditions":    int(T.shape[0]),
    }


def find_best_ckpt(run_dir: Path) -> Path:
    cks = sorted((run_dir / "checkpoints").glob("best-*.ckpt"))
    if not cks:
        raise FileNotFoundError(f"no best-*.ckpt in {run_dir / 'checkpoints'}")
    return cks[-1]


def predict_residual_lora(test_df: pd.DataFrame, n_hvg: int) -> np.ndarray:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer_op, _ = load_or_expand_tokenizer()
    ckpt_path = find_best_ckpt(RUN_DIR)
    print(f"  loading {ckpt_path.name}", flush=True)

    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    hparams = dict(ckpt.get("hyper_parameters", {}))
    hparams.pop("tokenizer_op", None)
    # Build the model the same way training did so state_dict keys align.
    lm = CellCastResidualLoRAModule(**hparams, tokenizer_op=tokenizer_op)
    # We need to install LoRA + freeze BEFORE load_state_dict, because the
    # ckpt has lora_A / lora_B keys that don't exist in a vanilla Mammal+head.
    apply_lora_to_encoder(
        lm.model,
        rank=hparams.get("lora_rank", 32),
        alpha=hparams.get("lora_alpha", 32),
        dropout=hparams.get("lora_dropout", 0.1),
    )
    freeze_for_lora(lm.model, tokenizer_op, DOSE_TOKENS)
    lm.load_state_dict(ckpt["state_dict"], strict=True)
    lm.eval().to(device)

    preds = np.empty((len(test_df), n_hvg), dtype=np.float32)
    from fuse.data.utils.collates import CollateDefault
    with torch.inference_mode():
        for start in range(0, len(test_df), BATCH):
            chunk = test_df.iloc[start:start + BATCH]
            samples = []
            for _, r in chunk.iterrows():
                samples.append(build_sample_dict(
                    smiles=r["smiles"], dose_bin=r["dose_bin"],
                    ranked_genes=list(r["input_gene_ranked_list"]),
                    lfc_vector=np.asarray(r["label_lfc_vector"], dtype=np.float32),
                    tokenizer_op=tokenizer_op,
                ))
            batch_dict = CollateDefault()(samples)
            for k in ("data.encoder_input_token_ids", "data.encoder_input_attention_mask",
                      "data.labels.scalars.values", "data.labels.scalars.valid_mask"):
                if k in batch_dict and torch.is_tensor(batch_dict[k]):
                    batch_dict[k] = batch_dict[k].to(device)
            with torch.autocast(device_type="cuda" if device == "cuda" else "cpu",
                                dtype=torch.bfloat16):
                out = lm.model.forward_encoder_only(batch_dict)
            out = process_model_output(out)
            preds[start:start + len(chunk)] = out[SLICED_PRED_KEY].float().cpu().numpy()
            if start % (BATCH * 4) == 0:
                print(f"  {start + len(chunk):>4} / {len(test_df)}", flush=True)
    return preds


def align_to(test_df, npz_path, key="preds"):
    if not npz_path.exists():
        return None
    data = np.load(npz_path, allow_pickle=True)
    ids = list(data["condition_ids"])
    df_ids = list(test_df["condition_id"])
    if ids != df_ids:
        order = {c: i for i, c in enumerate(ids)}
        idx = np.array([order[c] for c in df_ids])
        return data[key][idx]
    return data[key]


def main():
    t0 = time.time()
    df = pd.read_parquet(PARQUET)
    splits = json.loads(SPLITS_JSON.read_text())
    n_hvg = len([ln for ln in HVG_PATH.read_text().splitlines() if ln.strip()])

    train_df = df[df["drug_name"].isin(splits["train_drugs"])].reset_index(drop=True)
    test_df = df[df["drug_name"].isin(splits["test_drugs"])].reset_index(drop=True)
    print(f"  train conds={len(train_df)}  test conds={len(test_df)}  G={n_hvg}", flush=True)

    sm_test = StratumMean.fit(train_df)
    print(f"  stratum_mean refit on FULL train drugs: {len(sm_test.means)} strata", flush=True)

    targets_full = np.stack([np.asarray(v, dtype=np.float32)
                             for v in test_df["label_lfc_vector"]])

    print("[1/3] running CellCast-residual-LoRA predictions ...", flush=True)
    preds_residual = predict_residual_lora(test_df, n_hvg)
    preds_full = sm_test.reconstruct(preds_residual, test_df)
    np.savez(
        OUT_NPZ,
        preds_residual=preds_residual,
        preds_full=preds_full,
        targets=targets_full,
        condition_ids=test_df["condition_id"].to_numpy(),
        cell_lines=test_df["cell_line"].to_numpy(),
        drug_names=test_df["drug_name"].to_numpy(),
        dose_nM=test_df["dose_nM"].to_numpy(),
    )
    print(f"  wrote {OUT_NPZ}", flush=True)

    print("[2/3] loading comparison predictions ...", flush=True)
    bl_preds = align_to(test_df, BASELINE_NPZ, "preds")
    cc_preds = align_to(test_df, CC_M3_NPZ, "preds")
    m4b1_preds = align_to(test_df, M4B1_NPZ, "preds_full")
    p6_preds = align_to(test_df, P6_NPZ, "preds_full")
    missing = []
    if bl_preds is None: missing.append("baseline_predictions.npz (run scripts/evaluate.py baseline)")
    if cc_preds is None: missing.append("cellcast_v0_predictions.npz (run scripts/evaluate.py)")
    if m4b1_preds is None: missing.append("cellcast_residual_predictions.npz (run scripts/evaluate_residual.py)")
    if p6_preds is None: missing.append("p6_predictions.npz (run scripts/evaluate_fingerprint.py)")
    if missing:
        print(f"  WARNING: missing comparison NPZs: {missing}", flush=True)

    print("[3/3] computing metrics ...", flush=True)
    cmp = {"m4b2_lora_residual": preds_full}
    if bl_preds is not None:    cmp["stratified_mean_baseline"] = bl_preds
    if cc_preds is not None:    cmp["cellcast_v0_m3"] = cc_preds
    if m4b1_preds is not None:  cmp["cellcast_residual_m4b1"] = m4b1_preds
    if p6_preds is not None:    cmp["p6_fingerprint_mlp"] = p6_preds

    out_metrics = {
        "overall": {k: metrics_block(p, targets_full) for k, p in cmp.items()},
        "per_cell_line": {},
    }
    for cl in sorted(test_df["cell_line"].unique()):
        mask = (test_df["cell_line"] == cl).to_numpy()
        out_metrics["per_cell_line"][cl] = {
            k: metrics_block(p[mask], targets_full[mask]) for k, p in cmp.items()
        }
    OUT_JSON.write_text(json.dumps(out_metrics, indent=2))
    print(f"  wrote {OUT_JSON}", flush=True)

    # ---- Print headline tables ---- #
    cols_order = ["m4b2_lora_residual", "cellcast_residual_m4b1", "cellcast_v0_m3",
                  "stratified_mean_baseline", "p6_fingerprint_mlp"]
    cols = [c for c in cols_order if c in out_metrics["overall"]]
    short = {"m4b2_lora_residual": "M4B.2_LoRA", "cellcast_residual_m4b1": "M4B.1_resid",
             "cellcast_v0_m3": "M3_CCv0", "stratified_mean_baseline": "Baseline",
             "p6_fingerprint_mlp": "P6_FPmlp"}
    print(f"\n[m4b2.eval] OVERALL ({len(test_df)} test conditions)")
    print("  " + f"{'metric':<18s}" + "".join(f"  {short[c]:>14s}" for c in cols))
    for m in ("pcorr_macro", "spearcorr_macro", "top50_dir_acc", "mse"):
        vals = [out_metrics['overall'][c][m] for c in cols]
        line = "".join(f"  {v:>+14.4f}" for v in vals)
        print(f"  {m:<18s}{line}")

    print(f"\n[m4b2.eval] PER CELL LINE — pcorr_macro (M4B primary metric)")
    print("  " + f"{'cl':<5s}" + "".join(f"  {short[c]:>14s}" for c in cols))
    for cl in sorted(out_metrics["per_cell_line"]):
        blk = out_metrics["per_cell_line"][cl]
        vals = [blk[c]["pcorr_macro"] for c in cols]
        line = "".join(f"  {v:>+14.4f}" for v in vals)
        print(f"  {cl:<5s}{line}")

    print(f"\n[m4b2.eval] PER CELL LINE — top50_dir_acc")
    print("  " + f"{'cl':<5s}" + "".join(f"  {short[c]:>14s}" for c in cols))
    for cl in sorted(out_metrics["per_cell_line"]):
        blk = out_metrics["per_cell_line"][cl]
        vals = [blk[c]["top50_dir_acc"] for c in cols]
        line = "".join(f"  {v:>+14.4f}" for v in vals)
        print(f"  {cl:<5s}{line}")

    print(f"\n[m4b2.eval] DONE  {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
