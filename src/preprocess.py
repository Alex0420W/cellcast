"""CellCast preprocessing pipeline — milestone 3 sub-step 3B.

Turns scPerturb v1.4 Sci-Plex 3 into pseudobulk training examples:
  - 24h subset only
  - mouse genes dropped
  - cells with ncounts < 500 dropped
  - per-(cell_line, perturbation, dose) pseudobulk
  - normalize_total(1e4) + log1p
  - HVGs from cell-level CONTROLS (seurat_v3, top-3000 per cell line, union)
  - LFC = perturbed - per-cell-line control pseudobulk, restricted to HVGs
  - input gene-rank list: top-1294 (by control pseudobulk) genes that exist in MAMMAL's vocab
  - SMILES joined from data/sciplex/drug_smiles.csv

Outputs to data/sciplex/processed/:
  - cellcast_v0.parquet      one row per perturbed condition (2,256 rows)
  - hvg_genes.txt            HVG list, one symbol per line, alphabetical
  - control_pseudobulks.npz  per-cell-line control log-normalized pseudobulk
  - preprocess_log.txt       step-by-step summary
"""
from __future__ import annotations

import os
import resource
import time
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp

# ----- paths -----
H5AD = os.path.expanduser("~/data/sciplex/SrivatsanTrapnell2020_sciplex3.h5ad")
SMILES_CSV = os.path.expanduser("~/cellcast/data/sciplex/drug_smiles.csv")
OUT_DIR = Path(os.path.expanduser("~/cellcast/data/sciplex/processed"))

EXPECTED_24H_CELLS = 680_685       # asserted; if mismatch, halt
N_TOP_GENES_PER_CL = 3000          # HVG top-N per cell line
INPUT_GENES_LEN = 1294             # MAMMAL ranked-gene budget
DOSE_BIN = {
    10.0: "DOSE_10nM",
    100.0: "DOSE_100nM",
    1000.0: "DOSE_1000nM",
    10000.0: "DOSE_10000nM",
}

LOG: list[str] = []


def log(msg: str) -> None:
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    LOG.append(line)


def peak_rss_gb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 / 1024


def step_load_filter() -> ad.AnnData:
    log(f"(a) load {H5AD}")
    a = sc.read_h5ad(H5AD)
    log(f"    raw shape: {a.shape}")
    n0 = a.n_obs

    mask_valid = a.obs["cell_line"].notna()
    n_demux = int((~mask_valid).sum())
    a = a[mask_valid].copy()
    log(f"    drop demultiplex failures: -{n_demux}  -> {a.n_obs}")

    mask_24h = a.obs["time"] == 24.0
    n_72h = int((~mask_24h).sum())
    a = a[mask_24h].copy()
    log(f"    drop 72h cells:            -{n_72h}  -> {a.n_obs}")

    if a.n_obs != EXPECTED_24H_CELLS:
        raise RuntimeError(
            f"24h-subset cell count mismatch: got {a.n_obs}, expected {EXPECTED_24H_CELLS}. "
            f"Halting per pipeline contract."
        )
    log(f"    asserted 24h cell count == {EXPECTED_24H_CELLS}")
    log(f"    after load+filter peak_rss={peak_rss_gb():.1f} GB")
    return a


def step_drop_mouse(a: ad.AnnData) -> ad.AnnData:
    eid = a.var["ensembl_id"].astype(str)
    is_human = eid.str.startswith("ENSG").to_numpy()
    n_mouse = int((~is_human).sum())
    n_human = int(is_human.sum())
    a = a[:, is_human].copy()
    log(f"(b) drop mouse genes: -{n_mouse}  -> human genes: {n_human}")
    return a


def step_ncounts_qc(a: ad.AnnData) -> ad.AnnData:
    pre = a.obs.groupby("cell_line", observed=True).size()
    mask = a.obs["ncounts"] >= 500
    n_drop = int((~mask).sum())
    dropped_per_cl = a.obs[~mask].groupby("cell_line", observed=True).size()
    a = a[mask].copy()
    post = a.obs.groupby("cell_line", observed=True).size()
    log(f"(c) ncounts<500 QC: -{n_drop}")
    for cl in sorted(pre.index):
        log(f"    {cl}: {int(pre.get(cl, 0)):>7}  -> {int(post.get(cl, 0)):>7}  (dropped {int(dropped_per_cl.get(cl, 0)):>5})")
    log(f"    after QC peak_rss={peak_rss_gb():.1f} GB")
    return a


def step_hvg_per_cl_union(a: ad.AnnData) -> tuple[list[str], dict[str, int]]:
    """cell_ranger HVG on cell-level controls, per cell line, union of top-N.

    cell_ranger expects log-normalized data (per scanpy docs), so we normalize_total + log1p
    on a copy of each cell-line control subset before running HVG. The main AnnData `a`
    remains untouched (raw counts) so pseudobulking downstream sees the original X.
    """
    obs = a.obs
    cell_lines = sorted(obs["cell_line"].astype(str).unique())
    union: set[str] = set()
    per_cl_top: dict[str, int] = {}
    for cl in cell_lines:
        mask = (obs["cell_line"].astype(str) == cl) & (obs["perturbation"].astype(str) == "control")
        sub = a[mask, :].copy()  # copy so normalize_total / filter_genes don't touch the main adata
        n_genes_start = sub.n_vars
        # filter_genes(min_cells=10) is local to the HVG subset; cures cell_ranger
        # bin-edge collisions caused by mass-zero genes. Does not affect the global
        # gene pool used for LFC / gene-rank list.
        sc.pp.filter_genes(sub, min_cells=10)
        n_genes_after_filter = sub.n_vars
        sc.pp.normalize_total(sub, target_sum=1e4)
        sc.pp.log1p(sub)
        sc.pp.highly_variable_genes(sub, flavor="cell_ranger", n_top_genes=N_TOP_GENES_PER_CL)
        hvg = sub.var.index[sub.var["highly_variable"]].tolist()
        per_cl_top[cl] = len(hvg)
        log(
            f"(f) {cl} controls: n_cells={sub.n_obs}  "
            f"genes {n_genes_start} → {n_genes_after_filter} (after min_cells=10) "
            f"→ {len(hvg)} HVGs (target {N_TOP_GENES_PER_CL})"
        )
        union.update(hvg)
        del sub
    hvg_list = sorted(union)
    log(f"    UNION HVG count: {len(hvg_list)}  (per-cell-line sizes: {per_cl_top})")
    log(f"    after HVG peak_rss={peak_rss_gb():.1f} GB")
    return hvg_list, per_cl_top


def step_pseudobulk(a: ad.AnnData) -> tuple[ad.AnnData, np.ndarray, list[tuple[str, str, float]]]:
    """Group cells by (cell_line, perturbation, dose_value); sum raw counts.

    Returns (pb_adata, n_cells_per_cond, conds)
      pb_adata.X: raw-count pseudobulk, [n_cond, n_genes]
      conds: list of (cell_line, perturbation, dose) tuples, same order as pb_adata.obs
    """
    obs = a.obs[["cell_line", "perturbation", "dose_value"]].copy()
    obs["cell_line"] = obs["cell_line"].astype(str)
    obs["perturbation"] = obs["perturbation"].astype(str)
    obs["dose_value"] = obs["dose_value"].astype(float)

    cond_str = obs["cell_line"] + "||" + obs["perturbation"] + "||" + obs["dose_value"].map(str)
    codes, uniques = pd.factorize(cond_str)
    n_cond = len(uniques)
    n_cells = len(codes)
    log(f"(d) pseudobulk: {n_cells:,} cells -> {n_cond} conditions")

    # Indicator [n_cond, n_cells]; pseudobulk = indicator @ X
    indicator = sp.csr_matrix(
        (np.ones(n_cells, dtype=np.float32), (codes, np.arange(n_cells))),
        shape=(n_cond, n_cells),
    )
    X = a.X.tocsr() if not sp.isspmatrix_csr(a.X) else a.X
    X = X.astype(np.float32)
    pb_X = (indicator @ X).tocsr()  # [n_cond, n_genes] sparse
    log(f"    pseudobulk X: shape={pb_X.shape}  nnz={pb_X.nnz:,}  ({pb_X.nnz / pb_X.shape[0] / pb_X.shape[1] * 100:.2f}% dense)")

    n_cells_per_cond = np.asarray(indicator.sum(axis=1)).ravel().astype(np.int64)

    conds: list[tuple[str, str, float]] = []
    for cstr in uniques:
        cl, pert, dose = cstr.split("||")
        conds.append((cl, pert, float(dose)))

    pb_obs = pd.DataFrame({
        "cell_line": [c[0] for c in conds],
        "perturbation": [c[1] for c in conds],
        "dose_value": [c[2] for c in conds],
        "n_cells_aggregated": n_cells_per_cond,
    })
    pb_adata = ad.AnnData(X=pb_X, obs=pb_obs, var=a.var.copy())

    expected = 3 + 188 * 4 * 3
    if pb_adata.n_obs != expected:
        raise RuntimeError(
            f"pseudobulk condition count mismatch: got {pb_adata.n_obs}, expected {expected}. Halting."
        )
    log(f"    asserted condition count == {expected}  (3 control + 188*4*3 = 2256 perturbed)")
    log(f"    after pseudobulk peak_rss={peak_rss_gb():.1f} GB")
    return pb_adata, n_cells_per_cond, conds


def step_normalize(pb: ad.AnnData) -> ad.AnnData:
    log("(e) normalize_total(1e4) + log1p on pseudobulk")
    sc.pp.normalize_total(pb, target_sum=1e4)
    sc.pp.log1p(pb)
    return pb


def step_compute_lfc(
    pb: ad.AnnData,
    hvg_list: list[str],
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Returns (lfc_matrix [n_cond, n_hvg], control_pb_by_cl: dict[cell_line, np.array [n_hvg]])."""
    log(f"(g) LFC vs per-cell-line control, restricted to {len(hvg_list)} HVGs")
    hvg_idx = pb.var.index.get_indexer(hvg_list)
    if (hvg_idx < 0).any():
        missing = [g for g, i in zip(hvg_list, hvg_idx) if i < 0]
        raise RuntimeError(f"{len(missing)} HVG genes not present in pseudobulk var index: {missing[:5]}")

    Xh = pb.X[:, hvg_idx]
    if sp.issparse(Xh):
        Xh = Xh.toarray()
    Xh = np.asarray(Xh, dtype=np.float32)

    pb_obs = pb.obs
    control_pb_by_cl: dict[str, np.ndarray] = {}
    for cl in sorted(pb_obs["cell_line"].unique()):
        mask = (pb_obs["cell_line"] == cl) & (pb_obs["perturbation"] == "control")
        n = int(mask.sum())
        if n != 1:
            raise RuntimeError(f"expected exactly 1 control pseudobulk per cell line, {cl} has {n}")
        control_pb_by_cl[cl] = Xh[mask.to_numpy()][0]

    lfc = np.empty_like(Xh)
    for i in range(pb.n_obs):
        cl = pb_obs["cell_line"].iloc[i]
        lfc[i] = Xh[i] - control_pb_by_cl[cl]

    # Sanity: control rows LFC should be ~0
    ctrl_mask = (pb_obs["perturbation"] == "control").to_numpy()
    max_abs_ctrl = float(np.abs(lfc[ctrl_mask]).max())
    log(f"    control LFC max|x| = {max_abs_ctrl:.3e}  (asserting < 0.01)")
    if max_abs_ctrl >= 0.01:
        raise RuntimeError(f"control LFC sanity failed: max|x| = {max_abs_ctrl}")
    return lfc, control_pb_by_cl


def _gene_in_mammal_vocab(tokenizer_op, gene: str) -> bool:
    try:
        tokenizer_op.get_token_id(f"[{gene}]")
        return True
    except (AssertionError, KeyError):
        return False


def step_input_gene_ranked_list(
    pb: ad.AnnData,
    tokenizer_op,
) -> tuple[dict[str, list[str]], dict[str, int]]:
    """For each cell line, rank ~58k human genes desc by control pseudobulk expression,
    filter to MAMMAL vocab, take top INPUT_GENES_LEN. Returns (rank_list_by_cl, n_dropped_by_cl)."""
    log(f"(h) per-cell-line input gene-rank list, target length {INPUT_GENES_LEN}")
    rank_by_cl: dict[str, list[str]] = {}
    dropped_by_cl: dict[str, int] = {}
    all_genes = pb.var.index.to_numpy()
    for cl in sorted(pb.obs["cell_line"].unique()):
        mask = (pb.obs["cell_line"] == cl) & (pb.obs["perturbation"] == "control")
        row_idx = int(np.where(mask.to_numpy())[0][0])
        v = pb.X[row_idx]
        v = v.toarray().ravel() if sp.issparse(v) else np.asarray(v).ravel()
        order = np.argsort(-v)  # descending
        ranked_genes = all_genes[order]
        # Walk down ranked list, keep those in vocab, until we have INPUT_GENES_LEN
        kept: list[str] = []
        dropped = 0
        for g in ranked_genes:
            if _gene_in_mammal_vocab(tokenizer_op, g):
                kept.append(g)
                if len(kept) == INPUT_GENES_LEN:
                    break
            else:
                dropped += 1
        if len(kept) < INPUT_GENES_LEN:
            raise RuntimeError(f"{cl}: only {len(kept)} vocab-known genes available, need {INPUT_GENES_LEN}")
        rank_by_cl[cl] = kept
        dropped_by_cl[cl] = dropped
        log(f"    {cl}: kept {len(kept)} (top expressed in vocab), dropped {dropped} unknown to MAMMAL vocab")
    return rank_by_cl, dropped_by_cl


def step_smiles_join() -> dict[str, str]:
    log(f"(i) load SMILES map from {SMILES_CSV}")
    df = pd.read_csv(SMILES_CSV)
    if df["isomeric_smiles"].isna().any() or (df["isomeric_smiles"] == "").any():
        bad = df.loc[df["isomeric_smiles"].isna() | (df["isomeric_smiles"] == ""), "drug_name"].tolist()
        raise RuntimeError(f"drug_smiles.csv has empty isomeric_smiles for: {bad[:5]}")
    smap = dict(zip(df["drug_name"], df["isomeric_smiles"]))
    log(f"    {len(smap)} drugs in SMILES map")
    return smap


def step_save(
    pb: ad.AnnData,
    lfc: np.ndarray,
    hvg_list: list[str],
    control_pb_by_cl: dict[str, np.ndarray],
    rank_by_cl: dict[str, list[str]],
    smap: dict[str, str],
    n_cells_per_cond: np.ndarray,
) -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # HVG list
    hvg_path = OUT_DIR / "hvg_genes.txt"
    hvg_path.write_text("\n".join(hvg_list) + "\n")
    log(f"(j) wrote {hvg_path} ({len(hvg_list)} genes)")

    # Control pseudobulks
    cpb_path = OUT_DIR / "control_pseudobulks.npz"
    np.savez(cpb_path, **{f"{cl}": v for cl, v in control_pb_by_cl.items()},
             hvg_order=np.array(hvg_list, dtype=object))
    log(f"    wrote {cpb_path}")

    # Parquet: filter out controls (LFC is zero by construction; we don't train on them)
    obs = pb.obs.reset_index(drop=True).copy()
    obs["n_cells_aggregated"] = n_cells_per_cond
    keep = obs["perturbation"] != "control"
    obs = obs[keep].reset_index(drop=True)
    lfc_kept = lfc[keep.to_numpy()]
    log(f"    perturbed conditions for parquet: {len(obs)}")
    if len(obs) != 2256:
        raise RuntimeError(f"perturbed-condition count != 2256: got {len(obs)}")

    rows = []
    for i, r in obs.iterrows():
        cl = r["cell_line"]
        drug = r["perturbation"]
        dose = float(r["dose_value"])
        rows.append({
            "condition_id": f"{cl}_{drug}_{int(dose) if dose.is_integer() else dose}",
            "cell_line": cl,
            "drug_name": drug,
            "dose_nM": dose,
            "dose_bin": DOSE_BIN[dose],
            "smiles": smap[drug],
            "input_gene_ranked_list": rank_by_cl[cl],
            "label_lfc_vector": lfc_kept[i].astype(np.float32),
            "n_cells_aggregated": int(r["n_cells_aggregated"]),
        })
    df = pd.DataFrame(rows)
    parquet_path = OUT_DIR / "cellcast_v0.parquet"
    df.to_parquet(parquet_path, engine="pyarrow", compression="snappy")
    log(f"    wrote {parquet_path} ({len(df)} rows)")
    return parquet_path


def main() -> None:
    t0 = time.time()
    log("=" * 70)
    log("CellCast preprocessing — 3B")
    log("=" * 70)

    a = step_load_filter()
    a = step_drop_mouse(a)
    a = step_ncounts_qc(a)

    hvg_list, _per_cl_top = step_hvg_per_cl_union(a)

    pb, n_cells_per_cond, _ = step_pseudobulk(a)
    pb = step_normalize(pb)

    lfc, control_pb_by_cl = step_compute_lfc(pb, hvg_list)

    # Defer tokenizer import (heavy) until needed
    from fuse.data.tokenizers.modular_tokenizer.op import ModularTokenizerOp
    tokenizer_op = ModularTokenizerOp.from_pretrained("ibm/biomed.omics.bl.sm.ma-ted-458m")
    rank_by_cl, _dropped = step_input_gene_ranked_list(pb, tokenizer_op)

    smap = step_smiles_join()

    step_save(pb, lfc, hvg_list, control_pb_by_cl, rank_by_cl, smap, n_cells_per_cond)

    elapsed = time.time() - t0
    log(f"DONE  wall-clock={elapsed/60:.1f} min  peak_rss={peak_rss_gb():.1f} GB")

    log_path = OUT_DIR / "preprocess_log.txt"
    log_path.write_text("\n".join(LOG) + "\n")
    print(f"wrote {log_path}")


if __name__ == "__main__":
    main()
