"""Smoke test: load model, run one forward, verify spans + shapes."""
from __future__ import annotations
import time, sys, os, json
from pathlib import Path
import numpy as np
sys.path.insert(0, os.path.expanduser("~/cellcast"))

from scripts.diag._lib import (
    load_model, load_test_df, build_one_sample, forward_with_hidden, find_spans,
    P1_DRUGS, CKPT_SHA256,
)


def main():
    t0 = time.time()
    print("[smoke] loading model ...", flush=True)
    L = load_model()
    print(f"  device={L.device}  loaded in {time.time()-t0:.1f}s", flush=True)
    print(f"  special token IDs:")
    for k, v in L.special_token_ids.items():
        print(f"    {k!s:<55s} {v}")

    print("\n[smoke] loading test data ...", flush=True)
    df = load_test_df()
    print(f"  test_df rows: {len(df)}")

    # Pick one row for each P1 drug to verify they all exist
    for d in P1_DRUGS:
        sub = df[(df["drug_name"] == d) & (df["cell_line"] == "K562") & (df["dose_nM"] == 1000)]
        print(f"  {d!r:<55s} K562@1000nM rows: {len(sub)}")
        if len(sub) == 0:
            # Try fuzzy match
            similar = df["drug_name"].unique()
            print(f"    (no exact match; first 5 K562 test drugs: {sorted(set(df[df.cell_line=='K562'].drug_name))[:5]})")

    # Build sample for first P1 drug
    d = P1_DRUGS[0]
    sub = df[(df["drug_name"] == d) & (df["cell_line"] == "K562") & (df["dose_nM"] == 1000)]
    if len(sub) == 0:
        # Try without trailing space
        sub = df[(df["drug_name"].str.strip() == d.strip()) & (df["cell_line"] == "K562") & (df["dose_nM"] == 1000)]
    print(f"\n[smoke] sample for {d!r}: {len(sub)} rows")
    if len(sub) == 0:
        return

    r = sub.iloc[0]
    print(f"  smiles: {r['smiles']}")
    print(f"  dose_bin: {r['dose_bin']}")
    print(f"  ranked_genes: {len(r['input_gene_ranked_list'])} entries (first 5: {list(r['input_gene_ranked_list'])[:5]})")
    print(f"  lfc_vector shape: {np.asarray(r['label_lfc_vector']).shape}")

    s = build_one_sample(
        smiles=r["smiles"], dose_bin=r["dose_bin"],
        ranked_genes=list(r["input_gene_ranked_list"]),
        lfc_vector=np.asarray(r["label_lfc_vector"], dtype=np.float32),
        tokenizer_op=L.tokenizer_op,
    )
    print(f"  sample token len: {s['data.encoder_input_token_ids'].shape}")

    print("\n[smoke] running forward ...")
    t1 = time.time()
    out = forward_with_hidden(L, [s])
    print(f"  forward done in {time.time()-t1:.2f}s")
    print(f"  last_hidden: {tuple(out.last_hidden.shape)}")
    print(f"  pred:        {tuple(out.pred.shape)}")
    print(f"  token_ids:   {tuple(out.token_ids.shape)}")
    print(f"  attn_mask:   {tuple(out.attention_mask.shape)}")

    spans = find_spans(out.token_ids[0].cpu(), L.special_token_ids,
                       out.attention_mask[0].cpu())
    print(f"\n[smoke] spans:")
    print(f"  mask_pos:    {spans.mask}")
    print(f"  smiles:      [{spans.smiles_start}, {spans.smiles_end})  len={spans.smiles_end-spans.smiles_start}")
    print(f"  dose_pos:    {spans.dose_pos}  (token_id={int(out.token_ids[0,spans.dose_pos])})")
    print(f"  gene:        [{spans.gene_start}, {spans.gene_end})  len={spans.gene_end-spans.gene_start}")
    print(f"  eos:         {spans.eos}")
    print(f"  valid_len:   {spans.valid_len}")
    print(f"\n[smoke] OK  (total {time.time()-t0:.1f}s)")


if __name__ == "__main__":
    main()
