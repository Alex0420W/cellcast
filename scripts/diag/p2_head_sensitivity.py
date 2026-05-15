"""P2 — Head sensitivity to its input.

Take the trained head. Find the two most-different real <MASK> hidden states
from P1. Linearly interpolate between them (100 steps) and feed each synthetic
vector through the head. Plot output L2 distance from the midpoint vs
interpolation parameter, and compute the head Jacobian norm at one anchor.
"""
from __future__ import annotations
import json, os, sys, time
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0, os.path.expanduser("~/cellcast"))

from scripts.diag._lib import OUT_ROOT, load_model, P1_DRUGS

OUT = OUT_ROOT / "p2"
N_INTERP = 100
JACOBIAN_PROBE_DIM_LIMIT = 7153  # full HVG output


def main():
    t0 = time.time()
    print("[P2] loading model + P1 outputs ...", flush=True)
    L = load_model()
    head = L.head
    head.eval()

    p1 = np.load(OUT_ROOT / "p1" / "p1_distances.npz")
    # Pick the K562 mask vectors (most cells, mid-magnitude pcorr); choose the
    # two drugs with maximum pairwise cosine distance at MASK in K562.
    cl = "K562"
    mask_vecs = p1[f"{cl}__mask__vec"]   # [5, D]
    cosdist = p1[f"{cl}__mask__cosdist"]  # [5, 5]
    # find argmax pair
    n = cosdist.shape[0]
    best_pair = (0, 1)
    best_v = -1.0
    for i in range(n):
        for j in range(i + 1, n):
            if cosdist[i, j] > best_v:
                best_v = cosdist[i, j]
                best_pair = (i, j)
    labels = list(p1["drug_labels"])
    print(f"  most distant MASK pair in {cl}: ({labels[best_pair[0]]}) vs ({labels[best_pair[1]]})  "
          f"cosdist={best_v:.4f}", flush=True)

    a = torch.from_numpy(mask_vecs[best_pair[0]]).to(L.device, dtype=torch.float32)
    b = torch.from_numpy(mask_vecs[best_pair[1]]).to(L.device, dtype=torch.float32)

    # Build interpolated batch and run through head
    alphas = torch.linspace(0.0, 1.0, N_INTERP, device=L.device)
    # Head expects [B, S, D] (it was applied to last_hidden); for the head's MLP
    # path the sequence axis is benign — we pass [B, 1, D] and slice.
    interp = (1 - alphas).view(-1, 1) * a.view(1, -1) + alphas.view(-1, 1) * b.view(1, -1)  # [N, D]
    interp_seq = interp.unsqueeze(1)  # [N, 1, D]

    print("[P2] running head on interpolated MASK vectors ...", flush=True)
    with torch.inference_mode():
        head_out = head(interp_seq)  # [N, 1, G]
    head_out = head_out.squeeze(1).float().cpu().numpy()  # [N, G]

    midpoint = head_out[N_INTERP // 2]  # [G]
    l2_from_mid = np.linalg.norm(head_out - midpoint[None, :], axis=1)  # [N]
    end_l2 = np.linalg.norm(head_out[-1] - head_out[0])

    # Endpoint vs midpoint distances + norm of midpoint output
    print(f"  ||head(a) - head(b)||_2 = {end_l2:.4f}")
    print(f"  ||head(midpoint)||_2    = {np.linalg.norm(midpoint):.4f}")
    print(f"  max ||head(interp) - head(mid)||_2 = {l2_from_mid.max():.4f}")

    # ---- Jacobian norm at one real MASK point (anchor = drug 0 in K562) ----
    print("[P2] computing Jacobian Frobenius norm at anchor ...", flush=True)
    anchor = a.clone().detach().requires_grad_(True)  # [D]
    head_out_anchor = head(anchor.view(1, 1, -1)).squeeze(0).squeeze(0)  # [G]
    # Functional Jacobian (G x D); torch.autograd.functional may be heavy at G=7153
    jacobian = torch.autograd.functional.jacobian(
        lambda x: head(x.view(1, 1, -1)).squeeze(),
        anchor,
        create_graph=False,
        vectorize=True,
    )  # [G, D]
    jac_fro = jacobian.norm().item()
    jac_op = torch.linalg.matrix_norm(jacobian.float(), ord=2).item()
    head_param_norm = sum(p.norm().item() ** 2 for p in head.parameters() if p.requires_grad) ** 0.5
    head_param_count = sum(p.numel() for p in head.parameters())
    head_param_count_train = sum(p.numel() for p in head.parameters() if p.requires_grad)

    # Also compute "what fraction of input perturbation reaches output":
    # compare ||J @ delta|| / ||delta|| for delta = b - a normalized
    delta = (b - a).detach()
    delta_norm = delta.norm().item()
    direction = delta / (delta_norm + 1e-12)
    response = (jacobian @ direction).norm().item()
    # And the worst-case direction:
    sv = torch.linalg.svdvals(jacobian.float())
    sv_max = float(sv[0])
    sv_min = float(sv[-1])
    sv_med = float(sv[len(sv) // 2])

    summary = {
        "anchor_drug": labels[best_pair[0]],
        "endpoint_drug": labels[best_pair[1]],
        "anchor_endpoint_cosdist": float(best_v),
        "head_output_l2_endpoint_to_endpoint": float(end_l2),
        "head_output_l2_max_from_midpoint": float(l2_from_mid.max()),
        "head_output_l2_min_from_midpoint": float(l2_from_mid.min()),
        "head_output_norm_at_midpoint": float(np.linalg.norm(midpoint)),
        "jacobian_frobenius_norm": float(jac_fro),
        "jacobian_operator_norm_largest_sv": float(sv_max),
        "jacobian_smallest_sv": float(sv_min),
        "jacobian_median_sv": float(sv_med),
        "head_parameter_l2_norm": float(head_param_norm),
        "head_parameter_count_total": int(head_param_count),
        "head_parameter_count_trainable": int(head_param_count_train),
        "input_perturb_norm_a_to_b": float(delta_norm),
        "head_response_norm_to_a_to_b_perturb": float(response),
        "n_interp": int(N_INTERP),
    }
    (OUT / "p2_summary.json").write_text(json.dumps(summary, indent=2))
    np.savez(OUT / "p2_traces.npz",
             alphas=alphas.cpu().numpy(),
             head_out_l2_from_mid=l2_from_mid,
             head_out_norms=np.linalg.norm(head_out, axis=1),
             head_out_anchor=head_out[0],
             head_out_endpoint=head_out[-1],
             jacobian_singular_values=sv.cpu().numpy())

    # ---- Plot interp curve ----
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), dpi=150)
    axes[0].plot(alphas.cpu().numpy(), l2_from_mid, lw=1.5)
    axes[0].axhline(end_l2 / 2, ls="--", c="grey",
                    label=f"||head(a)-head(b)||/2 = {end_l2/2:.3f}")
    axes[0].set_xlabel("interpolation alpha (0=anchor, 1=endpoint)")
    axes[0].set_ylabel("||head(interp) - head(midpoint)||_2")
    axes[0].set_title(f"P2 head response to interpolated MASK\n"
                      f"K562  {labels[best_pair[0]]} -> {labels[best_pair[1]]}")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3)

    axes[1].semilogy(np.arange(len(sv)), sv.cpu().numpy(), lw=1)
    axes[1].set_xlabel("singular value index")
    axes[1].set_ylabel("singular value (log)")
    axes[1].set_title(f"P2 head Jacobian singular values at anchor\n"
                      f"max={sv_max:.3f} med={sv_med:.4f} min={sv_min:.2e}")
    axes[1].grid(alpha=0.3, which="both")

    fig.tight_layout()
    fig.savefig(OUT / "p2_interp_and_jacobian.png")
    plt.close(fig)
    print(f"  wrote {OUT / 'p2_interp_and_jacobian.png'}")
    print(f"  wrote {OUT / 'p2_summary.json'}")
    print(f"\n[P2] Headline numbers:")
    print(f"  head Jacobian Frobenius norm:    {jac_fro:.3f}")
    print(f"  head Jacobian operator norm:     {sv_max:.3f}")
    print(f"  head parameter L2 norm:          {head_param_norm:.3f}")
    print(f"  Frobenius / param-norm ratio:    {jac_fro / head_param_norm:.4f}")
    print(f"  ||head(a)-head(b)|| / ||a-b|| =  {end_l2 / delta_norm:.3f}  (effective input->output gain)")
    print(f"\n[P2] DONE  {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
