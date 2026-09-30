#!/usr/bin/env python3
"""
Empirical checks for the layerwise gate-change theory and gate-margin lemma.

Input is a .pt checkpoint produced by mlp_experiments.py
(factorial mode). No retraining is required.

Reports:
  - actual ||v_gate|| per example;
  - RHS of the norm bound in Corollary / Eq. 7;
  - bound/actual ratio distribution;
  - Pearson and Spearman correlation between bound and actual gate error;
  - exact telescoping reconstruction residual;
  - mean contribution norm by hidden layer;
  - gate-flip prediction from dense preactivation margin (ROC-AUC);
  - flip rate by dense-margin quantile;
  - numerical check of the gate-margin implication.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import List

import numpy as np
import torch

THIS = Path(__file__).resolve().parent
if str(THIS) not in sys.path:
    sys.path.insert(0, str(THIS))

from mlp_experiments import (  # noqa: E402
    ReLUMLP,
    get_dataset,
    make_loader,
    seed_all,
)

try:
    from scipy.stats import spearmanr, pearsonr
except Exception:
    spearmanr = pearsonr = None

try:
    from sklearn.metrics import roc_auc_score
except Exception:
    roc_auc_score = None


def safe_corr(x, y, kind="pearson"):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
        return None
    if kind == "pearson" and pearsonr is not None:
        return float(pearsonr(x, y).statistic)
    if kind == "spearman" and spearmanr is not None:
        return float(spearmanr(x, y).statistic)
    if kind == "pearson":
        return float(np.corrcoef(x, y)[0, 1])
    # fallback rank correlation
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    return float(np.corrcoef(rx, ry)[0, 1])


def quantile_summary(x):
    x = np.asarray(x, dtype=float)
    if len(x) == 0:
        return {}
    return {
        "mean": float(x.mean()),
        "median": float(np.median(x)),
        "p10": float(np.quantile(x, 0.10)),
        "p25": float(np.quantile(x, 0.25)),
        "p75": float(np.quantile(x, 0.75)),
        "p90": float(np.quantile(x, 0.90)),
        "p95": float(np.quantile(x, 0.95)),
        "p99": float(np.quantile(x, 0.99)),
        "max": float(x.max()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True, type=Path)
    ap.add_argument("--data-dir", default="./data")
    ap.add_argument("--max-samples", type=int, default=1000)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--output", type=Path, default=None)
    args = ap.parse_args()

    ck = torch.load(args.checkpoint, map_location="cpu")
    dataset = ck["dataset"]
    seed = int(ck["seed"])
    hidden = list(map(int, ck["hidden"]))
    in_dim = int(ck["in_dim"])
    out_dim = int(ck["out_dim"])
    seed_all(seed)
    device = torch.device(args.device)

    _, test_ds, in_dim2, out_dim2 = get_dataset(dataset, args.data_dir, seed)
    assert in_dim == in_dim2 and out_dim == out_dim2

    dense = ReLUMLP(in_dim, hidden, out_dim).to(device)
    sparse = ReLUMLP(in_dim, hidden, out_dim).to(device)
    dense.load_state_dict(ck["dense_state_dict"])
    sparse.load_state_dict(ck["sparse_state_dict"])
    dense.eval(); sparse.eval()

    loader = make_loader(test_ds, args.batch_size, False, 0, args.num_workers)

    actual_all: List[float] = []
    bound_all: List[float] = []
    ratio_all: List[float] = []
    exact_resid_all: List[float] = []
    layer_term_norms = [[] for _ in hidden]
    layer_bound_terms = [[] for _ in hidden]

    margins = [[] for _ in hidden]
    flips = [[] for _ in hidden]
    perturb_ratio = [[] for _ in hidden]
    margin_implication_violations = 0
    margin_implication_flips = 0

    seen = 0
    with torch.no_grad():
        for xb, _ in loader:
            if seen >= args.max_samples:
                break
            xb = xb[: args.max_samples - seen].to(device)
            dlog, dgates, dpre = dense(
                xb, return_gates=True, return_preacts=True
            )
            slog, sgates, spre = sparse(
                xb, return_gates=True, return_preacts=True
            )
            s_under_d = sparse.forward_with_gates(xb, dgates)
            vgate_batch = s_under_d - slog

            for b in range(xb.shape[0]):
                # Start with B_{s,H-1} = W_out.
                B = sparse.output_layer.weight.detach()  # C x H_last
                term_vecs = [None] * len(hidden)
                bterms = [0.0] * len(hidden)

                for k in range(len(hidden) - 1, -1, -1):
                    delta_gate_z = (
                        dgates[k][b].to(spre[k].dtype)
                        - sgates[k][b].to(spre[k].dtype)
                    ) * spre[k][b]
                    term_vec = B @ delta_gate_z
                    term_vecs[k] = term_vec
                    # spectral norm is exact for the matrix B.
                    spec = torch.linalg.matrix_norm(B, ord=2)
                    bterm = float((spec * delta_gate_z.norm()).item())
                    bterms[k] = bterm
                    layer_term_norms[k].append(float(term_vec.norm().item()))
                    layer_bound_terms[k].append(bterm)

                    # Update to B_{s,k-1} = B_{s,k} D_s^k W_k.
                    if k > 0:
                        B = B * dgates[k][b].to(B.dtype).unsqueeze(0)
                        B = B @ sparse.hidden_layers[k].weight.detach()

                exact_sum = torch.stack(term_vecs).sum(dim=0)
                actual = float(vgate_batch[b].norm().item())
                rhs = float(sum(bterms))
                residual = float((exact_sum - vgate_batch[b]).norm().item())
                actual_all.append(actual)
                bound_all.append(rhs)
                exact_resid_all.append(residual)
                if actual > 1e-12:
                    ratio_all.append(rhs / actual)

            # Margin diagnostics across all hidden units.
            for k in range(len(hidden)):
                margin = dpre[k].abs().detach().cpu().numpy().reshape(-1)
                flip = (dgates[k] != sgates[k]).detach().cpu().numpy().astype(np.int8).reshape(-1)
                dz = (spre[k] - dpre[k]).abs().detach().cpu().numpy().reshape(-1)
                ratio = dz / np.maximum(margin, 1e-12)
                margins[k].append(margin)
                flips[k].append(flip)
                perturb_ratio[k].append(ratio)
                if flip.sum() > 0:
                    margin_implication_flips += int(flip.sum())
                    margin_implication_violations += int(((ratio < 1.0 - 1e-6) & (flip == 1)).sum())

            seen += xb.shape[0]

    actual_arr = np.asarray(actual_all)
    bound_arr = np.asarray(bound_all)
    ratio_arr = np.asarray(ratio_all)
    resid_arr = np.asarray(exact_resid_all)

    margin_results = []
    for k in range(len(hidden)):
        m = np.concatenate(margins[k]) if margins[k] else np.array([])
        f = np.concatenate(flips[k]) if flips[k] else np.array([])
        r = np.concatenate(perturb_ratio[k]) if perturb_ratio[k] else np.array([])
        auc_margin = None
        auc_ratio = None
        if roc_auc_score is not None and len(np.unique(f)) == 2:
            auc_margin = float(roc_auc_score(f, -m))
            auc_ratio = float(roc_auc_score(f, r))
        qedges = np.quantile(m, [0, .1, .25, .5, .75, .9, 1.0]) if len(m) else np.array([])
        bins = []
        if len(qedges):
            for qi in range(len(qedges) - 1):
                lo, hi = qedges[qi], qedges[qi + 1]
                mask = (m >= lo) & (m <= hi if qi == len(qedges) - 2 else m < hi)
                bins.append(
                    {
                        "lo": float(lo),
                        "hi": float(hi),
                        "n": int(mask.sum()),
                        "flip_rate": float(f[mask].mean()) if mask.any() else None,
                    }
                )
        margin_results.append(
            {
                "layer": k + 1,
                "n_units_examples": int(len(m)),
                "flip_rate": float(f.mean()) if len(f) else None,
                "auc_negative_dense_margin": auc_margin,
                "auc_perturbation_over_margin": auc_ratio,
                "margin_quantile_bins": bins,
            }
        )

    result = {
        "checkpoint": str(args.checkpoint),
        "dataset": dataset,
        "seed": seed,
        "hidden": hidden,
        "saliency": ck.get("saliency"),
        "finetune": ck.get("finetune"),
        "n_test_examples": int(seen),
        "gate_error": {
            "actual_norm": quantile_summary(actual_arr),
            "bound_rhs": quantile_summary(bound_arr),
            "bound_over_actual": quantile_summary(ratio_arr),
            "pearson_bound_vs_actual": safe_corr(bound_arr, actual_arr, "pearson"),
            "spearman_bound_vs_actual": safe_corr(bound_arr, actual_arr, "spearman"),
            "exact_telescope_residual": quantile_summary(resid_arr),
            "mean_layer_exact_term_norm": [float(np.mean(x)) for x in layer_term_norms],
            "mean_layer_bound_term": [float(np.mean(x)) for x in layer_bound_terms],
        },
        "margin": {
            "per_layer": margin_results,
            "total_flips_checked": int(margin_implication_flips),
            "margin_implication_violations": int(margin_implication_violations),
            "violation_rate": float(margin_implication_violations / max(margin_implication_flips, 1)),
        },
    }

    if args.output is None:
        args.output = args.checkpoint.with_name(args.checkpoint.stem + "_theory.json")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as f:
        json.dump(result, f, indent=2, sort_keys=True)
    print(args.output)


if __name__ == "__main__":
    main()
