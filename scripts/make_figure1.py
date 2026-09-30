#!/usr/bin/env python3
"""Generate the controlled sparsity-trajectory figure from Table 1 runs.

All curves use task-Taylor saliency. The two branches differ only in the
fine-tuning objective (fixed-gate vs ordinary output matching).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
EXPECTED_SEEDS = [0, 1, 2, 3, 4]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--raw-dir", type=Path, default=ROOT / "results" / "table1" / "raw")
    p.add_argument("--out-prefix", type=Path, default=ROOT / "figures" / "figure1_controlled")
    p.add_argument("--dpi", type=int, default=300)
    return p.parse_args()


def load_trajectory(raw_dir):
    rows = []
    for p in sorted(raw_dir.glob("*.json")):
        with p.open() as f:
            o = json.load(f)
        if o.get("saliency") != "taylor" or o.get("finetune") not in {"fixed", "output"}:
            continue
        base = {"dataset": o["dataset"], "seed": int(o["seed"]), "finetune": o["finetune"]}
        for t in o["trajectory"]:
            r = dict(base)
            r.update({k: v for k, v in t.items() if isinstance(v, (int, float, str))})
            rows.append(r)
    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError(f"No controlled trajectory JSON files in {raw_dir}")
    # final fine-tuning epoch separately at each sparsity stage
    keys = ["dataset", "seed", "finetune", "target_sparsity"]
    df = df[df.epoch == df.groupby(keys).epoch.transform("max")].copy()
    return df


def check_complete(df):
    sparsities = sorted(df.target_sparsity.unique())
    for s in sparsities:
        for ds in ["mnist", "fmnist"]:
            for ft in ["fixed", "output"]:
                q = df[(df.dataset == ds) & (df.finetune == ft) & np.isclose(df.target_sparsity, s)]
                seeds = sorted(q.seed.unique().tolist())
                if seeds != EXPECTED_SEEDS:
                    raise RuntimeError(f"Incomplete {ds}/{ft}/{s}: {seeds}")
    return sparsities


def paired_ratio(df, ds, s, metric):
    q = df[(df.dataset == ds) & np.isclose(df.target_sparsity, s)]
    a = q[q.finetune == "fixed"].set_index("seed").sort_index()[metric]
    b = q[q.finetune == "output"].set_index("seed").sort_index()[metric]
    central = a.mean() / b.mean()  # exactly aligned with Table 1 aggregation
    paired = (a / b).to_numpy()
    return central, paired.std(ddof=1)


def kappa_stats(df, ds, s, ft):
    q = df[(df.dataset == ds) & (df.finetune == ft) & np.isclose(df.target_sparsity, s)].sort_values("seed")
    x = q.kappa.to_numpy()
    return x.mean(), x.std(ddof=1)


def main():
    args = parse_args()
    df = load_trajectory(args.raw_dir)
    sparsities = check_complete(df)
    x = np.asarray(sparsities) * 100.0

    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.2))
    datasets = [("mnist", "MNIST"), ("fmnist", "Fashion-MNIST")]

    for ds, label in datasets:
        for ax, metric in zip(axes[:2], ["D_fix", "D_func"]):
            means, sds = [], []
            for s in sparsities:
                m, sd = paired_ratio(df, ds, s, metric)
                means.append(m); sds.append(sd)
            means, sds = np.asarray(means), np.asarray(sds)
            ax.plot(x, means, marker="o", linewidth=2, label=label)
            ax.fill_between(x, means - sds, means + sds, alpha=0.15)

        for ft, ls in [("fixed", "-"), ("output", "--")]:
            means, sds = [], []
            for s in sparsities:
                m, sd = kappa_stats(df, ds, s, ft)
                means.append(m); sds.append(sd)
            means, sds = np.asarray(means), np.asarray(sds)
            label2 = f"{label}, {'fixed-gate' if ft == 'fixed' else 'output'}"
            axes[2].plot(x, means, marker="o", linestyle=ls, linewidth=2, label=label2)
            axes[2].fill_between(x, means - sds, means + sds, alpha=0.10)

    axes[0].axhline(1.0, linestyle=":", linewidth=1)
    axes[1].axhline(1.0, linestyle=":", linewidth=1)
    axes[2].axhline(0.0, linestyle=":", linewidth=1)
    axes[0].set_title(r"(a) Reference-gate error $D_{\rm fix}$")
    axes[1].set_title(r"(b) Realized error $D_{\rm func}$")
    axes[2].set_title(r"(c) Cancellation index $\kappa$")
    axes[0].set_ylabel("Fixed-gate / output matching")
    axes[1].set_ylabel("Fixed-gate / output matching")
    axes[2].set_ylabel(r"$\kappa$")
    axes[0].legend(frameon=False, fontsize=8)
    axes[2].legend(frameon=False, fontsize=7)
    for ax in axes:
        ax.set_xlabel("Sparsity (%)")
        ax.set_xticks(x)
        ax.grid(axis="y", alpha=0.15)
    axes[0].set_ylim(bottom=0)
    axes[1].set_ylim(bottom=0)
    axes[2].set_ylim(-1.02, 0.15)
    fig.tight_layout()

    args.out_prefix.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out_prefix.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(args.out_prefix.with_suffix(".png"), dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)
    print(args.out_prefix.with_suffix(".pdf"))


if __name__ == "__main__":
    main()
