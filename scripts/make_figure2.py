#!/usr/bin/env python3
"""Generate fine-tuning dynamics at the final 90% pruning stage."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
EXPECTED_SEEDS = [0, 1, 2, 3, 4]
METRICS = [("D_fix", r"$D_{\rm fix}$"), ("D_gate", r"$D_{\rm gate}$"), ("D_func", r"$D_{\rm func}$"), ("twoC", r"$2\mathcal{C}$")]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--raw-dir", type=Path, default=ROOT / "results" / "table1" / "raw")
    p.add_argument("--out-prefix", type=Path, default=ROOT / "figures" / "figure2_finetuning_dynamics")
    p.add_argument("--dpi", type=int, default=300)
    return p.parse_args()


def load(raw_dir):
    rows = []
    for p in sorted(raw_dir.glob("*.json")):
        with p.open() as f:
            o = json.load(f)
        if o.get("saliency") != "taylor" or o.get("finetune") not in {"fixed", "output"}:
            continue
        for t in o["trajectory"]:
            if not np.isclose(float(t["target_sparsity"]), 0.90):
                continue
            r = {"dataset": o["dataset"], "seed": int(o["seed"]), "finetune": o["finetune"]}
            r.update({k: v for k, v in t.items() if isinstance(v, (int, float, str))})
            rows.append(r)
    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError("No 90% controlled trajectories found")
    return df


def main():
    args = parse_args()
    df = load(args.raw_dir)
    for ds in ["mnist", "fmnist"]:
        for ft in ["fixed", "output"]:
            seeds = sorted(df[(df.dataset == ds) & (df.finetune == ft)].seed.unique())
            if seeds != EXPECTED_SEEDS:
                raise RuntimeError(f"Incomplete {ds}/{ft}: {seeds}")

    fig, axes = plt.subplots(2, 4, figsize=(13.2, 5.6), sharex=True)
    for row, (ds, title) in enumerate([("mnist", "MNIST"), ("fmnist", "Fashion-MNIST")]):
        for col, (metric, ylabel) in enumerate(METRICS):
            ax = axes[row, col]
            for ft in ["fixed", "output"]:
                q = df[(df.dataset == ds) & (df.finetune == ft)]
                g = q.groupby("epoch")[metric].agg(["mean", "std"]).reset_index()
                ax.errorbar(g.epoch, g["mean"], yerr=g["std"], marker="o", capsize=3, linewidth=2,
                            label="fixed-gate FT" if ft == "fixed" else "output FT")
            if metric == "twoC":
                ax.axhline(0, linestyle=":", linewidth=1)
            ax.set_ylabel(ylabel)
            ax.set_xticks([0, 1, 2, 3])
            if row == 1:
                ax.set_xlabel("Fine-tuning epoch")
            if col == 0:
                ax.legend(frameon=False, fontsize=8)
        axes[row, 1].set_title(f"{title}: task-Taylor saliency, 90% pruning", fontsize=11)

    fig.tight_layout()
    args.out_prefix.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out_prefix.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(args.out_prefix.with_suffix(".png"), dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)
    print(args.out_prefix.with_suffix(".pdf"))


if __name__ == "__main__":
    main()
