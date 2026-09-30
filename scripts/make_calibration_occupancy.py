#!/usr/bin/env python3
"""Aggregate calibration-pattern occupancy diagnostics into CSV and figure."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--raw-dir", type=Path, default=ROOT / "results" / "calibration")
    p.add_argument("--out-prefix", type=Path, default=ROOT / "figures" / "calibration_occupancy")
    return p.parse_args()


def main():
    args = parse_args()
    rows = []
    for p in sorted(args.raw_dir.glob("*.json")):
        with p.open() as f:
            o = json.load(f)
        occ = o["occupancy"]
        final = o["final"]
        rows.append({
            "dataset": o["dataset"], "seed": o["seed"], "calib_size": o["calib_size"],
            "n_unique_patterns": occ["n_unique_patterns"],
            "unique_fraction": occ["unique_fraction"],
            "singleton_pattern_fraction": occ["singleton_pattern_fraction"],
            "singleton_sample_fraction": occ["singleton_sample_fraction"],
            "occupancy_median": occ["occupancy_median"],
            "max_occupancy": occ["max_occupancy"],
            "D_fix": final["D_fix"],
            "jacobian": final["test_fixed_input_jacobian_diff"],
        })
    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError(f"No calibration JSON files in {args.raw_dir}")
    summary = df.groupby("calib_size").agg(["mean", "std"])
    out_csv = args.out_prefix.with_suffix(".csv")
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)

    fig, axes = plt.subplots(1, 4, figsize=(10.8, 2.8))
    x = sorted(df.calib_size.unique())
    specs = [
        ("n_unique_patterns", "Unique patterns"),
        ("occupancy_median", "Median occupancy"),
        ("D_fix", r"Test $D_{\rm fix}$"),
        ("jacobian", "Test Jacobian discrepancy"),
    ]
    for ax, (m, label) in zip(axes, specs):
        g = df.groupby("calib_size")[m].agg(["mean", "std"]).reindex(x)
        ax.errorbar(x, g["mean"], yerr=g["std"], marker="o", capsize=3)
        ax.set_xscale("log", base=2)
        ax.set_xlabel(r"$N_{\rm cal}$")
        ax.set_ylabel(label)
    fig.tight_layout()
    fig.savefig(args.out_prefix.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(out_csv)
    print(args.out_prefix.with_suffix(".pdf"))


if __name__ == "__main__":
    main()
