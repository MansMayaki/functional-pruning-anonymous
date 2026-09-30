#!/usr/bin/env python3
"""Aggregate raw Table 1 JSON files into paper-ready CSVs and LaTeX."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import t as student_t

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / "expected" / "table1_reference.csv"
EXPECTED_SEEDS = [0, 1, 2, 3, 4]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--raw-dir", type=Path, default=ROOT / "results" / "table1" / "raw")
    p.add_argument("--out-dir", type=Path, default=ROOT / "results" / "table1" / "summary")
    p.add_argument("--verify", action="store_true")
    return p.parse_args()


def load_rows(raw_dir):
    rows = []
    for p in sorted(raw_dir.glob("*.json")):
        with p.open() as f:
            o = json.load(f)
        if o.get("saliency") != "taylor" or o.get("finetune") not in {"fixed", "output"}:
            continue
        r = {
            "dataset": o["dataset"],
            "seed": int(o["seed"]),
            "finetune": o["finetune"],
            "dense_accuracy": 100.0 * float(o["dense_accuracy"]),
        }
        for k, v in o["final"].items():
            if isinstance(v, (int, float)):
                r[k] = float(v)
        r["accuracy"] *= 100.0
        rows.append(r)
    return pd.DataFrame(rows)


def check_complete(df):
    for ds in ["mnist", "fmnist"]:
        for ft in ["fixed", "output"]:
            q = df[(df.dataset == ds) & (df.finetune == ft)]
            seeds = sorted(q.seed.unique().tolist())
            if seeds != EXPECTED_SEEDS:
                raise RuntimeError(f"Incomplete {ds}/{ft}: expected {EXPECTED_SEEDS}, got {seeds}")


def paired_ci(x):
    x = np.asarray(x, dtype=float)
    n = len(x)
    mean = x.mean()
    sd = x.std(ddof=1)
    half = student_t.ppf(0.975, n - 1) * sd / math.sqrt(n)
    return mean, mean - half, mean + half


def aggregate(df):
    metrics = ["accuracy", "D_fix", "D_gate", "D_func", "kappa"]
    rows = []
    for ds in ["mnist", "fmnist"]:
        for ft in ["output", "fixed"]:
            q = df[(df.dataset == ds) & (df.finetune == ft)].sort_values("seed")
            row = {"dataset": ds, "finetune": ft}
            for m in metrics:
                row[f"{m}_mean"] = q[m].mean()
                row[f"{m}_sd"] = q[m].std(ddof=1)
            row["dense_accuracy_mean"] = q.dense_accuracy.mean()
            row["dense_accuracy_sd"] = q.dense_accuracy.std(ddof=1)
            rows.append(row)
    return pd.DataFrame(rows)


def paired_differences(df):
    metrics = ["accuracy", "D_fix", "D_func", "kappa"]
    rows = []
    for ds in ["mnist", "fmnist"]:
        a = df[(df.dataset == ds) & (df.finetune == "fixed")].set_index("seed")
        b = df[(df.dataset == ds) & (df.finetune == "output")].set_index("seed")
        common = sorted(a.index.intersection(b.index))
        for m in metrics:
            delta = (a.loc[common, m] - b.loc[common, m]).to_numpy()
            mean, lo, hi = paired_ci(delta)
            rows.append({
                "dataset": ds,
                "delta_definition": "fixed-output",
                "metric": "accuracy_pp" if m == "accuracy" else m,
                "mean": mean,
                "ci95_low": lo,
                "ci95_high": hi,
            })
    return pd.DataFrame(rows)


def write_latex(agg, out_path):
    def row(ds, ft, label):
        q = agg[(agg.dataset == ds) & (agg.finetune == ft)].iloc[0]
        return (
            f"& {label} & ${q.accuracy_mean:.2f}\\pm{q.accuracy_sd:.2f}$ "
            f"& ${q.D_fix_mean:.2f}\\pm{q.D_fix_sd:.2f}$ "
            f"& ${q.D_func_mean:.2f}\\pm{q.D_func_sd:.2f}$ "
            f"& ${q.kappa_mean:.3f}\\pm{q.kappa_sd:.3f}$\\\\"
        )
    text = "\n".join([
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Controlled comparison at $90\%$ sparsity using the same task-Taylor saliency rule. Results are mean $\pm$ sample standard deviation over five paired seeds.}",
        r"\label{tab:image}",
        r"\small",
        r"\begin{tabular}{llrrrr}",
        r"\toprule",
        r"Data & Fine-tuning & Acc. & $D_{\rm fix}$ & $D_{\rm func}$ & $\kappa$\\",
        r"\midrule",
        r"\multirow{2}{*}{MNIST}",
        row("mnist", "output", "Output"),
        row("mnist", "fixed", "Fixed-gate"),
        r"\midrule",
        r"\multirow{2}{*}{Fashion-MNIST}",
        row("fmnist", "output", "Output"),
        row("fmnist", "fixed", "Fixed-gate"),
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table*}",
        "",
    ])
    out_path.write_text(text)


def verify(agg):
    ref = pd.read_csv(REFERENCE)
    merged = agg.merge(ref, on=["dataset", "finetune"], suffixes=("_run", "_ref"))
    # Hardware/library changes can alter final digits. Report, do not silently ignore.
    cols = ["accuracy_mean", "D_fix_mean", "D_func_mean", "kappa_mean"]
    print("\nReference comparison (run - paper):")
    for _, r in merged.iterrows():
        print(f"{r.dataset:7s} {r.finetune:6s}", end="")
        for c in cols:
            print(f"  {c}={r[c + '_run'] - r[c + '_ref']:+.4g}", end="")
        print()


def main():
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    df = load_rows(args.raw_dir)
    check_complete(df)
    agg = aggregate(df)
    paired = paired_differences(df)

    df.to_csv(args.out_dir / "seed_level.csv", index=False)
    agg.to_csv(args.out_dir / "table1.csv", index=False)
    paired.to_csv(args.out_dir / "paired_ci.csv", index=False)
    write_latex(agg, args.out_dir / "table1.tex")

    print("\nTable 1 aggregate:\n")
    print(agg.to_string(index=False, float_format=lambda x: f"{x:.5f}"))
    print("\nPaired 95% CIs (fixed - output):\n")
    print(paired.to_string(index=False, float_format=lambda x: f"{x:.5f}"))
    if args.verify:
        verify(agg)


if __name__ == "__main__":
    main()
