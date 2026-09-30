#!/usr/bin/env python3
"""Reproduce the controlled Table 1 experiment from scratch.

The experiment holds task-Taylor saliency fixed and changes only the
fine-tuning objective (fixed-gate matching vs ordinary output matching).
For each (dataset, seed), both branches reuse the same dense checkpoint,
calibration subset/order, pruning targets, and deterministic minibatch schedule.

Examples
--------
Single GPU:
    python scripts/reproduce_table1.py --gpus 0

Two GPUs:
    python scripts/reproduce_table1.py --gpus 0 1

CPU smoke/debug run:
    python scripts/reproduce_table1.py --device cpu --gpus cpu
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CORE = ROOT / "src" / "mlp_experiments.py"
CONFIG = ROOT / "configs" / "table1.json"
SUMMARY = ROOT / "scripts" / "summarize_table1.py"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--gpus", nargs="+", default=["0"], help="GPU ids, e.g. 0 1. Use 'cpu' with --device cpu.")
    p.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    p.add_argument("--data-dir", default=str(ROOT / "data"))
    p.add_argument("--results-dir", default=str(ROOT / "results" / "table1"))
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--force", action="store_true", help="Rerun jobs even when result JSON already exists.")
    p.add_argument("--verify", action="store_true", help="Compare aggregate output with the paper reference table.")
    return p.parse_args()


def load_config():
    with CONFIG.open() as f:
        return json.load(f)


def command_for(cfg, dataset, seed, ft, results_dir, data_dir, num_workers, device):
    d = cfg["dense_training"]
    p = cfg["pruning"]
    f = cfg["fine_tuning"]
    c = cfg["calibration"]
    hidden = cfg["architecture"]["hidden"]
    raw = results_dir / "raw"
    dense = results_dir / "dense_cache"
    return [
        sys.executable, str(CORE), "factorial",
        "--dataset", dataset,
        "--seed", str(seed),
        "--hidden", *map(str, hidden),
        "--data-dir", str(data_dir),
        "--dense-cache-dir", str(dense),
        "--output-dir", str(raw),
        "--dense-epochs", str(d["epochs"]),
        "--ft-epochs", str(f["epochs_per_stage"]),
        "--batch-size", str(d["batch_size"]),
        "--eval-batch-size", str(d["batch_size"]),
        "--num-workers", str(num_workers),
        "--lr", str(d["learning_rate"]),
        "--lambda-reg", str(f["matching_coefficient"]),
        "--calib-size", str(c["max_examples"]),
        "--saliency-batches", str(p["saliency_batches"]),
        "--sparsities", *map(str, p["progressive_target_sparsities"]),
        "--saliency", "taylor",
        "--finetune", ft,
        "--device", device,
    ]


def run_group(gpu, jobs, results_dir, force):
    logs = results_dir / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    raw = results_dir / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    if gpu != "cpu":
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)

    for dataset, seed, ft, cmd in jobs:
        out_json = raw / f"{dataset}_seed{seed}_taylor_{ft}.json"
        if out_json.exists() and not force:
            print(f"[skip] {out_json.name}")
            continue
        name = f"{dataset}_seed{seed}_taylor_{ft}"
        log = logs / f"{name}.log"
        print(f"[{gpu}] START {name}")
        with log.open("w") as fp:
            proc = subprocess.run(cmd, env=env, stdout=fp, stderr=subprocess.STDOUT)
        if proc.returncode != 0:
            raise RuntimeError(f"Job failed: {name}. See {log}")
        print(f"[{gpu}] DONE  {name}")


def main():
    args = parse_args()
    cfg = load_config()
    results_dir = Path(args.results_dir)
    data_dir = Path(args.data_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    # Keep both fine-tuning branches for a given dataset/seed in the same group.
    # This guarantees that the dense checkpoint is created once and reused.
    groups = []
    for dataset in cfg["datasets"]:
        for seed in cfg["seeds"]:
            jobs = []
            for ft in cfg["fine_tuning"]["objectives"]:
                cmd = command_for(
                    cfg, dataset, seed, ft, results_dir, data_dir,
                    args.num_workers, args.device,
                )
                jobs.append((dataset, seed, ft, cmd))
            groups.append(jobs)

    gpus = args.gpus
    buckets = [[] for _ in gpus]
    for i, group in enumerate(groups):
        buckets[i % len(gpus)].extend(group)

    with ThreadPoolExecutor(max_workers=len(gpus)) as ex:
        futures = [
            ex.submit(run_group, gpu, bucket, results_dir, args.force)
            for gpu, bucket in zip(gpus, buckets) if bucket
        ]
        for fut in as_completed(futures):
            fut.result()

    summary_cmd = [
        sys.executable, str(SUMMARY),
        "--raw-dir", str(results_dir / "raw"),
        "--out-dir", str(results_dir / "summary"),
    ]
    if args.verify:
        summary_cmd.append("--verify")
    subprocess.run(summary_cmd, check=True)

    print("\nTable 1 reproduction complete.")
    print(f"Raw results: {results_dir / 'raw'}")
    print(f"Summary:     {results_dir / 'summary' / 'table1.csv'}")
    print(f"Paired CIs:  {results_dir / 'summary' / 'paired_ci.csv'}")


if __name__ == "__main__":
    main()
