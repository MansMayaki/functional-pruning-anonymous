#!/usr/bin/env python3
"""GPU launcher for the functional-pruning experiments.

The launcher keeps all jobs for a given (dataset, seed) on the same worker so
that the first job creates the dense checkpoint and later jobs reuse it without
race conditions.

Recommended diagnostic run:
  python scripts/launch_experiments.py \
    --gpu-ids 0 1 2 3 --seeds 0 1 2 \
    --phases factorial calibration batch theory

Optional breadth / positioning:
  python scripts/launch_experiments.py \
    --gpu-ids 0 1 2 3 --seeds 0 1 2 --phases cnn second_order

For five-seed key comparisons, rerun with --seeds 0 1 2 3 4 and optionally
--factorial-key-only to limit the cost.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Tuple

HERE = Path(__file__).resolve().parent
SRC = HERE.parent / "src"
CORE = SRC / "mlp_experiments.py"
THEORY = SRC / "theory_diagnostics.py"
CNN = SRC / "cnn_extension.py"


def run_group(gpu: str, commands: List[Tuple[str, List[str]]], log_dir: Path):
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    out = []
    for name, cmd in commands:
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / f"{name}.log"
        print(f"[GPU {gpu}] START {name}")
        with log_path.open("w") as f:
            p = subprocess.run(cmd, env=env, stdout=f, stderr=subprocess.STDOUT)
        if p.returncode != 0:
            print(f"[GPU {gpu}] FAIL  {name} -> {log_path}")
            raise RuntimeError(f"Job failed: {name}")
        print(f"[GPU {gpu}] DONE  {name}")
        out.append((name, str(log_path)))
    return out


def distribute_groups(groups, gpu_ids, log_dir):
    buckets = [[] for _ in gpu_ids]
    for i, group in enumerate(groups):
        buckets[i % len(gpu_ids)].extend(group)
    with ThreadPoolExecutor(max_workers=len(gpu_ids)) as ex:
        futs = [
            ex.submit(run_group, str(gpu), bucket, log_dir)
            for gpu, bucket in zip(gpu_ids, buckets)
            if bucket
        ]
        for fut in as_completed(futs):
            fut.result()


def common_core(args, dataset, seed):
    return [
        sys.executable, str(CORE),
        "--dataset", dataset,
        "--seed", str(seed),
        "--data-dir", args.data_dir,
        "--dense-cache-dir", str(Path(args.root) / "dense_cache"),
        "--dense-epochs", str(args.dense_epochs),
        "--ft-epochs", str(args.ft_epochs),
        "--batch-size", str(args.batch_size),
        "--eval-batch-size", str(args.eval_batch_size),
        "--num-workers", str(args.num_workers),
        "--lambda-reg", str(args.lambda_reg),
        "--calib-size", str(args.calib_size),
        "--saliency-batches", str(args.saliency_batches),
    ]


def factorial_groups(args, second_order=False):
    groups = []
    for dataset in args.datasets:
        for seed in args.seeds:
            jobs = []
            if args.factorial_key_only:
                cells = [
                    ("taylor", "fixed"),
                    ("taylor", "output"),
                    ("fixed_delete", "fixed"),
                    ("fixed_delete", "output"),
                ]
            else:
                cells = [
                    (sal, ft)
                    for sal in ["taylor", "fixed_delete"]
                    for ft in ["task", "fixed", "output"]
                ]
            if second_order:
                cells = [("diag_ggn", "task")]
            for sal, ft in cells:
                mode_root = "second_order" if second_order else "factorial"
                name = f"{mode_root}_{dataset}_seed{seed}_{sal}_{ft}"
                cmd = [sys.executable, str(CORE), "factorial"] + common_core(args, dataset, seed)[2:]
                cmd += ["--saliency", sal, "--finetune", ft, "--output-dir", str(Path(args.root) / mode_root)]
                jobs.append((name, cmd))
            groups.append(jobs)
    return groups


def calibration_groups(args):
    groups = []
    for dataset in args.calibration_datasets:
        for seed in args.seeds:
            jobs = []
            for ncal in args.calibration_sizes:
                name = f"calibration_{dataset}_seed{seed}_n{ncal}"
                cmd = [sys.executable, str(CORE), "calibration"] + common_core(args, dataset, seed)[2:]
                # Replace common calibration size.
                i = cmd.index("--calib-size")
                cmd[i + 1] = str(ncal)
                cmd += ["--output-dir", str(Path(args.root) / "calibration")]
                jobs.append((name, cmd))
            groups.append(jobs)
    return groups


def batch_groups(args):
    groups = []
    for dataset in args.batch_datasets:
        for seed in args.seeds:
            name = f"batch_{dataset}_seed{seed}"
            cmd = [sys.executable, str(CORE), "batch"] + common_core(args, dataset, seed)[2:]
            cmd += [
                "--batch-fractions", *map(str, args.batch_fractions),
                "--batch-target", str(args.batch_target),
                "--output-dir", str(Path(args.root) / "batch"),
            ]
            groups.append([(name, cmd)])
    return groups


def theory_groups(args):
    groups = []
    cells = [
        ("taylor", "fixed"),
        ("taylor", "output"),
        ("fixed_delete", "fixed"),
        ("fixed_delete", "output"),
    ]
    for dataset in args.theory_datasets:
        for seed in args.seeds:
            jobs = []
            for sal, ft in cells:
                ck = Path(args.root) / "factorial" / f"{dataset}_seed{seed}_{sal}_{ft}.pt"
                out = Path(args.root) / "theory" / f"{dataset}_seed{seed}_{sal}_{ft}_theory.json"
                name = f"theory_{dataset}_seed{seed}_{sal}_{ft}"
                cmd = [
                    sys.executable, str(THEORY),
                    "--checkpoint", str(ck),
                    "--data-dir", args.data_dir,
                    "--max-samples", str(args.theory_samples),
                    "--batch-size", str(min(args.eval_batch_size, 128)),
                    "--num-workers", str(args.num_workers),
                    "--output", str(out),
                ]
                jobs.append((name, cmd))
            groups.append(jobs)
    return groups


def cnn_groups(args):
    groups = []
    for seed in args.seeds:
        jobs = []
        for ft in ["task", "fixed", "output"]:
            name = f"cnn_cifar10_seed{seed}_{ft}"
            cmd = [
                sys.executable, str(CNN),
                "--seed", str(seed),
                "--finetune", ft,
                "--data-dir", args.data_dir,
                "--dense-cache-dir", str(Path(args.root) / "dense_cache"),
                "--output-dir", str(Path(args.root) / "cnn"),
                "--dense-epochs", str(args.cnn_epochs),
                "--ft-epochs", str(args.ft_epochs),
                "--batch-size", str(args.cnn_batch_size),
                "--eval-batch-size", str(args.eval_batch_size),
                "--num-workers", str(args.num_workers),
                "--lambda-reg", str(args.lambda_reg),
                "--saliency-batches", str(args.saliency_batches),
            ]
            jobs.append((name, cmd))
        groups.append(jobs)
    return groups


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--gpu-ids", nargs="+", default=["0"])
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    p.add_argument("--datasets", nargs="+", default=["mnist", "fmnist"])
    p.add_argument("--calibration-datasets", nargs="+", default=["mnist"])
    p.add_argument("--batch-datasets", nargs="+", default=["mnist"])
    p.add_argument("--theory-datasets", nargs="+", default=["mnist", "fmnist"])
    p.add_argument("--phases", nargs="+", default=["factorial", "calibration", "batch", "theory"])
    p.add_argument("--root", default="results")
    p.add_argument("--data-dir", default="./data")
    p.add_argument("--dense-epochs", type=int, default=20)
    p.add_argument("--cnn-epochs", type=int, default=40)
    p.add_argument("--ft-epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--cnn-batch-size", type=int, default=128)
    p.add_argument("--eval-batch-size", type=int, default=256)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--lambda-reg", type=float, default=0.01)
    p.add_argument("--calib-size", type=int, default=4096)
    p.add_argument("--saliency-batches", type=int, default=8)
    p.add_argument("--calibration-sizes", nargs="+", type=int, default=[128, 512, 2048, 4096])
    p.add_argument("--batch-fractions", nargs="+", type=float, default=[0.01, 0.05, 0.10, 0.50])
    p.add_argument("--batch-target", type=float, default=0.50)
    p.add_argument("--theory-samples", type=int, default=1000)
    p.add_argument("--factorial-key-only", action="store_true")
    args = p.parse_args()

    log_dir = Path(args.root) / "logs"
    phases = set(args.phases)

    # Factorial must precede theory because theory consumes its checkpoints.
    if "factorial" in phases:
        distribute_groups(factorial_groups(args), args.gpu_ids, log_dir)
    if "second_order" in phases:
        distribute_groups(factorial_groups(args, second_order=True), args.gpu_ids, log_dir)
    if "calibration" in phases:
        distribute_groups(calibration_groups(args), args.gpu_ids, log_dir)
    if "batch" in phases:
        distribute_groups(batch_groups(args), args.gpu_ids, log_dir)
    if "theory" in phases:
        distribute_groups(theory_groups(args), args.gpu_ids, log_dir)
    if "cnn" in phases:
        distribute_groups(cnn_groups(args), args.gpu_ids, log_dir)

    print("All requested phases completed.")


if __name__ == "__main__":
    main()
