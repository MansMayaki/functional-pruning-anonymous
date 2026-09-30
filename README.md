# Functional error after pruning ReLU networks — anonymous reproduction repository

This repository accompanies an anonymous conference submission on functional change after pruning feedforward ReLU networks. It contains the exact MLP training/pruning code used for the controlled experiments, machine-readable configurations, deterministic seed handling, result aggregation, paired confidence intervals, and figure scripts.

The primary experiment holds the **task-Taylor saliency rule fixed** and changes only the fine-tuning objective:

- `fixed`: task loss + fixed-gate matching;
- `output`: task loss + ordinary output matching.

Within each seed, both branches share the dense checkpoint, standard image split, calibration subset and ordering, pruning targets, and deterministic fine-tuning minibatch schedule.

## Quick start

Create the environment:

```bash
conda env create -f environment/environment.yml
conda activate functional-pruning
```

Reproduce the controlled Table 1 experiment on one GPU:

```bash
./reproduce_table1.sh 0
```

or on two GPUs:

```bash
./reproduce_table1.sh 0 1
```

That single command:

1. trains/reuses one dense checkpoint for every `(dataset, seed)` pair;
2. runs the two controlled branches (`Taylor/fixed`, `Taylor/output`) through progressive pruning to 90% sparsity;
3. evaluates all functional quantities only on the held-out test set;
4. writes seed-level and aggregate Table 1 CSV files;
5. computes paired seed-level 95% Student-t intervals;
6. regenerates the controlled sparsity figure and fine-tuning-dynamics figure.

Outputs are written to:

```text
results/table1/raw/             per-seed JSON files
results/table1/dense_cache/     dense checkpoints
results/table1/summary/         Table 1 + paired CI CSV/LaTeX
figures/figure1_controlled.*    controlled sparsity trajectory
figures/figure2_finetuning_dynamics.*
```

## Expected Table 1 values

The paper reports the following five-seed aggregates at 90% sparsity:

| Dataset | Fine-tuning | Accuracy (%) | D_fix | D_func | kappa |
|---|---|---:|---:|---:|---:|
| MNIST | Output | 98.07 ± 0.08 | 7.44 ± 0.74 | 3.95 ± 0.13 | -0.840 ± 0.028 |
| MNIST | Fixed-gate | 98.13 ± 0.06 | 3.46 ± 0.08 | 12.03 ± 0.92 | -0.029 ± 0.010 |
| Fashion-MNIST | Output | 88.95 ± 0.36 | 14.62 ± 0.60 | 4.97 ± 0.08 | -0.936 ± 0.006 |
| Fashion-MNIST | Fixed-gate | 88.81 ± 0.28 | 4.72 ± 0.14 | 23.00 ± 1.09 | -0.020 ± 0.006 |

`expected/table1_reference.csv` contains these values in machine-readable form. The reproduction script prints deviations from this reference rather than silently enforcing exact equality across hardware/software stacks.

## Exact controlled protocol

`configs/table1.json` is the canonical configuration. The main settings are:

- datasets: MNIST and Fashion-MNIST;
- seeds: `0,1,2,3,4`;
- network: two hidden ReLU layers, width 512;
- dense training: Adam, 20 epochs, learning rate `1e-3`, batch 256, no weight decay, no scheduler;
- pruning: global unstructured pruning over weights, biases dense;
- saliency: task-Taylor estimated from 8 deterministic calibration minibatches;
- calibration: up to 4096 training examples, deterministic within seed;
- pruning targets: 50%, 70%, 80%, 90%;
- fine-tuning: 3 epochs after each pruning stage;
- matching coefficient: `lambda=0.01`;
- Adam is reinitialized at each pruning stage and retained across the three fine-tuning epochs of that stage;
- masked weights remain zero and cannot regrow;
- functional diagnostics are evaluated only on the held-out test set.

MNIST normalization is `(0.1307, 0.3081)` and Fashion-MNIST normalization is `(0.2860, 0.3530)`.

## Pairing and randomness

Each seed controls model initialization, dense-training shuffle, calibration selection/order, and fine-tuning minibatch order. For a fixed seed, the two Table 1 branches share all of these quantities. This makes the reported fixed-vs-output differences paired within seed.

The summary script reports paired 95% Student-t intervals for:

```text
Delta = fixed-gate FT - output FT
```

on accuracy, `D_fix`, `D_func`, and `kappa`.

## Data isolation

No test example is used for training, pruning, calibration, score estimation, or hyperparameter selection. The test split is used only for final functional diagnostics and predictive metrics. MNIST and Fashion-MNIST use the standard torchvision train/test splits.

## Repository layout

```text
configs/
  table1.json                 canonical controlled experiment
  full_protocol.json          broader diagnostic protocol
src/
  mlp_experiments.py          MLP model, pruning, matching, diagnostics
  theory_diagnostics.py       telescoping/bound/margin checks
  cnn_extension.py            optional CNN diagnostic extension
scripts/
  reproduce_table1.py         exact Table 1 launcher
  summarize_table1.py         means, SDs, paired CIs, LaTeX
  make_figure1.py             controlled sparsity figure
  make_figure2.py             90% fine-tuning dynamics
  make_calibration_occupancy.py
  launch_experiments.py       broader multi-GPU launcher
  analyze_results.py          broader aggregation
  capture_environment.sh      archival environment snapshot
expected/
  table1_reference.csv
  paired_ci_reference.csv
environment/
  environment.yml
  README.md
```

## Additional diagnostics

The core MLP code also implements:

- the full 2x3 saliency/fine-tuning factorial experiment;
- reverse decomposition under the sparse-gate convention;
- centered-logit, KL, Jensen-Shannon, and dense/sparse top-1 agreement metrics;
- calibration-pattern occupancy;
- single-weight versus batch-deletion diagnostics;
- diagonal GGN comparison.

For the calibration occupancy diagnostic:

```bash
python scripts/launch_experiments.py \
  --gpu-ids 0 1 \
  --seeds 0 1 2 3 4 \
  --phases calibration \
  --calibration-sizes 128 512 2048 4096 \
  --root results

python scripts/analyze_results.py --root results --outdir analysis
python scripts/make_calibration_occupancy.py \
  --raw-dir results/calibration \
  --out-prefix figures/calibration_occupancy
```