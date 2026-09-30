# Reproducibility protocol

## Controlled Table 1

The scientific comparison is deliberately narrow: task-Taylor saliency is held fixed and only the fine-tuning objective changes. The two branches are therefore paired within seed.

For each dataset and seed:

1. initialize and train one dense width-512 x 2 ReLU MLP for 20 epochs;
2. construct one deterministic calibration ordering from the training set;
3. cache that dense checkpoint;
4. start each branch from the identical dense checkpoint;
5. prune progressively to 50%, 70%, 80%, and 90% global weight sparsity;
6. before each stage, recompute task-Taylor scores from the current sparse parameters using the same calibration subset/order;
7. fine-tune surviving weights for three epochs using either fixed-gate or output matching;
8. use the same deterministic minibatch order for both objectives within a seed;
9. evaluate functional metrics only on the held-out test set.

The primary output metrics are mean test-example Euclidean norms `D_fix`, `D_gate`, and `D_func`, plus the seed-level cancellation index

```text
kappa = 2 C / (E_fix + E_gate).
```

## Determinism

The code seeds Python, NumPy, PyTorch CPU, and all CUDA devices. cuDNN benchmarking is disabled and deterministic mode is enabled when CUDA is available. Dense and fine-tuning DataLoader shuffles use explicit per-epoch generators.

Exact bitwise equality across GPU architectures or library versions is not promised. The repository therefore stores paper reference aggregates and reports deviations after a rerun.

## Optimizer state

Dense training uses one Adam optimizer across all 20 dense epochs. During iterative pruning, Adam is reinitialized at the beginning of every pruning stage, then the same optimizer state is retained across the three fine-tuning epochs of that stage.

## Calibration construction

Calibration indices are generated once per seed using NumPy `default_rng(100000 + seed)` and truncated to the requested calibration size. All pruning criteria use the same subset and ordering within that seed.

## Test isolation

The held-out test set is not used for model fitting, saliency estimation, pruning decisions, calibration, or hyperparameter selection. Functional diagnostics are computed only after each pruning/fine-tuning state has been determined.

## Reproducing paper artifacts

```bash
./reproduce_table1.sh 0 1
```

The script produces:

- `results/table1/summary/table1.csv`
- `results/table1/summary/table1.tex`
- `results/table1/summary/paired_ci.csv`
- `figures/figure1_controlled.pdf`
- `figures/figure2_finetuning_dynamics.pdf`

The corresponding paper reference values are stored under `expected/`.
