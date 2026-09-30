#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   ./reproduce_table1.sh              # GPU 0
#   ./reproduce_table1.sh 0 1          # GPUs 0 and 1
#   DEVICE=cpu ./reproduce_table1.sh cpu

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"
DEVICE="${DEVICE:-cuda}"
if [ "$#" -eq 0 ]; then
  GPUS=(0)
else
  GPUS=("$@")
fi
python scripts/reproduce_table1.py --device "$DEVICE" --gpus "${GPUS[@]}" --verify
python scripts/make_figure1.py --raw-dir results/table1/raw --out-prefix figures/figure1_controlled
python scripts/make_figure2.py --raw-dir results/table1/raw --out-prefix figures/figure2_finetuning_dynamics
