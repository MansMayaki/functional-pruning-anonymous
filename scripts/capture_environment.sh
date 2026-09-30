#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
mkdir -p "$ROOT/environment"
python -V > "$ROOT/environment/runtime.txt" 2>&1
python - <<'PY' >> "$ROOT/environment/runtime.txt"
import platform
import torch
print("platform:", platform.platform())
print("torch:", torch.__version__)
print("cuda_runtime:", torch.version.cuda)
print("cudnn:", torch.backends.cudnn.version())
print("cuda_available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu_count:", torch.cuda.device_count())
    for i in range(torch.cuda.device_count()):
        print(f"gpu_{i}:", torch.cuda.get_device_name(i))
PY
python -m pip freeze > "$ROOT/environment/pip-freeze.txt"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi > "$ROOT/environment/nvidia-smi.txt"
fi
printf 'Wrote environment provenance under %s/environment\n' "$ROOT"
