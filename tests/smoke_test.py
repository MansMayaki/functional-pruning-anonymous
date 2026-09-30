#!/usr/bin/env python3
"""Fast CPU smoke test using the synthetic dataset."""
from __future__ import annotations
import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CORE = ROOT / "src" / "mlp_experiments.py"


def run(ft, out):
    cmd = [
        sys.executable, str(CORE), "factorial",
        "--dataset", "synthetic",
        "--seed", "0",
        "--hidden", "16", "16",
        "--dense-epochs", "1",
        "--ft-epochs", "1",
        "--batch-size", "128",
        "--eval-batch-size", "128",
        "--num-workers", "0",
        "--calib-size", "128",
        "--saliency-batches", "1",
        "--sparsities", "0.5",
        "--saliency", "taylor",
        "--finetune", ft,
        "--device", "cpu",
        "--dense-cache-dir", str(out / "dense"),
        "--output-dir", str(out / "raw"),
    ]
    subprocess.run(cmd, check=True)


def main():
    with tempfile.TemporaryDirectory() as td:
        out = Path(td)
        run("fixed", out)
        run("output", out)
        for ft in ["fixed", "output"]:
            p = out / "raw" / f"synthetic_seed0_taylor_{ft}.json"
            assert p.exists(), p
            obj = json.loads(p.read_text())
            final = obj["final"]
            for k in ["accuracy", "D_fix", "D_gate", "D_func", "kappa", "twoC"]:
                assert k in final, (ft, k)
            residual = abs(final["E_fix"] + final["E_gate"] + final["twoC"] - final["E_func"])
            assert residual < 1e-4, residual
    print("smoke test: PASS")


if __name__ == "__main__":
    main()
