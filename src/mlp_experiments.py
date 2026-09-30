#!/usr/bin/env python3
"""
Experiments for dense-reference functional decomposition after pruning.

This script implements the main and diagnostic experiments on fully-connected
ReLU networks:

  1) reverse / sparse-gate decomposition robustness;
  2) 2 x 3 saliency x fine-tuning factorial experiment;
  3) exact-single-weight versus batch-deletion diagnostic;
  4) calibration-size / activation-pattern occupancy experiment;
  5) secondary fidelity metrics (centered logits, KL, JS, top-1 agreement);
  6) optional diagonal Gauss-Newton (OBD-style) pruning comparator.

The script deliberately reuses one dense checkpoint, split, calibration order,
and minibatch order per seed. Results are saved as JSON plus optional PyTorch
checkpoints so that theory_diagnostics.py can evaluate the layerwise
bound and margin predictions without retraining.

Examples
--------
# Factorial cell, MNIST seed 0:
python src/mlp_experiments.py factorial \
  --dataset mnist --seed 0 --saliency taylor --finetune output \
  --output-dir results/factorial

# Exact fixed-deletion saliency + fixed-gate FT:
python src/mlp_experiments.py factorial \
  --dataset mnist --seed 0 --saliency fixed_delete --finetune fixed \
  --output-dir results/factorial

# Calibration-size experiment:
python src/mlp_experiments.py calibration \
  --dataset mnist --seed 0 --calib-size 128 \
  --output-dir results/calibration

# Batch deletion / recomputation diagnostic:
python src/mlp_experiments.py batch \
  --dataset mnist --seed 0 --calib-size 1024 \
  --batch-fractions 0.01 0.05 0.10 0.50 \
  --batch-target 0.50 --output-dir results/batch

# Diagonal GGN baseline:
python src/mlp_experiments.py factorial \
  --dataset mnist --seed 0 --saliency diag_ggn --finetune task \
  --output-dir results/second_order
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import random
import time
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset, TensorDataset

try:
    from torchvision import datasets, transforms
except Exception:
    datasets = None
    transforms = None

try:
    from sklearn.metrics import roc_auc_score
except Exception:
    roc_auc_score = None


# -----------------------------------------------------------------------------
# Reproducibility
# -----------------------------------------------------------------------------


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def make_generator(seed: int) -> torch.Generator:
    g = torch.Generator()
    g.manual_seed(int(seed))
    return g


def json_dump(obj: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w") as f:
        json.dump(obj, f, indent=2, sort_keys=True)
    tmp.replace(path)


# -----------------------------------------------------------------------------
# Model
# -----------------------------------------------------------------------------


class ReLUMLP(nn.Module):
    def __init__(self, in_dim: int, hidden: Sequence[int], out_dim: int):
        super().__init__()
        dims = [in_dim, *hidden, out_dim]
        self.in_dim = int(in_dim)
        self.hidden = list(map(int, hidden))
        self.out_dim = int(out_dim)
        self.layers = nn.ModuleList(
            [nn.Linear(dims[i], dims[i + 1]) for i in range(len(dims) - 1)]
        )

    @property
    def hidden_layers(self) -> List[nn.Linear]:
        return list(self.layers[:-1])

    @property
    def output_layer(self) -> nn.Linear:
        return self.layers[-1]

    def flatten(self, x: torch.Tensor) -> torch.Tensor:
        return x.view(x.shape[0], -1) if x.dim() > 2 else x

    def forward(
        self,
        x: torch.Tensor,
        return_gates: bool = False,
        return_preacts: bool = False,
        return_acts: bool = False,
    ):
        h = self.flatten(x)
        gates, preacts, acts = [], [], [h]
        for layer in self.layers[:-1]:
            z = layer(h)
            g = z > 0
            h = z * g.to(z.dtype)
            preacts.append(z)
            gates.append(g)
            acts.append(h)
        out = self.layers[-1](h)
        items = [out]
        if return_gates:
            items.append(gates)
        if return_preacts:
            items.append(preacts)
        if return_acts:
            items.append(acts)
        return items[0] if len(items) == 1 else tuple(items)

    def forward_with_gates(
        self,
        x: torch.Tensor,
        gates: Sequence[torch.Tensor],
        return_preacts: bool = False,
        return_acts: bool = False,
    ):
        h = self.flatten(x)
        preacts, acts = [], [h]
        for layer, gate in zip(self.layers[:-1], gates):
            z = layer(h)
            h = z * gate.to(z.dtype)
            preacts.append(z)
            acts.append(h)
        out = self.layers[-1](h)
        items = [out]
        if return_preacts:
            items.append(preacts)
        if return_acts:
            items.append(acts)
        return items[0] if len(items) == 1 else tuple(items)


# -----------------------------------------------------------------------------
# Datasets
# -----------------------------------------------------------------------------


def require_torchvision():
    if datasets is None or transforms is None:
        raise RuntimeError("torchvision is required for MNIST/Fashion-MNIST")


def get_dataset(name: str, data_dir: str, seed: int):
    name = name.lower()
    if name == "synthetic":
        g = torch.Generator().manual_seed(seed)
        xtr = torch.randn(1024, 32, generator=g)
        w = torch.randn(32, 5, generator=g)
        ytr = (xtr @ w).argmax(1)
        xte = torch.randn(512, 32, generator=g)
        yte = (xte @ w).argmax(1)
        return TensorDataset(xtr, ytr), TensorDataset(xte, yte), 32, 5

    require_torchvision()
    root = str(Path(data_dir))
    if name == "mnist":
        tfm = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        )
        tr = datasets.MNIST(root, train=True, download=True, transform=tfm)
        te = datasets.MNIST(root, train=False, download=True, transform=tfm)
        return tr, te, 784, 10
    if name in {"fmnist", "fashion-mnist", "fashion_mnist"}:
        tfm = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.2860,), (0.3530,))]
        )
        tr = datasets.FashionMNIST(root, train=True, download=True, transform=tfm)
        te = datasets.FashionMNIST(root, train=False, download=True, transform=tfm)
        return tr, te, 784, 10
    raise ValueError(f"Unsupported dataset: {name}")


def fixed_calibration_indices(n: int, max_size: int, seed: int) -> List[int]:
    rng = np.random.default_rng(100_000 + int(seed))
    order = rng.permutation(n)
    return order[: min(max_size, n)].tolist()


def make_loader(
    ds: Dataset,
    batch_size: int,
    shuffle: bool,
    seed: int,
    num_workers: int,
) -> DataLoader:
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        generator=make_generator(seed) if shuffle else None,
    )


# -----------------------------------------------------------------------------
# Masks and pruning utilities
# -----------------------------------------------------------------------------


def make_masks(model: ReLUMLP, device: torch.device) -> List[torch.Tensor]:
    return [torch.ones_like(layer.weight, device=device) for layer in model.layers]


def apply_masks_(model: ReLUMLP, masks: Sequence[torch.Tensor]) -> None:
    with torch.no_grad():
        for layer, mask in zip(model.layers, masks):
            layer.weight.mul_(mask)


def zero_masked_grads_(model: ReLUMLP, masks: Sequence[torch.Tensor]) -> None:
    for layer, mask in zip(model.layers, masks):
        if layer.weight.grad is not None:
            layer.weight.grad.mul_(mask)


def sparsity(masks: Sequence[torch.Tensor]) -> float:
    total = sum(m.numel() for m in masks)
    active = sum(int(m.sum().item()) for m in masks)
    return 1.0 - active / total


def active_count(masks: Sequence[torch.Tensor]) -> int:
    return sum(int(m.sum().item()) for m in masks)


def total_count(masks: Sequence[torch.Tensor]) -> int:
    return sum(m.numel() for m in masks)


def prune_to_target_(
    masks: Sequence[torch.Tensor],
    scores: Sequence[torch.Tensor],
    target_sparsity: float,
) -> int:
    total = total_count(masks)
    target_active = int(round(total * (1.0 - target_sparsity)))
    current_active = active_count(masks)
    n_prune = max(0, current_active - target_active)
    if n_prune == 0:
        return 0

    vals, refs = [], []
    for li, (m, s) in enumerate(zip(masks, scores)):
        idx = torch.nonzero(m.view(-1) > 0, as_tuple=False).squeeze(1)
        if idx.numel() == 0:
            continue
        vals.append(s.detach().view(-1)[idx].cpu())
        refs.extend((li, int(j)) for j in idx.cpu().tolist())
    all_vals = torch.cat(vals)
    chosen = torch.topk(all_vals, k=n_prune, largest=False).indices.tolist()
    for ci in chosen:
        li, flat_idx = refs[ci]
        masks[li].view(-1)[flat_idx] = 0.0
    return n_prune


def prune_n_lowest_(
    masks: Sequence[torch.Tensor], scores: Sequence[torch.Tensor], n_prune: int
) -> int:
    current_active = active_count(masks)
    n_prune = min(int(n_prune), current_active)
    if n_prune <= 0:
        return 0
    vals, refs = [], []
    for li, (m, s) in enumerate(zip(masks, scores)):
        idx = torch.nonzero(m.view(-1) > 0, as_tuple=False).squeeze(1)
        if idx.numel() == 0:
            continue
        vals.append(s.detach().view(-1)[idx].cpu())
        refs.extend((li, int(j)) for j in idx.cpu().tolist())
    all_vals = torch.cat(vals)
    chosen = torch.topk(all_vals, k=n_prune, largest=False).indices.tolist()
    for ci in chosen:
        li, flat_idx = refs[ci]
        masks[li].view(-1)[flat_idx] = 0.0
    return n_prune


# -----------------------------------------------------------------------------
# Training
# -----------------------------------------------------------------------------


def accuracy(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    ok = n = 0
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb).argmax(1)
            ok += int((pred == yb).sum().item())
            n += yb.numel()
    return ok / max(n, 1)


def train_dense(
    model: ReLUMLP,
    train_ds: Dataset,
    test_ds: Dataset,
    device: torch.device,
    seed: int,
    epochs: int,
    batch_size: int,
    lr: float,
    num_workers: int,
) -> dict:
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    history = []
    for epoch in range(epochs):
        model.train()
        loader = make_loader(
            train_ds, batch_size, True, 10_000 + seed * 1000 + epoch, num_workers
        )
        total_loss = total_n = 0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = F.cross_entropy(logits, yb)
            loss.backward()
            opt.step()
            total_loss += float(loss.item()) * yb.numel()
            total_n += yb.numel()
        if epoch == epochs - 1 or epoch in {0, 4, 9}:
            te = make_loader(test_ds, batch_size, False, 0, num_workers)
            history.append(
                {
                    "epoch": epoch + 1,
                    "train_loss": total_loss / total_n,
                    "test_accuracy": accuracy(model, te, device),
                }
            )
    return {"history": history}


def load_or_train_dense(
    dataset: str,
    seed: int,
    hidden: Sequence[int],
    args,
    device: torch.device,
):
    train_ds, test_ds, in_dim, out_dim = get_dataset(dataset, args.data_dir, seed)
    model = ReLUMLP(in_dim, hidden, out_dim).to(device)
    cache_dir = Path(args.dense_cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    htag = "x".join(map(str, hidden))
    ckpt = cache_dir / f"{dataset}_h{htag}_seed{seed}.pt"
    if ckpt.exists() and not args.retrain_dense:
        obj = torch.load(ckpt, map_location=device)
        model.load_state_dict(obj["state_dict"])
        meta = obj.get("meta", {})
    else:
        seed_all(seed)
        meta = train_dense(
            model,
            train_ds,
            test_ds,
            device,
            seed,
            args.dense_epochs,
            args.batch_size,
            args.lr,
            args.num_workers,
        )
        te = make_loader(test_ds, args.eval_batch_size, False, 0, args.num_workers)
        meta["dense_accuracy"] = accuracy(model, te, device)
        torch.save(
            {
                "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                "dataset": dataset,
                "seed": seed,
                "hidden": list(hidden),
                "in_dim": in_dim,
                "out_dim": out_dim,
                "meta": meta,
            },
            ckpt,
        )
    te = make_loader(test_ds, args.eval_batch_size, False, 0, args.num_workers)
    dense_acc = accuracy(model, te, device)
    return model, train_ds, test_ds, in_dim, out_dim, dense_acc, str(ckpt)


# -----------------------------------------------------------------------------
# Losses, saliency, exact Q, diagonal GGN
# -----------------------------------------------------------------------------


def fixed_gate_loss(
    model: ReLUMLP, dense_model: ReLUMLP, xb: torch.Tensor
) -> torch.Tensor:
    with torch.no_grad():
        dense_logits, dense_gates = dense_model(xb, return_gates=True)
    pred = model.forward_with_gates(xb, dense_gates)
    return (pred - dense_logits).pow(2).sum(dim=1).mean()


def output_match_loss(
    model: ReLUMLP, dense_model: ReLUMLP, xb: torch.Tensor
) -> torch.Tensor:
    with torch.no_grad():
        dense_logits = dense_model(xb)
    return (model(xb) - dense_logits).pow(2).sum(dim=1).mean()


def compute_task_grads(
    model: ReLUMLP,
    masks: Sequence[torch.Tensor],
    calib_ds: Dataset,
    device: torch.device,
    args,
) -> List[torch.Tensor]:
    model.zero_grad(set_to_none=True)
    # Use the same deterministic calibration minibatches for every method.
    loader = make_loader(
        calib_ds, args.batch_size, False, 0, args.num_workers
    )
    n_batches = 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        loss = F.cross_entropy(model(xb), yb)
        (loss / args.saliency_batches).backward()
        n_batches += 1
        if n_batches >= args.saliency_batches:
            break
    grads = []
    for layer, mask in zip(model.layers, masks):
        g = layer.weight.grad.detach().clone() if layer.weight.grad is not None else torch.zeros_like(layer.weight)
        grads.append(g * mask)
    model.zero_grad(set_to_none=True)
    return grads


def compute_fixed_grads_and_Q(
    model: ReLUMLP,
    dense_model: ReLUMLP,
    masks: Sequence[torch.Tensor],
    calib_ds: Dataset,
    device: torch.device,
    args,
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    """Compute grad E_fix and exact Q_e for every scalar weight."""
    model.zero_grad(set_to_none=True)
    Q = [torch.zeros_like(layer.weight) for layer in model.layers]
    total_examples = 0
    loader = make_loader(calib_ds, args.batch_size, False, 0, args.num_workers)
    used_batches = 0

    for xb, _ in loader:
        xb = xb.to(device)
        with torch.no_grad():
            _, dense_gates = dense_model(xb, return_gates=True)
            _, acts = model.forward_with_gates(xb, dense_gates, return_acts=True)
        bsz = xb.shape[0]

        with torch.no_grad():
            # Hidden-layer output sensitivities d output / d z_l.
            W_out = model.output_layer.weight.detach()  # C x Hlast
            Bz: List[torch.Tensor] = [None] * len(model.hidden_layers)  # type: ignore
            if len(model.hidden_layers) > 0:
                cur = W_out.unsqueeze(0).expand(bsz, -1, -1)
                cur = cur * dense_gates[-1].to(cur.dtype).unsqueeze(1)
                Bz[-1] = cur
                for l in range(len(model.hidden_layers) - 2, -1, -1):
                    W_next = model.hidden_layers[l + 1].weight.detach()
                    cur = torch.einsum("bco,oi->bci", cur, W_next)
                    cur = cur * dense_gates[l].to(cur.dtype).unsqueeze(1)
                    Bz[l] = cur

                for l in range(len(model.hidden_layers)):
                    hprev2 = acts[l].detach().pow(2)  # B x in_l
                    sens2 = Bz[l].pow(2).sum(dim=1)  # B x out_l
                    Q[l] += torch.einsum("bo,bi->oi", sens2, hprev2)

            # Output layer: ||d f / d W[c,j]||^2 = h_j^2.
            hlast2 = acts[-1].detach().pow(2).sum(dim=0)  # Hlast
            Q[-1] += hlast2.unsqueeze(0).expand_as(Q[-1])

        total_examples += bsz
        used_batches += 1
        if used_batches >= args.saliency_batches:
            break

    # The gradient above is normalized by len(calib_ds), but only a prefix of
    # saliency batches may have been used. Recompute a consistent denominator.
    # Q is always averaged over the actually used examples.
    if total_examples == 0:
        raise RuntimeError("Empty calibration loader")
    Q = [(q / total_examples) * m for q, m in zip(Q, masks)]

    # Recompute fixed gradient with average over actually used examples for
    # consistency when saliency_batches truncates the loader.
    model.zero_grad(set_to_none=True)
    loader = make_loader(calib_ds, args.batch_size, False, 0, args.num_workers)
    seen = 0
    cached_batches = []
    for bi, (xb, _) in enumerate(loader):
        if bi >= args.saliency_batches:
            break
        cached_batches.append(xb)
        seen += xb.shape[0]
    for xb_cpu in cached_batches:
        xb = xb_cpu.to(device)
        with torch.no_grad():
            dense_logits, dense_gates = dense_model(xb, return_gates=True)
        out = model.forward_with_gates(xb, dense_gates)
        loss = ((out - dense_logits) ** 2).sum() / seen
        loss.backward()
    grads = []
    for layer, mask in zip(model.layers, masks):
        g = layer.weight.grad.detach().clone() if layer.weight.grad is not None else torch.zeros_like(layer.weight)
        grads.append(g * mask)
    model.zero_grad(set_to_none=True)
    return grads, Q


def compute_diag_ggn(
    model: ReLUMLP,
    masks: Sequence[torch.Tensor],
    calib_ds: Dataset,
    device: torch.device,
    args,
) -> List[torch.Tensor]:
    """Exact diagonal generalized Gauss-Newton for cross-entropy on an MLP.

    For a hidden weight w_ij: diag GGN = E[h_j^2 * b_i^T H_logit b_i],
    H_logit = diag(p) - p p^T. This is computed without per-parameter
    Jacobian materialization.
    """
    G = [torch.zeros_like(layer.weight) for layer in model.layers]
    loader = make_loader(calib_ds, args.batch_size, False, 0, args.num_workers)
    total = 0
    for bi, (xb, _) in enumerate(loader):
        if bi >= args.saliency_batches:
            break
        xb = xb.to(device)
        with torch.no_grad():
            logits, gates, acts = model(xb, return_gates=True, return_acts=True)
            p = logits.softmax(dim=1)  # B x C
            bsz = xb.shape[0]

            # Hidden Bz as above but under realized gates.
            if len(model.hidden_layers) > 0:
                cur = model.output_layer.weight.detach().unsqueeze(0).expand(bsz, -1, -1)
                cur = cur * gates[-1].to(cur.dtype).unsqueeze(1)
                Bz: List[torch.Tensor] = [None] * len(model.hidden_layers)  # type: ignore
                Bz[-1] = cur
                for l in range(len(model.hidden_layers) - 2, -1, -1):
                    W_next = model.hidden_layers[l + 1].weight.detach()
                    cur = torch.einsum("bco,oi->bci", cur, W_next)
                    cur = cur * gates[l].to(cur.dtype).unsqueeze(1)
                    Bz[l] = cur

                for l in range(len(model.hidden_layers)):
                    B = Bz[l]  # B x C x H_l
                    # b_i^T H b_i = sum_c p_c b_c^2 - (sum_c p_c b_c)^2
                    first = (p.unsqueeze(2) * B.pow(2)).sum(dim=1)
                    second = (p.unsqueeze(2) * B).sum(dim=1).pow(2)
                    q = (first - second).clamp_min(0.0)  # B x H_l
                    h2 = acts[l].pow(2)  # B x in_l
                    G[l] += torch.einsum("bo,bi->oi", q, h2)

            # Output layer exact diagonal: p_c(1-p_c) h_j^2.
            qout = p * (1.0 - p)  # B x C
            h2 = acts[-1].pow(2)  # B x H
            G[-1] += torch.einsum("bo,bi->oi", qout, h2)
            total += bsz
    if total == 0:
        raise RuntimeError("No examples for diagonal GGN")
    return [(g / total) * m for g, m in zip(G, masks)]


def compute_scores(
    model: ReLUMLP,
    dense_model: ReLUMLP,
    masks: Sequence[torch.Tensor],
    calib_ds: Dataset,
    saliency: str,
    device: torch.device,
    args,
):
    saliency = saliency.lower()
    if saliency == "magnitude":
        return [layer.weight.detach().abs() * mask for layer, mask in zip(model.layers, masks)], {}

    if saliency == "taylor":
        gt = compute_task_grads(model, masks, calib_ds, device, args)
        scores = [(-layer.weight.detach() * g).abs() * mask for layer, g, mask in zip(model.layers, gt, masks)]
        return scores, {}

    if saliency == "fixed_delete":
        gt = compute_task_grads(model, masks, calib_ds, device, args)
        gf, Q = compute_fixed_grads_and_Q(model, dense_model, masks, calib_ds, device, args)
        scores = []
        for layer, gtask, gfix, q, mask in zip(model.layers, gt, gf, Q, masks):
            w = layer.weight.detach()
            d_task = -w * gtask
            d_fix = -w * gfix + w.pow(2) * q
            scores.append((d_task + args.lambda_reg * d_fix).abs() * mask)
        return scores, {"fixed_grad": gf, "Q": Q}

    if saliency == "diag_ggn":
        ggn = compute_diag_ggn(model, masks, calib_ds, device, args)
        scores = [0.5 * layer.weight.detach().pow(2) * g * mask for layer, g, mask in zip(model.layers, ggn, masks)]
        return scores, {"diag_ggn": ggn}

    raise ValueError(f"Unknown saliency: {saliency}")


# -----------------------------------------------------------------------------
# Fine-tuning
# -----------------------------------------------------------------------------


def fine_tune_one_epoch(
    model: ReLUMLP,
    dense_model: ReLUMLP,
    masks: Sequence[torch.Tensor],
    train_ds: Dataset,
    objective: str,
    stage_id: int,
    epoch_id: int,
    device: torch.device,
    args,
    optimizer: Optional[torch.optim.Optimizer] = None,
):
    model.train()
    dense_model.eval()
    opt = optimizer if optimizer is not None else torch.optim.Adam(model.parameters(), lr=args.lr)
    loader = make_loader(
        train_ds,
        args.batch_size,
        True,
        200_000 + args.seed * 10_000 + stage_id * 100 + epoch_id,
        args.num_workers,
    )
    totals = {"task": 0.0, "match": 0.0, "n": 0}
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        opt.zero_grad(set_to_none=True)
        logits = model(xb)
        task = F.cross_entropy(logits, yb)
        match = torch.zeros((), device=device)
        if objective == "fixed":
            with torch.no_grad():
                dense_logits, dense_gates = dense_model(xb, return_gates=True)
            fixed_logits = model.forward_with_gates(xb, dense_gates)
            match = (fixed_logits - dense_logits).pow(2).sum(dim=1).mean()
        elif objective == "output":
            with torch.no_grad():
                dense_logits = dense_model(xb)
            match = (logits - dense_logits).pow(2).sum(dim=1).mean()
        elif objective != "task":
            raise ValueError(objective)
        loss = task + (args.lambda_reg * match if objective != "task" else 0.0)
        loss.backward()
        zero_masked_grads_(model, masks)
        opt.step()
        apply_masks_(model, masks)
        totals["task"] += float(task.item()) * yb.numel()
        totals["match"] += float(match.item()) * yb.numel()
        totals["n"] += yb.numel()
    return {
        "task_loss": totals["task"] / max(totals["n"], 1),
        "match_loss": totals["match"] / max(totals["n"], 1),
    }


# -----------------------------------------------------------------------------
# Metrics: primary and reverse decomposition + secondary fidelity
# -----------------------------------------------------------------------------


def _cosine(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-12):
    na = a.norm(dim=1)
    nb = b.norm(dim=1)
    good = (na > eps) & (nb > eps)
    out = torch.zeros_like(na)
    out[good] = (a[good] * b[good]).sum(dim=1) / (na[good] * nb[good])
    return out, good


def _kl_teacher_student(dense_logits: torch.Tensor, sparse_logits: torch.Tensor):
    p = dense_logits.softmax(1)
    logp = dense_logits.log_softmax(1)
    logq = sparse_logits.log_softmax(1)
    return (p * (logp - logq)).sum(1)


def _js(dense_logits: torch.Tensor, sparse_logits: torch.Tensor):
    p = dense_logits.softmax(1)
    q = sparse_logits.softmax(1)
    m = 0.5 * (p + q)
    klpm = (p * (p.clamp_min(1e-12).log() - m.clamp_min(1e-12).log())).sum(1)
    klqm = (q * (q.clamp_min(1e-12).log() - m.clamp_min(1e-12).log())).sum(1)
    return 0.5 * (klpm + klqm)


def evaluate_all_metrics(
    dense_model: ReLUMLP,
    sparse_model: ReLUMLP,
    test_ds: Dataset,
    device: torch.device,
    args,
) -> dict:
    dense_model.eval(); sparse_model.eval()
    loader = make_loader(test_ds, args.eval_batch_size, False, 0, args.num_workers)
    sums = Counter()
    n = 0
    primary_cos_sum = reverse_cos_sum = 0.0
    primary_cos_n = reverse_cos_n = 0

    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            dense_logits, dense_gates = dense_model(xb, return_gates=True)
            sparse_logits, sparse_gates = sparse_model(xb, return_gates=True)

            # Primary dense-reference convention.
            sparse_under_dense = sparse_model.forward_with_gates(xb, dense_gates)
            vfix = dense_logits - sparse_under_dense
            vgate = sparse_under_dense - sparse_logits
            vfunc = dense_logits - sparse_logits

            # Reverse convention: gate change under dense parameters first,
            # then parameter change under sparse gates.
            dense_under_sparse = dense_model.forward_with_gates(xb, sparse_gates)
            vgate_dense = dense_logits - dense_under_sparse
            vparam_sparse = dense_under_sparse - sparse_logits

            b = xb.shape[0]
            n += b
            for name, vec in [("fix", vfix), ("gate", vgate), ("func", vfunc)]:
                norms = vec.norm(dim=1)
                sums[f"D_{name}"] += float(norms.sum().item())
                sums[f"E_{name}"] += float(vec.pow(2).sum(dim=1).sum().item())
            cross = (vfix * vgate).sum(dim=1)
            sums["C"] += float(cross.sum().item())

            for name, vec in [("gate_dense", vgate_dense), ("param_sparse", vparam_sparse)]:
                norms = vec.norm(dim=1)
                sums[f"Drev_{name}"] += float(norms.sum().item())
                sums[f"Erev_{name}"] += float(vec.pow(2).sum(dim=1).sum().item())
            cross_r = (vgate_dense * vparam_sparse).sum(dim=1)
            sums["Crev"] += float(cross_r.sum().item())

            c, good = _cosine(vfix, vgate)
            primary_cos_sum += float(c[good].sum().item())
            primary_cos_n += int(good.sum().item())
            c2, good2 = _cosine(vgate_dense, vparam_sparse)
            reverse_cos_sum += float(c2[good2].sum().item())
            reverse_cos_n += int(good2.sum().item())

            # Secondary realized-function metrics.
            dc = dense_logits - dense_logits.mean(dim=1, keepdim=True)
            sc = sparse_logits - sparse_logits.mean(dim=1, keepdim=True)
            sums["centered_logit_l2"] += float((dc - sc).norm(dim=1).sum().item())
            sums["kl_dense_sparse"] += float(_kl_teacher_student(dense_logits, sparse_logits).sum().item())
            sums["js_dense_sparse"] += float(_js(dense_logits, sparse_logits).sum().item())
            sums["teacher_agreement"] += float((dense_logits.argmax(1) == sparse_logits.argmax(1)).sum().item())
            sums["accuracy"] += float((sparse_logits.argmax(1) == yb).sum().item())
            sums["dense_accuracy"] += float((dense_logits.argmax(1) == yb).sum().item())

            # Gate flip fraction over all hidden units.
            flips = total_gates = 0
            for gd, gs in zip(dense_gates, sparse_gates):
                flips += int((gd != gs).sum().item())
                total_gates += gd.numel()
            sums["gate_flips"] += flips
            sums["gate_total"] += total_gates

    out = {k: v / n for k, v in sums.items() if k not in {"gate_flips", "gate_total"}}
    out["accuracy"] = sums["accuracy"] / n
    out["dense_accuracy"] = sums["dense_accuracy"] / n
    out["teacher_agreement"] = sums["teacher_agreement"] / n
    out["gate_flip_rate"] = sums["gate_flips"] / max(sums["gate_total"], 1)
    out["mean_cosine"] = primary_cos_sum / max(primary_cos_n, 1)
    out["reverse_mean_cosine"] = reverse_cos_sum / max(reverse_cos_n, 1)
    out["twoC"] = 2.0 * out["C"]
    denom = out["E_fix"] + out["E_gate"]
    out["kappa"] = (2.0 * out["C"] / denom) if denom > 0 else 0.0
    out["decomp_residual"] = out["E_func"] - (out["E_fix"] + out["E_gate"] + 2.0 * out["C"])
    denom_r = out["Erev_gate_dense"] + out["Erev_param_sparse"]
    out["reverse_kappa"] = (2.0 * out["Crev"] / denom_r) if denom_r > 0 else 0.0
    out["reverse_E_func_from_parts"] = out["Erev_gate_dense"] + out["Erev_param_sparse"] + 2.0 * out["Crev"]
    out["reverse_decomp_residual"] = out["E_func"] - out["reverse_E_func_from_parts"]
    return out


def evaluate_fixed_mse(
    dense_model: ReLUMLP,
    model: ReLUMLP,
    ds: Dataset,
    device: torch.device,
    args,
) -> float:
    loader = make_loader(ds, args.eval_batch_size, False, 0, args.num_workers)
    total = n = 0
    dense_model.eval(); model.eval()
    with torch.no_grad():
        for xb, _ in loader:
            xb = xb.to(device)
            dlog, dg = dense_model(xb, return_gates=True)
            slog = model.forward_with_gates(xb, dg)
            total += float((dlog - slog).pow(2).sum(dim=1).sum().item())
            n += xb.shape[0]
    return total / max(n, 1)


# -----------------------------------------------------------------------------
# Fixed-gate input Jacobian discrepancy
# -----------------------------------------------------------------------------


def fixed_input_jacobian(model: ReLUMLP, gates: Sequence[torch.Tensor]) -> torch.Tensor:
    """Batched fixed-gate d logits / d input: B x C x input_dim."""
    B = model.output_layer.weight.unsqueeze(0).expand(gates[0].shape[0], -1, -1)
    for l in range(len(model.hidden_layers) - 1, -1, -1):
        B = B * gates[l].to(B.dtype).unsqueeze(1)
        W = model.hidden_layers[l].weight
        B = torch.einsum("bco,oi->bci", B, W)
    return B


def evaluate_jacobian_diff(
    dense_model: ReLUMLP,
    model: ReLUMLP,
    test_ds: Dataset,
    device: torch.device,
    args,
    max_samples: int = 1000,
) -> float:
    loader = make_loader(test_ds, min(args.eval_batch_size, 128), False, 0, args.num_workers)
    total = n = 0
    dense_model.eval(); model.eval()
    with torch.no_grad():
        for xb, _ in loader:
            if n >= max_samples:
                break
            xb = xb[: max_samples - n].to(device)
            _, dg = dense_model(xb, return_gates=True)
            Jd = fixed_input_jacobian(dense_model, dg)
            Js = fixed_input_jacobian(model, dg)
            total += float((Jd - Js).flatten(1).norm(dim=1).sum().item())
            n += xb.shape[0]
    return total / max(n, 1)


# -----------------------------------------------------------------------------
# Pattern occupancy
# -----------------------------------------------------------------------------


def pattern_occupancy(dense_model: ReLUMLP, calib_ds: Dataset, device, args) -> dict:
    loader = make_loader(calib_ds, args.eval_batch_size, False, 0, args.num_workers)
    counter = Counter()
    dense_model.eval()
    with torch.no_grad():
        for xb, _ in loader:
            xb = xb.to(device)
            _, gates = dense_model(xb, return_gates=True)
            packed = [np.packbits(g.cpu().numpy().astype(np.uint8), axis=1) for g in gates]
            for i in range(xb.shape[0]):
                key = b"".join(arr[i].tobytes() for arr in packed)
                counter[key] += 1
    occ = np.array(list(counter.values()), dtype=np.int64)
    n = len(calib_ds)
    return {
        "n_calib": n,
        "n_unique_patterns": int(len(counter)),
        "unique_fraction": float(len(counter) / max(n, 1)),
        "n_singletons": int((occ == 1).sum()),
        "singleton_pattern_fraction": float((occ == 1).mean()) if len(occ) else 0.0,
        "singleton_sample_fraction": float((occ[occ == 1].sum() / n)) if n else 0.0,
        "max_occupancy": int(occ.max()) if len(occ) else 0,
        "occupancy_mean": float(occ.mean()) if len(occ) else 0.0,
        "occupancy_median": float(np.median(occ)) if len(occ) else 0.0,
        "occupancy_p90": float(np.quantile(occ, 0.90)) if len(occ) else 0.0,
        "occupancy_p99": float(np.quantile(occ, 0.99)) if len(occ) else 0.0,
    }


# -----------------------------------------------------------------------------
# Iterative pruning
# -----------------------------------------------------------------------------


def run_iterative_pruning(
    dense_model: ReLUMLP,
    train_ds: Dataset,
    test_ds: Dataset,
    calib_ds: Dataset,
    saliency: str,
    finetune: str,
    device: torch.device,
    args,
):
    model = copy.deepcopy(dense_model).to(device)
    masks = make_masks(model, device)
    trajectory = []

    for stage_id, target in enumerate(args.sparsities):
        scores, _ = compute_scores(
            model, dense_model, masks, calib_ds, saliency, device, args
        )
        npr = prune_to_target_(masks, scores, float(target))
        apply_masks_(model, masks)

        # Reviewer asks for immediately-after-pruning and epoch-wise evolution.
        pre = evaluate_all_metrics(dense_model, model, test_ds, device, args)
        trajectory.append(
            {
                "stage": stage_id,
                "target_sparsity": float(target),
                "phase": "post_prune_pre_ft",
                "epoch": 0,
                "n_pruned_this_stage": npr,
                "actual_sparsity": sparsity(masks),
                **pre,
            }
        )

        ft_optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        for ep in range(1, args.ft_epochs + 1):
            losses = fine_tune_one_epoch(
                model,
                dense_model,
                masks,
                train_ds,
                finetune,
                stage_id,
                ep,
                device,
                args,
                optimizer=ft_optimizer,
            )
            met = evaluate_all_metrics(dense_model, model, test_ds, device, args)
            trajectory.append(
                {
                    "stage": stage_id,
                    "target_sparsity": float(target),
                    "phase": "finetune",
                    "epoch": ep,
                    "actual_sparsity": sparsity(masks),
                    **losses,
                    **met,
                }
            )

    final = trajectory[-1].copy()
    return model, masks, trajectory, final


# -----------------------------------------------------------------------------
# Batch-deletion diagnostic
# -----------------------------------------------------------------------------


def flatten_selected_cost_sum(
    model: ReLUMLP,
    masks_before: Sequence[torch.Tensor],
    masks_after: Sequence[torch.Tensor],
    gfix: Sequence[torch.Tensor],
    Q: Sequence[torch.Tensor],
):
    total = 0.0
    per_layer = []
    for layer, mb, ma, gf, q in zip(model.layers, masks_before, masks_after, gfix, Q):
        deleted = (mb > 0) & (ma == 0)
        w = layer.weight.detach()
        delta = -w * gf + w.pow(2) * q
        val = float(delta[deleted].sum().item())
        per_layer.append(val)
        total += val
    return total, per_layer


def run_batch_diagnostic(dense_model, train_ds, test_ds, calib_ds, device, args):
    results = {"one_shot": [], "recompute": []}

    # One-shot additivity from dense checkpoint.
    base_model = copy.deepcopy(dense_model).to(device)
    base_masks = make_masks(base_model, device)
    scores, aux = compute_scores(
        base_model, dense_model, base_masks, calib_ds, "fixed_delete", device, args
    )
    gf, Q = aux["fixed_grad"], aux["Q"]
    base_E = evaluate_fixed_mse(dense_model, base_model, calib_ds, device, args)
    total = total_count(base_masks)

    for frac in args.batch_fractions:
        model = copy.deepcopy(base_model)
        mb = [m.clone() for m in base_masks]
        ma = [m.clone() for m in base_masks]
        npr = max(1, int(round(total * float(frac))))
        prune_n_lowest_(ma, scores, npr)
        apply_masks_(model, ma)
        predicted, per_layer = flatten_selected_cost_sum(base_model, mb, ma, gf, Q)
        actual_E = evaluate_fixed_mse(dense_model, model, calib_ds, device, args)
        actual_delta = actual_E - base_E
        results["one_shot"].append(
            {
                "fraction_total_weights": float(frac),
                "n_deleted": npr,
                "predicted_sum_single_weight_delta": predicted,
                "actual_batch_delta": actual_delta,
                "actual_over_predicted": actual_delta / predicted if abs(predicted) > 1e-15 else None,
                "relative_prediction_error": abs(actual_delta - predicted) / max(abs(actual_delta), 1e-15),
                "predicted_per_layer": per_layer,
            }
        )

    # Recompute/rerank frequency until batch_target, no FT between chunks.
    for chunk_frac in args.batch_fractions:
        model = copy.deepcopy(dense_model).to(device)
        masks = make_masks(model, device)
        target_deleted = int(round(total * args.batch_target))
        chunk = max(1, int(round(total * float(chunk_frac))))
        deleted = 0
        trace = []
        while deleted < target_deleted:
            scores, _ = compute_scores(
                model, dense_model, masks, calib_ds, "fixed_delete", device, args
            )
            n_this = min(chunk, target_deleted - deleted)
            prune_n_lowest_(masks, scores, n_this)
            apply_masks_(model, masks)
            deleted += n_this
            trace.append(
                {
                    "deleted": deleted,
                    "sparsity": sparsity(masks),
                    "calib_Efix": evaluate_fixed_mse(dense_model, model, calib_ds, device, args),
                }
            )
        pre_metrics = evaluate_all_metrics(dense_model, model, test_ds, device, args)
        # Fine-tune with one persistent Adam state across epochs, matching the paper protocol.
        ft_optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        for ep in range(1, args.ft_epochs + 1):
            fine_tune_one_epoch(
                model, dense_model, masks, train_ds, "fixed", 99, ep, device, args,
                optimizer=ft_optimizer,
            )
        post_metrics = evaluate_all_metrics(dense_model, model, test_ds, device, args)
        results["recompute"].append(
            {
                "chunk_fraction_total_weights": float(chunk_frac),
                "target_sparsity": args.batch_target,
                "n_score_recomputations": len(trace),
                "trace": trace,
                "pre_ft_metrics": pre_metrics,
                "post_ft_metrics": post_metrics,
            }
        )
    return results


# -----------------------------------------------------------------------------
# CLI modes
# -----------------------------------------------------------------------------


def build_common_parser(sub):
    sub.add_argument("--dataset", default="mnist", choices=["mnist", "fmnist", "synthetic"])
    sub.add_argument("--seed", type=int, default=0)
    sub.add_argument("--hidden", type=int, nargs="+", default=[512, 512])
    sub.add_argument("--data-dir", default="./data")
    sub.add_argument("--dense-cache-dir", default="results/dense_cache")
    sub.add_argument("--output-dir", default="results")
    sub.add_argument("--retrain-dense", action="store_true")
    sub.add_argument("--dense-epochs", type=int, default=20)
    sub.add_argument("--ft-epochs", type=int, default=3)
    sub.add_argument("--batch-size", type=int, default=256)
    sub.add_argument("--eval-batch-size", type=int, default=256)
    sub.add_argument("--num-workers", type=int, default=0)
    sub.add_argument("--lr", type=float, default=1e-3)
    sub.add_argument("--lambda-reg", type=float, default=1e-2)
    sub.add_argument("--calib-size", type=int, default=4096)
    sub.add_argument("--saliency-batches", type=int, default=8)
    sub.add_argument("--sparsities", type=float, nargs="+", default=[0.50, 0.70, 0.80, 0.90])
    sub.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")


def prepare(args):
    seed_all(args.seed)
    device = torch.device(args.device)
    dense, train_ds, test_ds, in_dim, out_dim, dense_acc, dense_ckpt = load_or_train_dense(
        args.dataset, args.seed, args.hidden, args, device
    )
    max_cal = min(4096, len(train_ds))
    idx = fixed_calibration_indices(len(train_ds), max_cal, args.seed)
    calib = Subset(train_ds, idx[: min(args.calib_size, len(idx))])
    return device, dense, train_ds, test_ds, calib, in_dim, out_dim, dense_acc, dense_ckpt, idx


def mode_factorial(args):
    device, dense, train_ds, test_ds, calib, in_dim, out_dim, dense_acc, dense_ckpt, _ = prepare(args)
    start = time.time()
    model, masks, trajectory, final = run_iterative_pruning(
        dense, train_ds, test_ds, calib, args.saliency, args.finetune, device, args
    )
    final["fixed_input_jacobian_diff"] = evaluate_jacobian_diff(
        dense, model, test_ds, device, args
    )
    result = {
        "mode": "factorial",
        "dataset": args.dataset,
        "seed": args.seed,
        "hidden": args.hidden,
        "saliency": args.saliency,
        "finetune": args.finetune,
        "lambda_reg": args.lambda_reg,
        "calib_size": len(calib),
        "sparsities": args.sparsities,
        "dense_accuracy": dense_acc,
        "dense_checkpoint": dense_ckpt,
        "trajectory": trajectory,
        "final": final,
        "runtime_seconds": time.time() - start,
    }
    outdir = Path(args.output_dir)
    tag = f"{args.dataset}_seed{args.seed}_{args.saliency}_{args.finetune}"
    json_path = outdir / f"{tag}.json"
    pt_path = outdir / f"{tag}.pt"
    json_dump(result, json_path)
    torch.save(
        {
            "dataset": args.dataset,
            "seed": args.seed,
            "hidden": args.hidden,
            "in_dim": in_dim,
            "out_dim": out_dim,
            "saliency": args.saliency,
            "finetune": args.finetune,
            "lambda_reg": args.lambda_reg,
            "sparsities": args.sparsities,
            "dense_state_dict": {k: v.detach().cpu() for k, v in dense.state_dict().items()},
            "sparse_state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
            "masks": [m.detach().cpu() for m in masks],
        },
        pt_path,
    )
    print(json_path)
    print(pt_path)


def mode_calibration(args):
    device, dense, train_ds, test_ds, calib, in_dim, out_dim, dense_acc, dense_ckpt, idx = prepare(args)
    occ = pattern_occupancy(dense, calib, device, args)
    start = time.time()
    model, masks, trajectory, final = run_iterative_pruning(
        dense, train_ds, test_ds, calib, "fixed_delete", "fixed", device, args
    )
    final["calib_Efix"] = evaluate_fixed_mse(dense, model, calib, device, args)
    final["test_fixed_input_jacobian_diff"] = evaluate_jacobian_diff(
        dense, model, test_ds, device, args
    )
    result = {
        "mode": "calibration",
        "dataset": args.dataset,
        "seed": args.seed,
        "hidden": args.hidden,
        "calib_size": len(calib),
        "calib_index_prefix_sha": str(hash(tuple(idx[: len(calib)]))),
        "occupancy": occ,
        "dense_accuracy": dense_acc,
        "trajectory": trajectory,
        "final": final,
        "runtime_seconds": time.time() - start,
    }
    path = Path(args.output_dir) / f"{args.dataset}_seed{args.seed}_calib{len(calib)}.json"
    json_dump(result, path)
    print(path)


def mode_batch(args):
    device, dense, train_ds, test_ds, calib, *_ = prepare(args)
    start = time.time()
    result = {
        "mode": "batch",
        "dataset": args.dataset,
        "seed": args.seed,
        "calib_size": len(calib),
        "batch_fractions": args.batch_fractions,
        "batch_target": args.batch_target,
        "diagnostic": run_batch_diagnostic(dense, train_ds, test_ds, calib, device, args),
        "runtime_seconds": time.time() - start,
    }
    path = Path(args.output_dir) / f"{args.dataset}_seed{args.seed}_batchdiag.json"
    json_dump(result, path)
    print(path)


def main():
    p = argparse.ArgumentParser()
    sp = p.add_subparsers(dest="mode", required=True)

    f = sp.add_parser("factorial")
    build_common_parser(f)
    f.add_argument("--saliency", choices=["taylor", "fixed_delete", "diag_ggn", "magnitude"], required=True)
    f.add_argument("--finetune", choices=["task", "fixed", "output"], required=True)

    c = sp.add_parser("calibration")
    build_common_parser(c)

    b = sp.add_parser("batch")
    build_common_parser(b)
    b.add_argument("--batch-fractions", type=float, nargs="+", default=[0.01, 0.05, 0.10, 0.50])
    b.add_argument("--batch-target", type=float, default=0.50)

    args = p.parse_args()
    if args.mode == "factorial":
        mode_factorial(args)
    elif args.mode == "calibration":
        mode_calibration(args)
    elif args.mode == "batch":
        mode_batch(args)


if __name__ == "__main__":
    main()
