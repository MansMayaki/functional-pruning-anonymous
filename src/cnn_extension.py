#!/usr/bin/env python3
"""
Small CIFAR-10 CNN diagnostic extension for the functional-pruning study.

Purpose
-------
This is not used to claim that the MLP theory covers arbitrary CNNs. It tests
whether the central *diagnostic phenomenon* survives outside fully-connected
MLPs while holding pruning saliency fixed.

All three cells use exactly the same first-order task-Taylor pruning masks at
each stage, but different fine-tuning objectives:
    task, fixed-gate, output matching.
This isolates the effect of the fine-tuning objective while holding the pruning rule fixed.

The CNN contains only affine Conv/Linear layers and ReLUs (strided convolutions
replace pooling), so prescribed ReLU gates are straightforward to evaluate.
No fixed-deletion theorem is invoked for this extension.

Example
-------
python src/cnn_extension.py --seed 0 --finetune output \
  --output-dir results/cnn
"""

from __future__ import annotations

import argparse
import copy
import json
import random
import time
from collections import Counter
from pathlib import Path
from typing import List, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


def seed_all(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def make_loader(ds, batch, shuffle, seed, workers):
    g = torch.Generator().manual_seed(seed) if shuffle else None
    return DataLoader(ds, batch_size=batch, shuffle=shuffle, num_workers=workers,
                      pin_memory=torch.cuda.is_available(), generator=g)


class SmallReLUCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, stride=2, padding=1)
        self.fc1 = nn.Linear(128 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, 10)

    def prunable_layers(self):
        return [self.conv1, self.conv2, self.conv3, self.fc1, self.fc2]

    def forward(self, x, return_gates=False):
        gates = []
        z = self.conv1(x); g = z > 0; x = z * g.to(z.dtype); gates.append(g)
        z = self.conv2(x); g = z > 0; x = z * g.to(z.dtype); gates.append(g)
        z = self.conv3(x); g = z > 0; x = z * g.to(z.dtype); gates.append(g)
        x = x.flatten(1)
        z = self.fc1(x); g = z > 0; x = z * g.to(z.dtype); gates.append(g)
        out = self.fc2(x)
        return (out, gates) if return_gates else out

    def forward_with_gates(self, x, gates):
        z = self.conv1(x); x = z * gates[0].to(z.dtype)
        z = self.conv2(x); x = z * gates[1].to(z.dtype)
        z = self.conv3(x); x = z * gates[2].to(z.dtype)
        x = x.flatten(1)
        z = self.fc1(x); x = z * gates[3].to(z.dtype)
        return self.fc2(x)


def make_data(data_dir, seed):
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2470, 0.2435, 0.2616)
    train_t = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    eval_t = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
    train_aug = datasets.CIFAR10(data_dir, train=True, download=True, transform=train_t)
    train_eval = datasets.CIFAR10(data_dir, train=True, download=True, transform=eval_t)
    test = datasets.CIFAR10(data_dir, train=False, download=True, transform=eval_t)
    rng = np.random.default_rng(100000 + seed)
    calib_idx = rng.permutation(len(train_eval))[:4096].tolist()
    return train_aug, train_eval, test, calib_idx


def make_masks(model, device):
    return [torch.ones_like(m.weight, device=device) for m in model.prunable_layers()]


def apply_masks_(model, masks):
    with torch.no_grad():
        for m, mask in zip(model.prunable_layers(), masks):
            m.weight.mul_(mask)


def zero_masked_grads_(model, masks):
    for m, mask in zip(model.prunable_layers(), masks):
        if m.weight.grad is not None:
            m.weight.grad.mul_(mask)


def sparsity(masks):
    total = sum(x.numel() for x in masks)
    active = sum(int(x.sum().item()) for x in masks)
    return 1.0 - active / total


def prune_to_target_(masks, scores, target):
    total = sum(x.numel() for x in masks)
    active = sum(int(x.sum().item()) for x in masks)
    target_active = int(round(total * (1.0 - target)))
    npr = max(0, active - target_active)
    if npr == 0: return 0
    vals, refs = [], []
    for li, (mask, score) in enumerate(zip(masks, scores)):
        idx = torch.nonzero(mask.view(-1) > 0, as_tuple=False).squeeze(1)
        vals.append(score.view(-1)[idx].detach().cpu())
        refs.extend((li, int(j)) for j in idx.cpu().tolist())
    allv = torch.cat(vals)
    chosen = torch.topk(allv, k=npr, largest=False).indices.tolist()
    for c in chosen:
        li, j = refs[c]; masks[li].view(-1)[j] = 0
    return npr


def accuracy(model, ds, device, args):
    model.eval(); ok = n = 0
    loader = make_loader(ds, args.eval_batch_size, False, 0, args.num_workers)
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            ok += int((model(xb).argmax(1) == yb).sum().item()); n += yb.numel()
    return ok / n


def train_dense(model, train_ds, test_ds, device, args):
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=5e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.dense_epochs)
    hist = []
    for ep in range(args.dense_epochs):
        seed_all(args.seed * 10000 + ep)
        model.train()
        loader = make_loader(train_ds, args.batch_size, True, 1000 + args.seed * 100 + ep, args.num_workers)
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(xb), yb)
            loss.backward(); opt.step()
        sched.step()
        if ep == args.dense_epochs - 1 or ep in {9, 19}:
            hist.append({"epoch": ep + 1, "test_accuracy": accuracy(model, test_ds, device, args)})
    return hist


def load_or_train_dense(args, device, train_ds, test_ds):
    cache = Path(args.dense_cache_dir); cache.mkdir(parents=True, exist_ok=True)
    path = cache / f"cifar10_smallcnn_seed{args.seed}.pt"
    model = SmallReLUCNN().to(device)
    if path.exists() and not args.retrain_dense:
        obj = torch.load(path, map_location=device); model.load_state_dict(obj["state_dict"])
        hist = obj.get("history", [])
    else:
        seed_all(args.seed); hist = train_dense(model, train_ds, test_ds, device, args)
        torch.save({"state_dict": {k:v.detach().cpu() for k,v in model.state_dict().items()}, "history":hist}, path)
    return model, str(path), hist


def taylor_scores(model, masks, calib_ds, device, args):
    model.zero_grad(set_to_none=True)
    loader = make_loader(calib_ds, args.batch_size, True, 70000 + args.seed, args.num_workers)
    for bi, (xb, yb) in enumerate(loader):
        if bi >= args.saliency_batches: break
        xb, yb = xb.to(device), yb.to(device)
        (F.cross_entropy(model(xb), yb) / args.saliency_batches).backward()
    scores = []
    for m, mask in zip(model.prunable_layers(), masks):
        g = m.weight.grad if m.weight.grad is not None else torch.zeros_like(m.weight)
        scores.append((m.weight.detach() * g.detach()).abs() * mask)
    model.zero_grad(set_to_none=True)
    return scores


def finetune_epoch(model, dense, masks, train_ds, objective, stage, ep, device, args, optimizer=None):
    model.train(); dense.eval()
    opt = optimizer if optimizer is not None else torch.optim.Adam(model.parameters(), lr=args.ft_lr)
    seed_all(200000 + args.seed * 1000 + stage * 10 + ep)
    loader = make_loader(train_ds, args.batch_size, True, 200000 + args.seed*1000 + stage*10 + ep, args.num_workers)
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        opt.zero_grad(set_to_none=True)
        out = model(xb); task = F.cross_entropy(out, yb)
        match = torch.zeros((), device=device)
        if objective == "fixed":
            with torch.no_grad(): dlog, dg = dense(xb, return_gates=True)
            match = (model.forward_with_gates(xb, dg) - dlog).pow(2).sum(dim=1).mean()
        elif objective == "output":
            with torch.no_grad(): dlog = dense(xb)
            match = (out - dlog).pow(2).sum(dim=1).mean()
        loss = task + (args.lambda_reg * match if objective != "task" else 0.0)
        loss.backward(); zero_masked_grads_(model, masks); opt.step(); apply_masks_(model, masks)


def cosine(a,b,eps=1e-12):
    na=a.norm(dim=1); nb=b.norm(dim=1); good=(na>eps)&(nb>eps)
    c=torch.zeros_like(na); c[good]=(a[good]*b[good]).sum(1)/(na[good]*nb[good]); return c,good


def evaluate(dense, sparse, test_ds, device, args):
    loader = make_loader(test_ds, args.eval_batch_size, False, 0, args.num_workers)
    sums = Counter(); n=0; csum=crsum=0.; cn=crn=0
    dense.eval(); sparse.eval()
    with torch.no_grad():
        for xb,yb in loader:
            xb,yb=xb.to(device),yb.to(device)
            dl,dg=dense(xb,return_gates=True); sl,sg=sparse(xb,return_gates=True)
            sud=sparse.forward_with_gates(xb,dg); dus=dense.forward_with_gates(xb,sg)
            vf=dl-sud; vg=sud-sl; v=dl-sl
            rg=dl-dus; rp=dus-sl
            b=xb.shape[0]; n+=b
            for name,z in [("fix",vf),("gate",vg),("func",v)]:
                sums[f"D_{name}"] += float(z.norm(dim=1).sum().item())
                sums[f"E_{name}"] += float(z.pow(2).sum(dim=1).sum().item())
            sums["C"] += float((vf*vg).sum(1).sum().item())
            for name,z in [("gate_dense",rg),("param_sparse",rp)]:
                sums[f"Drev_{name}"] += float(z.norm(dim=1).sum().item())
                sums[f"Erev_{name}"] += float(z.pow(2).sum(dim=1).sum().item())
            sums["Crev"] += float((rg*rp).sum(1).sum().item())
            c,g=cosine(vf,vg); csum+=float(c[g].sum()); cn+=int(g.sum())
            c,g=cosine(rg,rp); crsum+=float(c[g].sum()); crn+=int(g.sum())
            dc=dl-dl.mean(1,keepdim=True); sc=sl-sl.mean(1,keepdim=True)
            sums["centered_logit_l2"] += float((dc-sc).norm(dim=1).sum())
            p=dl.softmax(1); q=sl.softmax(1); lp=dl.log_softmax(1); lq=sl.log_softmax(1)
            sums["kl"] += float((p*(lp-lq)).sum(1).sum())
            m=0.5*(p+q)
            sums["js"] += float((0.5*((p*(p.clamp_min(1e-12).log()-m.log())).sum(1)+(q*(q.clamp_min(1e-12).log()-m.log())).sum(1))).sum())
            sums["agreement"] += int((dl.argmax(1)==sl.argmax(1)).sum())
            sums["accuracy"] += int((sl.argmax(1)==yb).sum())
            flips=tot=0
            for a,bg in zip(dg,sg): flips+=int((a!=bg).sum()); tot+=a.numel()
            sums["flips"]+=flips; sums["gate_total"]+=tot
    out={k:v/n for k,v in sums.items() if k not in {"flips","gate_total"}}
    out["accuracy"]=sums["accuracy"]/n; out["agreement"]=sums["agreement"]/n
    out["gate_flip_rate"]=sums["flips"]/sums["gate_total"]
    out["mean_cosine"]=csum/max(cn,1); out["reverse_mean_cosine"]=crsum/max(crn,1)
    out["twoC"]=2*out["C"]; den=out["E_fix"]+out["E_gate"]; out["kappa"]=2*out["C"]/den
    denr=out["Erev_gate_dense"]+out["Erev_param_sparse"]; out["reverse_kappa"]=2*out["Crev"]/denr
    return out


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--seed",type=int,default=0)
    p.add_argument("--finetune",choices=["task","fixed","output"],required=True)
    p.add_argument("--data-dir",default="./data")
    p.add_argument("--dense-cache-dir",default="results/dense_cache")
    p.add_argument("--output-dir",default="results/cnn")
    p.add_argument("--retrain-dense",action="store_true")
    p.add_argument("--dense-epochs",type=int,default=40)
    p.add_argument("--ft-epochs",type=int,default=3)
    p.add_argument("--batch-size",type=int,default=128)
    p.add_argument("--eval-batch-size",type=int,default=256)
    p.add_argument("--num-workers",type=int,default=2)
    p.add_argument("--lr",type=float,default=1e-3)
    p.add_argument("--ft-lr",type=float,default=3e-4)
    p.add_argument("--lambda-reg",type=float,default=1e-2)
    p.add_argument("--saliency-batches",type=int,default=8)
    p.add_argument("--sparsities",type=float,nargs="+",default=[.5,.7,.8,.9])
    p.add_argument("--device",default="cuda" if torch.cuda.is_available() else "cpu")
    args=p.parse_args()
    seed_all(args.seed); device=torch.device(args.device)
    train_aug, train_eval, test, calib_idx=make_data(args.data_dir,args.seed)
    calib=Subset(train_eval,calib_idx)
    dense,dense_path,hist=load_or_train_dense(args,device,train_aug,test)
    model=copy.deepcopy(dense).to(device); masks=make_masks(model,device); traj=[]
    for si,target in enumerate(args.sparsities):
        scores=taylor_scores(model,masks,calib,device,args); prune_to_target_(masks,scores,target); apply_masks_(model,masks)
        traj.append({"stage":si,"sparsity":target,"phase":"post_prune_pre_ft","epoch":0,**evaluate(dense,model,test,device,args)})
        ft_optimizer = torch.optim.Adam(model.parameters(), lr=args.ft_lr)
        for ep in range(1,args.ft_epochs+1):
            finetune_epoch(model,dense,masks,train_aug,args.finetune,si,ep,device,args,optimizer=ft_optimizer)
            traj.append({"stage":si,"sparsity":target,"phase":"finetune","epoch":ep,**evaluate(dense,model,test,device,args)})
    result={"dataset":"cifar10","architecture":"SmallReLUCNN","seed":args.seed,"saliency":"task_taylor_common","finetune":args.finetune,"dense_checkpoint":dense_path,"dense_history":hist,"dense_accuracy":accuracy(dense,test,device,args),"trajectory":traj,"final":traj[-1]}
    out=Path(args.output_dir); out.mkdir(parents=True,exist_ok=True); path=out/f"cifar10_seed{args.seed}_taylor_{args.finetune}.json"
    with path.open("w") as f: json.dump(result,f,indent=2,sort_keys=True)
    print(path)

if __name__=="__main__": main()
