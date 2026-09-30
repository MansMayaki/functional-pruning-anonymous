#!/usr/bin/env python3
"""Aggregate experiment JSON files into CSV tables and figures.

Reads the directory tree produced by scripts/launch_experiments.py.
Produces:
  factorial_final.csv
  factorial_trajectory.csv
  paired_factorial.csv
  calibration_summary.csv
  batch_one_shot.csv
  batch_recompute.csv
  theory_summary.csv
  cnn_final.csv
  second_order_final.csv (if present)

Diagnostic figures:
  fig_factorial_ft_dynamics_<dataset>.pdf
  fig_calibration_size.pdf
  fig_batch_deletion.pdf
  fig_theory_bound.pdf
  fig_reverse_decomposition.pdf
  fig_cnn_extension.pdf (if CNN results exist)
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from scipy.stats import t as student_t
except Exception:
    student_t = None


def load_jsons(path: Path):
    out = []
    if not path.exists():
        return out
    for p in sorted(path.glob("*.json")):
        try:
            with p.open() as f:
                obj = json.load(f)
            obj["_file"] = str(p)
            out.append(obj)
        except Exception as e:
            print(f"[warn] could not read {p}: {e}")
    return out


def flatten_final(objs):
    rows = []
    for o in objs:
        r = {
            "dataset": o.get("dataset"),
            "seed": o.get("seed"),
            "saliency": o.get("saliency"),
            "finetune": o.get("finetune"),
            "dense_accuracy": o.get("dense_accuracy"),
            "calib_size": o.get("calib_size"),
            "lambda_reg": o.get("lambda_reg"),
            "file": o.get("_file"),
        }
        for k, v in o.get("final", {}).items():
            if isinstance(v, (int, float)) or v is None:
                r[k] = v
        rows.append(r)
    return pd.DataFrame(rows)


def flatten_trajectory(objs):
    rows = []
    for o in objs:
        base = {
            "dataset": o.get("dataset"), "seed": o.get("seed"),
            "saliency": o.get("saliency"), "finetune": o.get("finetune")
        }
        for t in o.get("trajectory", []):
            row = dict(base)
            row.update({k: v for k, v in t.items() if isinstance(v, (int,float,str)) or v is None})
            rows.append(row)
    return pd.DataFrame(rows)


def paired_ci(x: np.ndarray):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n == 0:
        return (np.nan, np.nan, np.nan, 0)
    mean = float(x.mean())
    if n == 1:
        return (mean, np.nan, np.nan, 1)
    sd = float(x.std(ddof=1))
    crit = float(student_t.ppf(0.975, n - 1)) if student_t is not None else 1.96
    half = crit * sd / math.sqrt(n)
    return mean, mean - half, mean + half, n


def make_paired_factorial(df):
    if df.empty:
        return pd.DataFrame()
    metrics = [
        "accuracy", "D_fix", "D_gate", "D_func", "kappa", "twoC",
        "centered_logit_l2", "kl_dense_sparse", "js_dense_sparse",
        "teacher_agreement", "gate_flip_rate",
        "reverse_kappa", "Drev_gate_dense", "Drev_param_sparse",
    ]
    comparisons = [
        ("taylor", "fixed", "taylor", "output", "Taylor: fixed vs output FT"),
        ("fixed_delete", "fixed", "fixed_delete", "output", "Deletion: fixed vs output FT"),
        ("taylor", "task", "taylor", "fixed", "Taylor: task vs fixed FT"),
        ("taylor", "task", "taylor", "output", "Taylor: task vs output FT"),
    ]
    rows = []
    for dataset in sorted(df.dataset.dropna().unique()):
        d = df[df.dataset == dataset]
        for s1, f1, s2, f2, label in comparisons:
            a = d[(d.saliency == s1) & (d.finetune == f1)].set_index("seed")
            b = d[(d.saliency == s2) & (d.finetune == f2)].set_index("seed")
            common = a.index.intersection(b.index)
            for metric in metrics:
                if metric not in a.columns or metric not in b.columns or len(common) == 0:
                    continue
                av = a.loc[common, metric].astype(float).to_numpy()
                bv = b.loc[common, metric].astype(float).to_numpy()
                diff = av - bv
                ratio = av / np.where(np.abs(bv) > 1e-15, bv, np.nan)
                md, lo, hi, n = paired_ci(diff)
                mr, rlo, rhi, _ = paired_ci(ratio[np.isfinite(ratio)])
                rows.append({
                    "dataset": dataset, "comparison": label, "metric": metric,
                    "n_pairs": n, "mean_a_minus_b": md, "ci95_low": lo,
                    "ci95_high": hi, "mean_ratio_a_over_b": mr,
                    "ratio_ci95_low": rlo, "ratio_ci95_high": rhi,
                })
    return pd.DataFrame(rows)


def calibration_df(objs):
    rows=[]
    for o in objs:
        r={"dataset":o.get("dataset"),"seed":o.get("seed"),"calib_size":o.get("calib_size")}
        for k,v in o.get("occupancy",{}).items(): r[k]=v
        for k,v in o.get("final",{}).items():
            if isinstance(v,(int,float)) or v is None: r[k]=v
        rows.append(r)
    return pd.DataFrame(rows)


def batch_dfs(objs):
    one=[]; rec=[]
    for o in objs:
        base={"dataset":o.get("dataset"),"seed":o.get("seed"),"calib_size":o.get("calib_size")}
        for r in o.get("diagnostic",{}).get("one_shot",[]):
            z=dict(base); z.update({k:v for k,v in r.items() if isinstance(v,(int,float)) or v is None}); one.append(z)
        for r in o.get("diagnostic",{}).get("recompute",[]):
            z=dict(base)
            for k,v in r.items():
                if isinstance(v,(int,float)) or v is None: z[k]=v
            for prefix in ["pre_ft_metrics","post_ft_metrics"]:
                for k,v in r.get(prefix,{}).items():
                    if isinstance(v,(int,float)) or v is None: z[f"{prefix}_{k}"]=v
            rec.append(z)
    return pd.DataFrame(one),pd.DataFrame(rec)


def theory_df(objs):
    rows=[]
    for o in objs:
        r={"dataset":o.get("dataset"),"seed":o.get("seed"),"saliency":o.get("saliency"),"finetune":o.get("finetune")}
        ge=o.get("gate_error",{})
        r["pearson_bound_actual"]=ge.get("pearson_bound_vs_actual")
        r["spearman_bound_actual"]=ge.get("spearman_bound_vs_actual")
        for name in ["actual_norm","bound_rhs","bound_over_actual","exact_telescope_residual"]:
            q=ge.get(name,{})
            for stat in ["mean","median","p90","p95","max"]:
                r[f"{name}_{stat}"]=q.get(stat)
        mar=o.get("margin",{})
        r["margin_violation_rate"]=mar.get("violation_rate")
        layers=mar.get("per_layer",[])
        aucs=[x.get("auc_negative_dense_margin") for x in layers if x.get("auc_negative_dense_margin") is not None]
        r["mean_auc_negative_margin"]=float(np.mean(aucs)) if aucs else np.nan
        rows.append(r)
    return pd.DataFrame(rows)


def set_style():
    plt.rcParams.update({"font.size":9,"axes.labelsize":9,"axes.titlesize":9,"legend.fontsize":8,"pdf.fonttype":42,"ps.fonttype":42})


def savefig(fig,path):
    path.parent.mkdir(parents=True,exist_ok=True); fig.savefig(path,bbox_inches="tight"); fig.savefig(path.with_suffix(".png"),dpi=250,bbox_inches="tight"); plt.close(fig)


def plot_factorial_dynamics(traj,outdir):
    if traj.empty: return
    final_s=max(traj.target_sparsity.dropna().astype(float))
    d=traj[np.isclose(traj.target_sparsity.astype(float),final_s)]
    for dataset in sorted(d.dataset.unique()):
        # Most important controlled comparison: same Taylor saliency, different FT.
        z=d[(d.dataset==dataset)&(d.saliency=="taylor")&d.finetune.isin(["fixed","output"])]
        if z.empty: continue
        fig,axs=plt.subplots(1,4,figsize=(7.2,2.2))
        metrics=[("D_fix",r"$D_{\rm fix}$"),("D_gate",r"$D_{\rm gate}$"),("D_func",r"$D_{\rm func}$"),("twoC",r"$2\mathcal{C}$")]
        for ax,(metric,label) in zip(axs,metrics):
            for ft in ["fixed","output"]:
                q=z[z.finetune==ft].copy(); q["step"]=q["epoch"].astype(int)
                g=q.groupby("step")[metric].agg(["mean","std"]).reset_index()
                ax.errorbar(g.step,g["mean"],yerr=g["std"],marker="o",capsize=2,label=ft)
            ax.axhline(0,linewidth=.7,linestyle=":") if metric=="twoC" else None
            ax.set_xlabel("Fine-tuning epoch"); ax.set_ylabel(label); ax.set_xticks([0,1,2,3])
        axs[0].legend(frameon=False); fig.suptitle(f"{dataset}: Taylor saliency, 90% pruning")
        fig.tight_layout(); savefig(fig,outdir/f"fig_factorial_ft_dynamics_{dataset}.pdf")


def plot_calibration(df,outdir):
    if df.empty: return
    z=df[df.dataset==df.dataset.iloc[0]]
    g=z.groupby("calib_size").agg({"unique_fraction":["mean","std"],"singleton_sample_fraction":["mean","std"],"D_fix":["mean","std"],"test_fixed_input_jacobian_diff":["mean","std"]})
    g.columns=["_".join(c) for c in g.columns]; g=g.reset_index()
    fig,axs=plt.subplots(1,4,figsize=(7.2,2.2))
    specs=[("unique_fraction","Unique pattern fraction"),("singleton_sample_fraction","Singleton sample fraction"),("D_fix",r"Test $D_{\rm fix}$"),("test_fixed_input_jacobian_diff","Test Jacobian discrepancy")]
    for ax,(m,lbl) in zip(axs,specs):
        ax.errorbar(g.calib_size,g[f"{m}_mean"],yerr=g[f"{m}_std"],marker="o",capsize=2); ax.set_xscale("log",base=2); ax.set_xlabel(r"$N_{\rm cal}$"); ax.set_ylabel(lbl)
    fig.tight_layout(); savefig(fig,outdir/"fig_calibration_size.pdf")


def plot_batch(one,rec,outdir):
    if one.empty and rec.empty: return
    fig,axs=plt.subplots(1,2,figsize=(5.6,2.3))
    if not one.empty:
        g=one.groupby("fraction_total_weights")["actual_over_predicted"].agg(["mean","std"]).reset_index()
        axs[0].errorbar(100*g.fraction_total_weights,g["mean"],yerr=g["std"],marker="o",capsize=2); axs[0].axhline(1,linestyle=":",linewidth=.8); axs[0].set_xlabel("One-shot deletion batch (% weights)"); axs[0].set_ylabel("Actual / summed marginal cost")
    if not rec.empty:
        m="post_ft_metrics_D_fix" if "post_ft_metrics_D_fix" in rec.columns else "pre_ft_metrics_D_fix"
        g=rec.groupby("chunk_fraction_total_weights")[m].agg(["mean","std"]).reset_index()
        axs[1].errorbar(100*g.chunk_fraction_total_weights,g["mean"],yerr=g["std"],marker="o",capsize=2); axs[1].set_xlabel("Re-ranking chunk (% weights)"); axs[1].set_ylabel(r"Final $D_{\rm fix}$")
    fig.tight_layout(); savefig(fig,outdir/"fig_batch_deletion.pdf")


def plot_reverse(df,outdir):
    if df.empty:return
    z=df[df.saliency=="taylor"]
    fig,axs=plt.subplots(1,2,figsize=(5.6,2.3))
    x=0
    labels=[]
    for dataset in sorted(z.dataset.unique()):
        for ft in ["fixed","output"]:
            q=z[(z.dataset==dataset)&(z.finetune==ft)]
            if q.empty: continue
            vals=[q.kappa.mean(),q.reverse_kappa.mean()]
            errs=[q.kappa.std(ddof=1),q.reverse_kappa.std(ddof=1)]
            axs[0].errorbar([x,x+0.25],vals,yerr=errs,marker="o",capsize=2); labels.append(f"{dataset}-{ft}"); x+=1
    # simpler second panel: output minus fixed kappa by convention
    rows=[]
    for dataset in sorted(z.dataset.unique()):
        a=z[(z.dataset==dataset)&(z.finetune=="fixed")].set_index("seed")
        b=z[(z.dataset==dataset)&(z.finetune=="output")].set_index("seed"); common=a.index.intersection(b.index)
        if len(common):
            rows.append((dataset,(b.loc[common,"kappa"]-a.loc[common,"kappa"]).mean(),(b.loc[common,"reverse_kappa"]-a.loc[common,"reverse_kappa"]).mean()))
    if rows:
        xx=np.arange(len(rows)); axs[1].plot(xx,[r[1] for r in rows],marker="o",label="dense-reference"); axs[1].plot(xx,[r[2] for r in rows],marker="s",label="reverse"); axs[1].axhline(0,linestyle=":",linewidth=.8); axs[1].set_xticks(xx,[r[0] for r in rows]); axs[1].set_ylabel(r"Output FT $\kappa$ - fixed FT $\kappa$"); axs[1].legend(frameon=False)
    axs[0].set_ylabel(r"Cancellation index $\kappa$"); axs[0].set_title("Both decomposition conventions"); fig.tight_layout(); savefig(fig,outdir/"fig_reverse_decomposition.pdf")


def plot_cnn(df,outdir):
    if df.empty:return
    requested=["D_fix","D_func","kappa","teacher_agreement"]
    metrics=[m for m in requested if m in df.columns]
    if not metrics:
        return
    g=df.groupby("finetune")[metrics].agg(["mean","std"])
    fig,axs=plt.subplots(1,4,figsize=(7.2,2.2))
    fts=[x for x in ["task","fixed","output"] if x in g.index]
    for ax,m in zip(axs,metrics):
        vals=[g.loc[x,(m,"mean")] for x in fts]; err=[g.loc[x,(m,"std")] for x in fts]
        ax.errorbar(np.arange(len(fts)),vals,yerr=err,fmt="o",capsize=2); ax.set_xticks(np.arange(len(fts)),fts,rotation=25); ax.set_ylabel(m)
    fig.tight_layout(); savefig(fig,outdir/"fig_cnn_extension.pdf")


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--root",default="results"); ap.add_argument("--outdir",default="analysis"); args=ap.parse_args()
    root=Path(args.root); out=Path(args.outdir); out.mkdir(parents=True,exist_ok=True); set_style()

    fobjs=load_jsons(root/"factorial"); fdf=flatten_final(fobjs); tdf=flatten_trajectory(fobjs)
    fdf.to_csv(out/"factorial_final.csv",index=False); tdf.to_csv(out/"factorial_trajectory.csv",index=False)
    paired=make_paired_factorial(fdf); paired.to_csv(out/"paired_factorial.csv",index=False)

    cdf=calibration_df(load_jsons(root/"calibration")); cdf.to_csv(out/"calibration_summary.csv",index=False)
    one,rec=batch_dfs(load_jsons(root/"batch")); one.to_csv(out/"batch_one_shot.csv",index=False); rec.to_csv(out/"batch_recompute.csv",index=False)
    th=theory_df(load_jsons(root/"theory")); th.to_csv(out/"theory_summary.csv",index=False)
    cnn=flatten_final(load_jsons(root/"cnn")); cnn.to_csv(out/"cnn_final.csv",index=False)
    so=flatten_final(load_jsons(root/"second_order")); so.to_csv(out/"second_order_final.csv",index=False)

    plot_factorial_dynamics(tdf,out); plot_calibration(cdf,out); plot_batch(one,rec,out); plot_reverse(fdf,out); plot_cnn(cnn,out)
    print(f"Wrote analysis to {out.resolve()}")

if __name__=="__main__": main()
