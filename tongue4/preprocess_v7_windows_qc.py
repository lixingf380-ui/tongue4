#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""V7 continuous-signal preprocessing, session-safe sliding windows and QC.

Designed only for synthetic data produced by generate_v7_continuous_synthetic.py.
Preprocessing operates on each continuous session BEFORE slicing windows.
Labels/quality/oracle artifacts are targets only; never included in model features.

Usage: python preprocess_v7_windows_qc.py --dataset-dir dataset_v7_continuous_synthetic
       python preprocess_v7_windows_qc.py --dataset-dir dataset_v7_smoke --window-size 128 --stride 32
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np


def parse_args():
    p=argparse.ArgumentParser(description="Session-level preprocess, windows and QC for V7")
    p.add_argument("--dataset-dir",type=Path,default=Path("dataset_v7_continuous_synthetic"))
    p.add_argument("--window-size",type=int,default=128)
    p.add_argument("--stride",type=int,default=32)
    p.add_argument("--min-event-coverage",type=float,default=.45,
                   help="Fraction of the window occupied by its center-frame event; otherwise y=-1")
    p.add_argument("--no-plots",action="store_true",help="Skip matplotlib, still make CSV+JSON QC")
    p.add_argument("--overwrite",action="store_true")
    return p.parse_args()


def format_eta(s):
    if s is None:return "--:--:--"
    s=max(0,int(s)); h,r=divmod(s,3600);m,s=divmod(r,60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def get_continuous_features(x, fs):
    """Causal EMA across a full session (no future test statistics).
    Three modalities: fixed-scale raw amplitude; EMA smoothed; sample derivative.
    No per-window min/max or fitted test-distribution statistics.
    """
    n,c=x.shape
    ema=np.empty_like(x,dtype=np.float32)
    alpha=float(1.0-np.exp(-1.0/(fs*.22)))
    state=x[0].astype(np.float32).copy()
    for t in range(n):
        state=state+alpha*(x[t]-state)
        ema[t]=state
    derivative=np.diff(x,axis=0,prepend=x[:1])
    # 1.25 is the fixed synthetic sensor range, not inferred from the test set.
    features=np.stack([x/1.25,ema/1.25,np.clip(derivative/.30,-1,1)],axis=-1)
    return features.astype(np.float32)


def session_windows(path, conf, window, stride, min_coverage):
    with np.load(path,allow_pickle=False) as d:
        x=d["x_raw"].astype(np.float32)
        y=d["y_frame"].astype(np.int16)
        eid=d["event_id"].astype(np.int32)
        intensity=d["control_intensity_frame"].astype(np.float32)
        mask=d["mask_frame"].astype(np.uint8)
        quality=d["quality_frame"].astype(np.float32)
        phase=d["phase_frame"].astype(np.float32)
        nuisance=d["nuisance_frame"].astype(np.uint8)
        fs=int(d["fs"])
        session_id=str(d["session_id"].item())
    if x.ndim!=2 or x.shape[1]!=6:raise ValueError(f"Expected [T,6]: {path}, got {x.shape}")
    if not (len(x)==len(y)==len(eid)==len(intensity)==len(mask)==len(quality)==len(phase)==len(nuisance)):
        raise ValueError(f"Length mismatch: {path}")
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(quality)):
        raise ValueError(f"Nonfinite data: {path}")
    if len(x)<window:return None
    feats=get_continuous_features(x,fs)
    acc={k:[] for k in ("x","y_command10","is_event","target_mask","control_intensity",
                         "quality_seq","target_quality","nuisance_seq","target_nuisance_fraction",
                         "event_coverage","event_phase_center",
                         "window_start","window_center","event_id")}
    for st in range(0,len(x)-window+1,stride):
        en=st+window
        mid=st+window//2
        center_event_id=int(eid[mid])
        if center_event_id>=0:
            coverage=float(np.count_nonzero(eid[st:en]==center_event_id))/window
            label=int(y[mid]) if coverage>=min_coverage else -1
        else:
            coverage=0.0
            label=-1
        acc["x"].append(feats[st:en])
        acc["y_command10"].append(label)
        acc["is_event"].append(int(label>=0))
        acc["target_mask"].append(mask[mid] if label>=0 else np.zeros(6,dtype=np.uint8))
        acc["control_intensity"].append(float(intensity[mid]) if label>=0 else 0.0)
        acc["quality_seq"].append(quality[st:en].astype(np.float16))
        acc["target_quality"].append(quality[st:en].mean(axis=0))
        acc["nuisance_seq"].append(nuisance[st:en])
        acc["target_nuisance_fraction"].append(nuisance[st:en].mean(axis=0))
        acc["event_coverage"].append(coverage)
        acc["event_phase_center"].append(float(phase[mid]) if label>=0 else -1.)
        acc["window_start"].append(st)
        acc["window_center"].append(mid)
        acc["event_id"].append(center_event_id if label>=0 else -1)
    result={k:np.asarray(v) for k,v in acc.items()}
    result["y_command10"]=result["y_command10"].astype(np.int16)
    result["is_event"]=result["is_event"].astype(np.uint8)
    result["control_intensity"]=result["control_intensity"].astype(np.float32)
    result["target_quality"]=result["target_quality"].astype(np.float32)
    result["nuisance_seq"]=result["nuisance_seq"].astype(np.uint8)
    result["target_nuisance_fraction"]=result["target_nuisance_fraction"].astype(np.float32)
    result["event_coverage"]=result["event_coverage"].astype(np.float32)
    result["event_phase_center"]=result["event_phase_center"].astype(np.float32)
    result["window_start"]=result["window_start"].astype(np.int32)
    result["window_center"]=result["window_center"].astype(np.int32)
    result["event_id"]=result["event_id"].astype(np.int32)
    result["session_id"]=np.repeat(np.array([session_id]),len(result["y_command10"]))
    return result


def write_csv(path,rows):
    if not rows:return
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("w",encoding="utf-8-sig",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0].keys()));w.writeheader();w.writerows(rows)


def plots(root, filenames, conf, report, window, stride):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib missing: QC PNG plots skipped. CSV and JSON reports remain available.",flush=True)
        return
    img_dir=root/"qc_figures";img_dir.mkdir(exist_ok=True)
    available={n:list((root/"sessions"/n).glob("*.npz")) for n in filenames}
    for name in ["train_generatorA_mixed","test_generatorA_nominal","test_generatorA_high", "test_generatorB_ood"]:
        if name not in available or not available[name]:continue
        with np.load(sorted(available[name])[0],allow_pickle=False) as d:
            x=d["x_raw"].astype(np.float32)
            y=d["y_frame"].astype(np.int32)
            q=d["quality_frame"].astype(np.float32)
            fs=int(d["fs"])
        duration=14
        k=min(len(x),fs*duration)
        t=np.arange(k)/fs
        fig,ax=plt.subplots(2,1,figsize=(12,5.5),sharex=True,layout="constrained",
                             gridspec_kw={"height_ratios":[3,1]})
        colors=["#2459a9","#159c90","#d58f2c","#9065a8","#cf6679","#64748b"]
        for ch in range(6):
            ax[0].plot(t,x[:k,ch]+.9*(5-ch),color=colors[ch],lw=.9,label=f"Ch{ch}")
        ax[0].set_yticks([.9*(5-ch) for ch in range(6)], [f"Ch{ch}" for ch in range(6)])
        ax[0].set_ylabel("6 channels (offset)")
        ax[0].set_title(f"SYNTHETIC continuous pressure | {name}")
        ax[1].plot(t,q[:k].mean(axis=1),color="#3366aa",lw=1.1,label="mean simulated quality")
        ax[1].fill_between(t,0,1,where=(y[:k]>=0),alpha=.10,color="teal",label="command-event region")
        ax[1].set_ylim(0,1.06)
        ax[1].set_ylabel("Quality")
        ax[1].set_xlabel("Time (s)")
        ax[1].legend(loc="lower left",fontsize=8)
        fig.savefig(img_dir/f"{name}_waveform.png",dpi=180)
        fig.savefig(img_dir/f"{name}_waveform.svg")
        plt.close(fig)
    labels=list(conf["command_names"])+["idle/transition"]
    names=["train_generatorA_mixed","validation_generatorA_mixed","test_generatorA_nominal",
           "test_generatorA_high","test_generatorB_ood"]
    rows=[r for r in report if r["group"] in names]
    if rows:
        fig,ax=plt.subplots(figsize=(12,4.3),layout="constrained")
        vals=np.array([[int(r.get(f"class_{j}",0)) for j in list(range(10))+[-1]] for r in rows],float)
        vals=vals/(vals.sum(axis=1,keepdims=True)+1e-9)
        b=np.arange(len(rows));bottom=np.zeros(len(rows))
        cmap=plt.get_cmap("tab20")
        for i,label in enumerate(labels):
            ax.bar(b,vals[:,i],bottom=bottom,label=label,color=cmap(i),width=.72)
            bottom+=vals[:,i]
        ax.set_xticks(b,[r["group"].replace("generator","G-") for r in rows],rotation=20,ha="right")
        ax.set_ylabel("Window proportion")
        ax.set_title("Label distribution: session-disjoint V7 splits")
        ax.legend(bbox_to_anchor=(1.01,1),loc="upper left",fontsize=7)
        fig.savefig(img_dir/"window_label_distribution.png",dpi=180)
        plt.close(fig)


def main():
    a=parse_args()
    root=a.dataset_dir
    if a.window_size<16 or a.stride<1 or not 0<a.min_event_coverage<=1:
        raise ValueError("Invalid window-size, stride or min-event-coverage")
    conf_file=root/"v7_config.json"
    if not conf_file.exists():raise FileNotFoundError(f"No generator config: {conf_file}")
    conf=json.loads(conf_file.read_text(encoding="utf-8"))
    source=root/"sessions"
    folders=sorted(d for d in source.iterdir() if d.is_dir())
    if not folders:raise RuntimeError(f"No session records in {source}")
    output_dir=root/"windows";output_dir.mkdir(exist_ok=True)
    report=[]; session_index=defaultdict(set); files_by_group={};probe_by_group={};start=time.perf_counter()
    print(f"V7 preprocessing | window={a.window_size}, stride={a.stride}, min_event_coverage={a.min_event_coverage}",flush=True)
    for i,d in enumerate(folders,1):
        files=sorted(d.glob("*.npz"))
        if not files:continue
        files_by_group[d.name]=files
        parts=[]
        for f in files:
            result=session_windows(f,conf,a.window_size,a.stride,a.min_event_coverage)
            if result is not None:
                parts.append(result)
                session_index[d.name.split('_')[0]].add(f.stem)
        if not parts:raise ValueError(f"All recordings shorter than window size in {d}")
        out=output_dir/f"window{a.window_size}_{d.name}.npz"
        if out.exists() and not a.overwrite:
            raise FileExistsError(f"Window file already exists: {out} (use --overwrite or new dataset folder)")
        merged={k:np.concatenate([part[k] for part in parts]) for k in parts[0]}
        if not np.isfinite(merged["x"]).all():raise ValueError(f"Nonfinite windows in {d}")
        with out.open("wb") as f:np.savez_compressed(f,**merged)
        ids=merged["y_command10"]
        # Lightweight dataset-difficulty diagnostic: an unsophisticated six-channel
        # energy-prototype rule. Prototypes are fitted from TRAIN ONLY, never test.
        # This is a QC probe, not a paper-grade baseline or model comparison.
        raw=merged["x"][:,:,:,0]
        upper=np.quantile(raw,.85,axis=1)
        lower=np.quantile(raw,.20,axis=1)
        score=(.65*upper+.35*(upper-lower)).astype(np.float32)
        score=score/(np.linalg.norm(score,axis=1,keepdims=True)+1e-7)
        probe_by_group[d.name]=(score,ids.copy())
        counts=Counter(ids.tolist())
        row=dict(group=d.name,sessions=len(parts),windows=len(ids),
                 event_windows=int(np.count_nonzero(ids>=0)),
                 event_fraction=round(float((ids>=0).mean()),5),
                 mean_quality=round(float(merged["target_quality"].mean()),5),
                 file=str(out.relative_to(root)),
                 **{f"class_{j}":counts.get(j,0) for j in list(range(10))+[-1]})
        report.append(row)
        elapsed=time.perf_counter()-start;eta=elapsed/i*(len(folders)-i)
        finish=(datetime.now()+timedelta(seconds=eta)).strftime("%Y-%m-%d %H:%M")
        print(f"[{i}/{len(folders)}] {d.name}: {len(ids)} windows, event={row['event_fraction']:.2%}, quality={row['mean_quality']:.3f} | ETA={format_eta(eta)} | finish≈{finish}",flush=True)
    assert not(session_index["train"] & session_index["validation"])
    assert not(session_index["train"] & session_index["test"])
    assert not(session_index["validation"] & session_index["test"])
    print("Split isolation check: PASS (train / validation / test session IDs disjoint)",flush=True)
    # Compare each condition to class prototypes constructed from training only.
    train_probe=probe_by_group.get("train_generatorA_mixed")
    prototype_rows=[]
    if train_probe is not None:
        feats_train,labels_train=train_probe
        valid=(labels_train>=0)
        if len(np.unique(labels_train[valid]))==10:
            proto=np.stack([feats_train[labels_train==j].mean(axis=0) for j in range(10)])
            proto=proto/(np.linalg.norm(proto,axis=1,keepdims=True)+1e-7)
            for group,(feat,truth) in sorted(probe_by_group.items()):
                keep=(truth>=0)
                pred=np.argmax(feat[keep]@proto.T,axis=1)
                acc=float(np.mean(pred==truth[keep]))
                prototype_rows.append(dict(group=group,event_windows=int(np.sum(keep)),
                                           simple_energy_prototype_accuracy=round(acc,5)))
            write_csv(root/"qc_simple_energy_probe.csv",prototype_rows)
            print("QC heuristic (train-only energy prototypes; NOT trained neural model):",flush=True)
            for r in prototype_rows:
                print(f"  {r['group']:<31s} accuracy={r['simple_energy_prototype_accuracy']:.2%}",flush=True)
    write_csv(root/"qc_window_summary.csv",report)
    qc=dict(kind="entirely synthetic",config=conf,window_size=a.window_size,stride=a.stride,
            min_event_coverage=a.min_event_coverage,split_isolation_pass=True,
            group_summary=report,simple_energy_probe=prototype_rows,
            probe_note="Train-only class energy prototypes for difficulty diagnostic. Never fit/tune on test. Do not cite as a full baseline.",
            feature_names=["raw_fixed_scale","causal_ema","first_difference"],
            input_tensor="x [N,W,6,3]",label_tensor="y_command10 [N] (-1 for idle/transition)",
            quality_target="quality_seq [N,W,6], target_quality [N,6]; never concatenate to x",
            nuisance_target="nuisance_seq [N,W,6], target_nuisance_fraction [N,6]; oracle training labels only",
            leakage_prevention="Split source sessions before windowing; apply EMA to each continuous session; no test-derived normalization",
            evaluation_warning="Same test latent event schedules shared across A/B conditions intentionally for paired comparisons; not independent test datasets")
    (root/"qc_report.json").write_text(json.dumps(qc,ensure_ascii=False,indent=2),encoding="utf-8")
    if not a.no_plots:
        plots(root,list(files_by_group),conf,report,a.window_size,a.stride)
    print(f"Finished in {format_eta(time.perf_counter()-start)}. QC: {root/'qc_window_summary.csv'}",flush=True)


if __name__ == "__main__":
    main()
