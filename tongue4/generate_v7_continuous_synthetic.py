#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""V7: session-wise continuous, entirely SYNTHETIC six-channel pressure signals.

No human subjects, physical tongue sensors, hardware calibration or physical validation.
Outputs continuous recordings first; window extraction is a SEPARATE script.

Usage: python generate_v7_continuous_synthetic.py --out-dir dataset_v7_continuous_synthetic
       python generate_v7_continuous_synthetic.py --out-dir dataset_v7_smoke --train-sessions 3 --val-sessions 2 --test-sessions 2 --events-per-session 12
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

FS = 50
C = 6
COMMAND_NAMES = ["forward", "backward", "left", "right", "stop", "switch",
                 "forward_left", "forward_right", "backward_left", "backward_right"]
COMMAND_CHANNELS = [(0,), (1,), (2,), (3,), (4,), (5,),
                    (0, 2), (0, 3), (1, 2), (1, 3)]
TEST_CONDITIONS = [
    ("A_nominal", "A", "nominal"), ("A_low", "A", "low"),
    ("A_medium", "A", "medium"), ("A_high", "A", "high"),
    ("A_ood", "A", "ood"), ("B_nominal", "B", "nominal"),
    ("B_medium", "B", "medium"), ("B_high", "B", "high"),
    ("B_ood", "B", "ood")
]
LEVELS = {
    "nominal": dict(noise=.007, drift=.006, fault_rate=.007, burst_rate=.006, gain=.025, cross=.007),
    "low":     dict(noise=.024, drift=.017, fault_rate=.020, burst_rate=.018, gain=.065, cross=.018),
    "medium":  dict(noise=.045, drift=.035, fault_rate=.045, burst_rate=.045, gain=.110, cross=.035),
    "high":    dict(noise=.085, drift=.065, fault_rate=.090, burst_rate=.085, gain=.180, cross=.070),
    "ood":     dict(noise=.110, drift=.090, fault_rate=.110, burst_rate=.115, gain=.240, cross=.100),
}


def parse_args():
    p = argparse.ArgumentParser(description="Generate V7 continuous synthetic tongue-pressure recordings")
    p.add_argument("--out-dir", type=Path, default=Path("dataset_v7_continuous_synthetic"))
    p.add_argument("--train-sessions", type=int, default=80)
    p.add_argument("--val-sessions", type=int, default=16)
    p.add_argument("--test-sessions", type=int, default=16)
    p.add_argument("--events-per-session", type=int, default=30)
    p.add_argument("--sampling-rate", type=int, default=FS)
    p.add_argument("--seed", type=int, default=20261009)
    p.add_argument("--nuisance-rate", type=float, default=1.1,
                   help="Mean class-independent brief unwanted-contact events per command; 0 disables")
    p.add_argument("--overwrite", action="store_true", help="Overwrite existing session files")
    return p.parse_args()


def fmt(sec):
    if sec is None: return "--:--:--"
    m, s = divmod(int(max(0, sec)), 60)
    h, m = divmod(m, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def make_schedule(rng, n_events, fs):
    # Class order is balanced per session, independently permuted.
    labels = np.tile(np.arange(10), int(math.ceil(n_events / 10)))[:n_events]
    rng.shuffle(labels)
    cursor = int(fs * rng.uniform(1.0, 1.8))
    events = []
    for event_idx, cls in enumerate(labels):
        idle = rng.uniform(.30, 1.7)
        cursor += int(round(fs * idle))
        duration = int(round(fs * rng.uniform(2.0, 4.1)))
        amplitude = float(rng.uniform(.30, 1.00))
        if cls in (4, 5):
            pressure_amplitude = float(rng.uniform(.48, .90))
            control_intensity = 0.0
        else:
            pressure_amplitude = amplitude
            control_intensity = amplitude  # latent intended intensity, NOT physical wheelchair speed
        events.append(dict(event_id=event_idx, command=int(cls), start=cursor,
                           end=cursor + duration, pressure_amplitude=pressure_amplitude,
                           control_intensity=control_intensity))
        cursor += duration
    n = cursor + int(round(fs * rng.uniform(.9, 1.4)))
    return events, n


def envelope(length, fs, rng, generator):
    # Continuous within-event envelope with independently sampled attack/release.
    z = np.arange(length, dtype=np.float32) / max(length - 1, 1)
    attack = rng.uniform(.13, .30)
    release = rng.uniform(.16, .34)
    if generator == "A":
        rise = np.clip(z / attack, 0, 1)
        fall = np.clip((1 - z) / release, 0, 1)
        env = (rise * rise * (3 - 2 * rise)) * (fall * fall * (3 - 2 * fall))
        env *= 1 + rng.uniform(.015, .08) * np.sin(z * rng.uniform(4.0, 7.5) * np.pi)
    else:
        # Alternative synthetic waveform family: smooth asymmetric multi-stage contraction.
        rise = 1 - np.exp(-np.maximum(z, 0) / max(attack * .33, .02))
        fall = 1 / (1 + np.exp(np.clip((z - (1 - release)) / .052, -40, 40)))
        env = rise * fall
        env *= 1 + rng.uniform(.04, .15) * np.sin(z * rng.uniform(3, 6) * np.pi + .5)
    return np.clip(env, 0, 1.12).astype(np.float32)


def make_clean_record(events, n, fs, generator, seed, nuisance_rate):
    rng = np.random.default_rng(seed)
    latent = np.zeros((n, C), np.float32)
    y = np.full(n, -1, np.int16)  # -1 means idle/unlabelled; not an 11th command class
    ids = np.full(n, -1, np.int32)
    intensity = np.zeros(n, np.float32)
    active_mask = np.zeros((n, C), np.uint8)
    phase = np.full(n, -1, np.float32)
    nuisance_mask = np.zeros((n,C), np.uint8)
    for ev in events:
        st, en, cls = ev["start"], ev["end"], ev["command"]
        env = envelope(en-st, fs, rng, generator)
        chans = COMMAND_CHANNELS[cls]
        for i, ch in enumerate(chans):
            # Asynchronous activations in diagonal commands, with channel-specific lag.
            delay = int(rng.integers(0, int(.20 * fs) + 1)) if len(chans) == 2 else 0
            a = min(n, st + delay)
            b = min(n, en)
            if b > a:
                weight = rng.uniform(.75, 1.10)
                latent[a:b, ch] += ev["pressure_amplitude"] * weight * env[:b-a]
        y[st:en] = cls
        ids[st:en] = ev["event_id"]
        intensity[st:en] = ev["control_intensity"]
        active_mask[st:en, list(chans)] = 1
        phase[st:en] = np.linspace(0, 1, en-st, dtype=np.float32)
    # Class-independent nuisance touches. They may overlap the intended pressure
    # event or occur during idle. These short contacts are NOT command labels.
    # This predeclared generative factor makes a long-range event decision distinct
    # from simply selecting whichever channel has the greatest instantaneous energy.
    # Importantly, it is applied identically to all models and cannot depend on
    # the downstream classifier, loss or evaluation results.
    if nuisance_rate > 0:
        for ev in events:
            true_set=set(COMMAND_CHANNELS[ev["command"]])
            candidates=[ch for ch in range(C) if ch not in true_set]
            for _ in range(int(rng.poisson(nuisance_rate))):
                ch=int(rng.choice(candidates))
                st=int(np.clip(rng.integers(ev["start"]-int(fs*.4),ev["end"]+int(fs*.22)),0,n-1))
                length=int(rng.integers(max(5,int(fs*.25)),max(7,int(fs*1.35))))
                en=min(n,st+length)
                shape=envelope(en-st,fs,rng,generator)
                amp=ev["pressure_amplitude"]*rng.uniform(.35,1.20)
                latent[st:en,ch]+=amp*shape
                nuisance_mask[st:en,ch]=1
        # Unintended contacts can also appear in idle periods.
        for _ in range(int(rng.poisson(nuisance_rate*len(events)*.2))):
            ch=int(rng.integers(C))
            st=int(rng.integers(0,n))
            if y[st]>=0:continue
            en=min(n,st+int(rng.integers(max(5,fs//4),max(6,fs))))
            amp=rng.uniform(.18,.64)
            latent[st:en,ch]+=amp*envelope(en-st,fs,rng,generator)
            nuisance_mask[st:en,ch]=1

    # Simulated sensor response: inter-channel mixing, gain/nonlinearity, smooth response.
    mixing = np.eye(C, dtype=np.float32)
    off = rng.uniform(.010, .050 if generator == "A" else .078, (C, C)).astype(np.float32)
    mixing += off * (1 - np.eye(C, dtype=np.float32))
    signal = latent @ mixing
    gains = rng.lognormal(0, .10 if generator == "A" else .16, C).astype(np.float32)
    gamma = rng.uniform(.85, 1.20, C) if generator == "A" else rng.uniform(.75, 1.30, C)
    signal = np.power(np.maximum(signal * gains, 0), gamma).astype(np.float32)
    # Stateful IIR response (applied on continuous recording, not per window).
    tau = rng.uniform(.045, .14 if generator == "A" else .23, C)
    alpha = np.clip(1 - np.exp(-1 / (fs * tau)), .005, .9)
    filt = np.zeros_like(signal)
    state = np.zeros(C, np.float32)
    for t in range(n):
        state += alpha * (signal[t] - state)
        filt[t] = state
    filt = 1.12 * np.tanh(filt / 1.12)
    bias = rng.uniform(.008, .045, C)
    filt = filt + bias
    return filt.astype(np.float32), y, ids, intensity, active_mask, phase, nuisance_mask


def _ar_noise(rng, n, rho):
    source = rng.standard_normal(n).astype(np.float32)
    out = np.empty(n, np.float32)
    x = 0.0
    alpha = math.sqrt(1 - rho*rho)
    for t in range(n):
        x = rho * x + alpha * float(source[t])
        out[t] = x
    return out


def corrupt(clean, level, generator, seed, fs):
    p = LEVELS[level]
    rng = np.random.default_rng(seed)
    n = len(clean)
    x = clean.copy()
    severity = np.zeros_like(clean)

    # Smooth slow drift and non-white noise; common mode is also present.
    drift = np.stack([_ar_noise(rng, n, .997) for _ in range(C)], axis=1)
    x += p["drift"] * drift
    severity += np.abs(p["drift"] * drift) * 1.9

    noise = np.stack([_ar_noise(rng, n, .72) for _ in range(C)], axis=1)
    shared = _ar_noise(rng, n, .86)[:, None]
    disturbance = p["noise"] * (0.82*noise + 0.34*shared)
    x += disturbance
    severity += np.abs(disturbance) * 1.8

    # Session-/slow-time-dependent gain distortion independent of the command label.
    scale = rng.normal(0.0, p["gain"], (1, C)).astype(np.float32)
    wobble = .35 * p["gain"] * np.sin(np.arange(n)[:, None] * (2*np.pi/(fs*18)) + rng.uniform(-2, 2, (1,C)))
    scale = scale + wobble
    x = x * (1 + scale)
    severity += np.abs(scale) * (.55 + .45*np.clip(clean, 0, 1.25))

    # Cross-talk perturbation independent of whether a pair is a target command.
    leakage = rng.uniform(0, p["cross"], (C, C)).astype(np.float32)
    leakage *= 1 - np.eye(C, dtype=np.float32)
    spilled = clean @ leakage
    x += spilled
    severity += np.abs(spilled) * 1.0

    # Sporadic channel attenuation / dropout and additive bursts.
    expected_faults = max(0.0, p["fault_rate"] * n / (fs * 4.0))
    for _ in range(int(rng.poisson(expected_faults))):
        ch = int(rng.integers(C))
        st = int(rng.integers(n))
        en = min(n, st + int(rng.integers(max(3, fs//6), max(4, fs*2))))
        factor = float(rng.uniform(.0, .55))
        x[st:en, ch] *= factor
        severity[st:en, ch] += (1 - factor)*.85

    expected_bursts = max(0.0, p["burst_rate"] * n / (fs * 3.5))
    for _ in range(int(rng.poisson(expected_bursts))):
        ch = int(rng.integers(C))
        st = int(rng.integers(n))
        length = int(rng.integers(max(3, fs//10), max(4, fs//2)))
        en = min(n, st + length)
        impact = rng.uniform(.13, .47) * (1 if rng.random() > .3 else -1)
        bump = impact * np.hanning(max(2, 2*(en-st)))[:en-st]
        x[st:en, ch] += bump
        severity[st:en, ch] += .6 + np.abs(bump)*1.5

    if level == "ood":
        # Held-out distortion family: channel-wise delayed echoes and ADC-style clipping.
        for ch in range(C):
            if rng.random() < .7:
                lag = int(rng.integers(max(2, fs//10), max(3, fs//2)))
                strength = float(rng.uniform(.12, .32))
                echo = np.zeros(n, np.float32)
                echo[lag:] = clean[:-lag, ch] * strength
                x[:, ch] += echo
                severity[:, ch] += np.abs(echo) * 1.3
        ceiling = rng.uniform(.64, 1.00, C)
        clipped = np.minimum(x, ceiling)
        severity += np.abs(x-clipped) * 2
        x = clipped

    # Values are synthetic sensor-scale values, not calibrated physical pressure units.
    x = np.clip(x, 0, 1.25).astype(np.float32)
    # An *injected-artifact* target for auxiliary supervision, not a measured sensor metric.
    q = np.exp(-2.1*np.maximum(severity, 0)).clip(.03, 1).astype(np.float32)
    return x, q


def generate_one(outpath, events, n, fs, generator, level, seed_clean, seed_noise, meta, nuisance_rate):
    clean, y, eid, intensity, mask, phase, nuisance = make_clean_record(events, n, fs, generator, seed_clean, nuisance_rate)
    x, q = corrupt(clean, level, generator, seed_noise, fs)
    outpath.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(outpath, x_raw=x, y_frame=y, event_id=eid,
                        control_intensity_frame=intensity, mask_frame=mask,
                        quality_frame=q, phase_frame=phase, nuisance_frame=nuisance, fs=np.int32(fs),
                        session_id=np.array(meta["session_id"]))
    return dict(**meta, n_samples=n, n_events=len(events), duration_seconds=round(n/fs, 3),
                file=str(outpath.relative_to(outpath.parents[2])),
                mean_quality=round(float(q.mean()), 5),
                fraction_active=round(float((y >= 0).mean()), 5))


def write_csv(path, rows):
    if not rows: return
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wr.writeheader(); wr.writerows(rows)


def main():
    a = parse_args()
    if a.nuisance_rate < 0:raise ValueError("nuisance-rate must be nonnegative")
    if a.sampling_rate <= 0 or min(a.train_sessions, a.val_sessions, a.test_sessions, a.events_per_session) < 1:
        raise ValueError("Sampling rate, sessions and events must be positive")
    root = a.out_dir
    root.mkdir(parents=True, exist_ok=True)
    config = dict(kind="entirely synthetic simulated multichannel pressure recordings",
                  limitation="NOT human measurements, NOT sensor calibration or physical validation",
                  fs=a.sampling_rate, channels=C, command_names=COMMAND_NAMES,
                  command_channels=[list(z) for z in COMMAND_CHANNELS],
                  idle_label=-1, control_intensity="simulator latent intensity; zero for stop/switch",
                  quality_frame="simulator-injected corruption proxy (supervision only)",
                  test_conditions=[z[0] for z in TEST_CONDITIONS],
                  train_mixture={"nominal": .30, "low": .40, "medium": .30},
                  split_policy="Independent session/event seeds across train/validation/test; paired latent test plans across conditions",
                  nuisance_rate=a.nuisance_rate,
                  nuisance_definition="Brief class-independent unintended contacts are NOT command labels; nuisance_frame is an oracle annotation",
                  seed=a.seed, train_sessions=a.train_sessions, val_sessions=a.val_sessions,
                  test_sessions=a.test_sessions, events_per_session=a.events_per_session)
    (root/"v7_config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    tasks=[]
    for split, n_sess in (("train", a.train_sessions), ("validation", a.val_sessions), ("test", a.test_sessions)):
        for k in range(n_sess):
            session_id = f"{split}_{k:04d}"
            planseed = a.seed + {"train":100000, "validation":200000, "test":300000}[split] + 97*k
            schedule, n = make_schedule(np.random.default_rng(planseed), a.events_per_session, a.sampling_rate)
            if split == "test":
                for cond, g, level in TEST_CONDITIONS:
                    tasks.append((split,session_id,schedule,n,cond,g,level,planseed, 900000 + planseed + sum(map(ord,cond))))
            else:
                prob = np.random.default_rng(planseed + 1).choice(["nominal", "low", "medium"], p=[.30,.40,.30])
                tasks.append((split,session_id,schedule,n,"A_mixed","A",str(prob),planseed, 800000+planseed))

    session_rows, event_rows = [], []
    start = time.perf_counter()
    print(f"V7 synthetic continuous records | {len(tasks)} session-condition records | output={root}", flush=True)
    for k, (split,sid,events,n,cond,g,lev,pseed,nseed) in enumerate(tasks,1):
        folder = f"{split}_generator{g}_{'mixed' if split!='test' else lev}"
        path = root/"sessions"/folder/f"session_{sid}.npz"
        if path.exists() and not a.overwrite:
            raise FileExistsError(f"File already exists: {path} (use --overwrite or new --out-dir)")
        meta = dict(split=split, session_id=sid, generator=g, condition=cond,
                    perturbation=lev, schedule_seed=pseed)
        session_rows.append(generate_one(path,events,n,a.sampling_rate,g,lev,pseed+71,nseed,meta,a.nuisance_rate))
        for ev in events:
            event_rows.append(dict(split=split, session_id=sid, generator=g, condition=cond,
                                   event_id=ev["event_id"], command=ev["command"],
                                   command_name=COMMAND_NAMES[ev["command"]],
                                   start_frame=ev["start"], end_frame=ev["end"],
                                   start_seconds=round(ev["start"]/a.sampling_rate,3),
                                   end_seconds=round(ev["end"]/a.sampling_rate,3),
                                   control_intensity=round(ev["control_intensity"],5)))
        elapsed=time.perf_counter()-start
        eta=elapsed/max(k,1)*(len(tasks)-k)
        if k==1 or k==len(tasks) or k%max(1,len(tasks)//25)==0:
            finish=(datetime.now()+timedelta(seconds=eta)).strftime('%Y-%m-%d %H:%M')
            print(f"[{k:4d}/{len(tasks)}] {folder}/{sid} | elapsed={fmt(elapsed)} | ETA={fmt(eta)} | finish≈{finish}",flush=True)
    write_csv(root/"session_manifest.csv",session_rows)
    write_csv(root/"event_annotations.csv",event_rows)
    print(f"Done in {fmt(time.perf_counter()-start)}. Run preprocess_v7_windows_qc.py next.", flush=True)


if __name__=="__main__":
    main()
