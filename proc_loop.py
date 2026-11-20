#!/usr/bin/env python3
# /home/tdrsvr/Desktop/slop/streams/proc_loop.py
# Local live loop on /proc, self-settling: the realized state h ticks after a
# fork is the label for that fork. No external teacher.
#   phase 1 (record) : fork -> settle, collect (x, y, 1.0) pairs
#   consolidate      : fork_generator.train on the collected pairs (absorbs hot state)
#   phase 2 (live)   : fork -> settle, scored against the phase-1 mean predictor
# Linux only (reads /proc). Needs numpy and fork_generator on the parent path.

import argparse
import json
import math
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import fork_generator as fg  # noqa: E402

K = 8        # raw channels per tick
D_FORK = 6   # targets: the first six channels, realized change over the horizon


# ---------------------------------------------------------------- readers
def read_stat():
    cpu, ctxt, procs = None, 0.0, 0.0
    with open("/proc/stat") as f:
        for line in f:
            if line.startswith("cpu "):
                cpu = [float(v) for v in line.split()[1:9]]   # user nice system idle iowait irq softirq steal
            elif line.startswith("ctxt "):
                ctxt = float(line.split()[1])
            elif line.startswith("procs_running "):
                procs = float(line.split()[1])
    return cpu, ctxt, procs


def read_mem():
    d = {}
    with open("/proc/meminfo") as f:
        for line in f:
            k, v = line.split(":", 1)
            d[k] = float(v.split()[0])
    return 1.0 - d["MemAvailable"] / d["MemTotal"]


def read_load():
    with open("/proc/loadavg") as f:
        return float(f.read().split()[0])


def read_net():
    rx = tx = 0.0
    with open("/proc/net/dev") as f:
        for line in list(f)[2:]:
            name, rest = line.split(":", 1)
            if name.strip() == "lo":
                continue
            v = rest.split()
            rx += float(v[0])
            tx += float(v[8])
    return rx, tx


class ProcSampler:
    """One raw vector per tick:
       cpu busy fraction, cpu iowait fraction, memory used fraction, load1,
       log1p(rx bytes/s), log1p(tx bytes/s), log1p(context switches/s), procs_running.
    Rates come from differencing cumulative counters across ticks."""

    def __init__(self):
        self.prev = None

    def read(self):
        now = time.monotonic()
        cpu, ctxt, procs = read_stat()
        rx, tx = read_net()
        cur = dict(t=now, cpu=cpu, ctxt=ctxt, rx=rx, tx=tx)
        mem, load = read_mem(), read_load()
        if self.prev is None:
            self.prev = cur
            return None
        p, self.prev = self.prev, cur
        dt = max(cur["t"] - p["t"], 1e-3)
        dc = [a - b for a, b in zip(cur["cpu"], p["cpu"])]
        tot = max(sum(dc), 1e-9)
        busy = 1.0 - (dc[3] + dc[4]) / tot          # idle = field 3, iowait = field 4
        iow = dc[4] / tot
        return np.array([
            busy, iow, mem, load,
            math.log1p(max(cur["rx"] - p["rx"], 0.0) / dt),
            math.log1p(max(cur["tx"] - p["tx"], 0.0) / dt),
            math.log1p(max(cur["ctxt"] - p["ctxt"], 0.0) / dt),
            procs,
        ], float)


# ---------------------------------------------------------------- features
class Features:
    """Standardize raw channels with slow EMA mean/var, then build
       x = [delta16 | hum8 | state8] with
         delta16 = [z_t - z_{t-1} | z_t - slow]
         hum8    = fast EMA of z   (the level, quickly tracked)
         state8  = slow EMA of z   (the level, slowly tracked)
       x is normalized to target_norm (same convention as screen_sensory.to_x)."""

    def __init__(self, a_fast=0.3, a_slow=0.02, target_norm=0.5):
        self.a_fast, self.a_slow, self.tn = a_fast, a_slow, target_norm
        self.mu = self.var = self.fast = self.slow = self.prev_z = None
        self.z = None

    def update(self, r):
        if self.mu is None:
            self.mu = r.copy()
            self.var = np.full(K, 1e-2)
            self.fast = np.zeros(K)
            self.slow = np.zeros(K)
        d = r - self.mu
        self.mu = self.mu + self.a_slow * d
        self.var = (1 - self.a_slow) * self.var + self.a_slow * d * d
        z = np.clip((r - self.mu) / np.sqrt(self.var + 1e-6), -5, 5)
        self.fast = (1 - self.a_fast) * self.fast + self.a_fast * z
        self.slow = (1 - self.a_slow) * self.slow + self.a_slow * z
        x = None
        if self.prev_z is not None:
            delta = np.concatenate([z - self.prev_z, z - self.slow])
            x = np.concatenate([delta, self.fast, self.slow])
            n = float(np.linalg.norm(x))
            if n > 1e-6 and self.tn > 0:
                x = x * (self.tn / n)
        self.prev_z = z
        self.z = z
        return x, z


# ---------------------------------------------------------------- loop
def run(args):
    sampler = ProcSampler()
    feats = Features(target_norm=args.target_norm)
    model = fg.ForkGenerator(seed=args.seed, ttl=1e9)
    zs = {}                  # tick -> standardized state
    pending = []             # open forks awaiting their label
    corpus = []              # phase-1 (x, y, w) pairs
    ysum = np.zeros(D_FORK)
    yn = 0
    sq_model = sq_base = 0.0
    n_live = 0
    trained = False
    total = args.T1 + args.T2
    t = 0
    shown = 0

    while t < total:
        time.sleep(args.dt)
        r = sampler.read()
        if r is None:
            continue
        x, z = feats.update(r)
        zs[t] = z
        phase = 1 if t < args.T1 else 2

        if phase == 2 and not trained:
            model.train(corpus, epochs=args.epochs, verbose=False)
            trained = True
            print(f"[consolidate] tick={t} pairs={len(corpus)} "
                  f"steps={model.step} absorbed={model.absorbed}")

        # settle every fork whose horizon has arrived
        for p in [q for q in pending if q["due"] <= t]:
            pending.remove(p)
            y = z[:D_FORK] - zs[p["t0"]][:D_FORK]
            if phase == 2:
                mu = np.clip(p["mu"], -fg.MU_CLIP, fg.MU_CLIP)
                base = ysum / max(yn, 1)
                sq_model += float(np.mean((mu - y) ** 2))
                sq_base += float(np.mean((base - y) ** 2))
                n_live += 1
                if shown < args.show:
                    print(json.dumps(dict(tick=t, mu_err=round(float(np.mean((mu - y) ** 2)), 5),
                                          base_err=round(float(np.mean((base - y) ** 2)), 5))))
                    shown += 1
            else:
                corpus.append((p["x"], y, 1.0))
                ysum += y
                yn += 1
            model.settle(p["tid"], y, "observed")
            model.update_pi()

        # fork at this tick (needs a feature vector: second valid sample onward)
        if x is not None:
            prop = model.fork(x)
            pending.append(dict(tid=prop.trajectory_id, due=t + args.h, t0=t,
                                x=x.copy(), mu=prop.latent_trajectory.copy()))
        model.advance(args.field_dt)
        t += 1

    skill = (1.0 - sq_model / max(sq_base, 1e-12)) if n_live else None
    report = dict(
        phase1_pairs=len(corpus),
        live_settled=n_live,
        live_skill_vs_phase1_mean=None if skill is None else round(skill, 4),
        ledger=model.ledger_check(),
        pi=model.pi.round(4).tolist(),
        bump_peak=round(float(model.bump.max()), 4),
        train_steps=model.step,
    )
    print("[report]", json.dumps(report, indent=2))
    return 0 if report["ledger"]["balanced"] else 1


def main():
    ap = argparse.ArgumentParser(prog="proc_loop")
    ap.add_argument("--dt", type=float, default=0.2, help="seconds per tick")
    ap.add_argument("--T1", type=int, default=100, help="record ticks")
    ap.add_argument("--T2", type=int, default=100, help="live ticks")
    ap.add_argument("--h", type=int, default=8, help="settlement horizon in ticks")
    ap.add_argument("--field-dt", type=float, default=0.25, help="field-time per tick")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--target-norm", type=float, default=0.5)
    ap.add_argument("--show", type=int, default=5, help="live lines to print")
    args = ap.parse_args()
    if args.T1 <= args.h:
        sys.exit("T1 must exceed the horizon h so phase 1 yields pairs")
    sys.exit(run(args))


if __name__ == "__main__":
    main()
