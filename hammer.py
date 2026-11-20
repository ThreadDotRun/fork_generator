#!/usr/bin/env python3
# /home/tdrsvr/Desktop/slop/streams/hammer.py
# Hammer: drive fork_generator with a synthetic firehose (regime shifts, bursts,
# malformed and late events), consolidating periodically and settling by the
# realized future. Then two probes:
#   tolerance : how far the proposal moves under input perturbation of relative size d
#   poisoning : whether a single non-finite value passes the harness

import copy
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.join(HERE, ".."), HERE]
import fork_generator as fg  # noqa: E402
from datafeeder import Feeder, Synthetic  # noqa: E402
from proc_loop import Features  # noqa: E402

H = 8          # settlement horizon, ticks
D_FORK = 6
RETRAIN = 400  # consolidate every N ticks
WIN = 800      # consolidation window: last WIN settled pairs
EPOCHS = 20

# (seconds, events/s, bot, new, log, cat, median |dlen|, big-change frac, n_wikis)
REGIMES = [
    (1200, 15, 0.05, 0.05, 0.05, 0.05, 200, 0.01, 5),   # baseline
    (800, 60, 0.70, 0.20, 0.10, 0.02, 50, 0.02, 2),     # bot storm
    (1200, 15, 0.05, 0.05, 0.05, 0.05, 200, 0.01, 5),   # recovery
    (60, 0.2, 0.10, 0.10, 0.10, 0.10, 100, 0.00, 1),    # quiet
]
BOUNDS = [1200, 2000, 3200, 3260]
NAMES = ["baseline", "storm", "recovery", "quiet"]


def regime_of(t):
    for i, b in enumerate(BOUNDS):
        if t < b:
            return i
    return len(BOUNDS) - 1


def hammer(seed=1):
    feeder = Feeder(Synthetic(REGIMES, seed=seed, p_malformed=0.02, p_late=0.01), tick_s=1.0)
    feats = Features(target_norm=0.5)
    model = fg.ForkGenerator(seed=0, ttl=50.0)
    pending, zs, corpus = [], {}, []
    blk = {n: dict(n=0, sq_m=0.0, sq_b=0.0, open_max=0, bump_peak=0.0, trains=0)
           for n in NAMES}
    t_start = time.time()
    t = 0
    for raw in feeder:
        x, z = feats.update(raw)
        zs[t] = z
        r = regime_of(t)
        b = blk[NAMES[r]]
        if x is not None and t > 0 and t % RETRAIN == 0 and len(corpus) >= 64:
            model.train(corpus[-WIN:], epochs=EPOCHS, verbose=False)
            b["trains"] += 1

        base = np.mean([c[1] for c in corpus[-WIN:]], axis=0) if corpus else np.zeros(D_FORK)
        for p in [q for q in pending if q["due"] <= t]:
            pending.remove(p)
            y = z[:D_FORK] - zs[p["t0"]][:D_FORK]
            bb = blk[NAMES[regime_of(p["t0"])]]
            mu = np.clip(p["mu"], -fg.MU_CLIP, fg.MU_CLIP)
            bb["sq_m"] += float(np.mean((mu - y) ** 2))
            bb["sq_b"] += float(np.mean((base - y) ** 2))
            bb["n"] += 1
            model.settle(p["tid"], y, "observed")
            corpus.append((p["x"], y, 1.0))
            model.update_pi()

        if x is not None:
            prop = model.fork(x)
            pending.append(dict(tid=prop.trajectory_id, due=t + H, t0=t,
                                x=x.copy(), mu=prop.latent_trajectory.copy()))
        model.advance(1.0)
        model.relax(1.0)
        led = model.ledger_check()
        b["open_max"] = max(b["open_max"], led["n_open"])
        b["bump_peak"] = max(b["bump_peak"], float(model.bump.max()))
        t += 1

    elapsed = time.time() - t_start
    return model, corpus, feeder, blk, t, elapsed


def report(model, feeder, blk, t, elapsed, corpus):
    print(f"[hammer] ticks={t}  wall={elapsed:.1f}s  rate={t / max(elapsed, 1e-9):.0f} ticks/s")
    print(f"[hammer] feeder: {feeder.counts()}")
    print(f"[hammer] settled pairs={len(corpus)}  trains={sum(v['trains'] for v in blk.values())}")
    print(f"{'block':<10}{'settled':>9}{'skill':>9}{'open_max':>10}{'bump_peak':>11}")
    for name in NAMES:
        v = blk[name]
        skill = (1.0 - v["sq_m"] / v["sq_b"]) if v["sq_b"] > 0 else float("nan")
        print(f"{name:<10}{v.get('n', 0):>9}{skill:>9.3f}{v['open_max']:>10}{v['bump_peak']:>11.3f}")
    led = model.ledger_check()
    print(f"[hammer] ledger: {led}")
    print(f"[hammer] pi={model.pi.round(4).tolist()}  logits={model.logits.round(3).tolist()}")
    print(f"[hammer] train steps={model.step}  absorbed={model.absorbed}")


def tolerance_probe(model, xs, rng):
    """Tolerance curve: perturb inputs by relative size d; a proposal stays in its
    semantic class if its drift is <= 10% of the unperturbed trajectory's norm."""
    mu0 = np.clip(model._predict_raw(xs, 0.0, True)[0], -fg.MU_CLIP, fg.MU_CLIP)
    n0 = np.linalg.norm(mu0, axis=1) + 1e-9
    print("[tolerance] d, median drift, fraction in class (drift <= 10%)")
    for d in [0.001, 0.01, 0.03, 0.1, 0.3, 1.0]:
        noise = rng.normal(size=xs.shape)
        noise /= np.linalg.norm(noise, axis=1, keepdims=True)
        xp = xs + d * np.linalg.norm(xs, axis=1, keepdims=True) * noise
        mup = np.clip(model._predict_raw(xp, 0.0, True)[0], -fg.MU_CLIP, fg.MU_CLIP)
        drift = np.linalg.norm(mup - mu0, axis=1) / n0
        print(f"  d={d:<6} median={np.median(drift):.4f}  in_class={np.mean(drift <= 0.1):.3f}")


def poison_probe(model, x, y):
    """Does a single non-finite value get through each entry point?"""
    out = {}
    pm = copy.deepcopy(model)
    xn = x.copy()
    xn[3] = np.nan
    try:
        prop = pm.fork(xn)
        out["fork accepts NaN x"] = True
        out["  -> proposal finite"] = bool(np.all(np.isfinite(prop.latent_trajectory)))
    except ValueError:
        out["fork accepts NaN x"] = False
    try:
        pm.ingest_corpus([(xn, y, 1.0)])
        out["ingest accepts NaN x"] = True
    except ValueError:
        out["ingest accepts NaN x"] = False
    pa = copy.deepcopy(model)
    p2 = pa.fork(x)
    yn = y.copy()
    yn[0] = np.nan
    try:
        pa.settle(p2.trajectory_id, yn, "observed")
        out["settle accepts NaN y"] = True
    except ValueError:
        out["settle accepts NaN y"] = False
    out["hot state finite after settle"] = bool(np.all(np.isfinite(pa.A)) and np.all(np.isfinite(pa.B)))
    print("[poison]", out)


def main():
    model, corpus, feeder, blk, t, elapsed = hammer()
    report(model, feeder, blk, t, elapsed, corpus)
    rng = np.random.default_rng(7)
    xs = np.stack([c[0] for c in corpus[-200:]])
    tolerance_probe(model, xs, rng)
    poison_probe(model, xs[0], corpus[-1][1])


if __name__ == "__main__":
    main()
