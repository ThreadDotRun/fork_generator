# /home/tdrsvr/Desktop/slop/fork_generator.py
# fork_generator.py  (v10)
# Fork Generator, Executive Organelle 1. numpy only.
#
# Clocks
#   hot    : fork / settle / observe / advance / relax / update_pi
#   consol : absorb, train (SGD on features and heads)
#
# Hot state per provenance class p in {observed, inferred, hallucinated}:
#   A_p, B_p   forgetting-kernel statistics of residual evidence
#   M_p        support mass (decayed sum of w): gates the pull, resists decay
#   bump_p     dissonance bump (transient extra decay, decays on field-time)
#   D_p        residual readout correction on [R(phase) z2 ; 1]
# Settlement statistics per class (antagonism):
#   SN_p, SA_p, SNA_p   decayed counts: settled, agreement sum, novel-and-agreed
#
# Relaxation (per class, on relax(dt)):
#   pull  = kappa_hot * dt * M/(M + s0)                  (no pull without support)
#   decay = min((mu0/(1 + gamma M) + bump) * dt, 0.9)    (rail)
#   D <- D + pull (A (B + beta I)^-1 - D) - decay D
#   Stable when kappa_hot*dt < 1 (guarded) and decay <= 0.9.
#
# Dissonance: on observe, z = mean(e^2)/unc is the surprise of the survivor against
#   the current hot readout; bump_p += diss_gain * w * tanh(max(z - 1, 0)).
#   tanh bounds each survivor's increment to diss_gain * w; bump_max caps the total.
#   Valence (v10): dissonance is pain. Pain is a hold signal for persistent
#   surprise and an accelerate signal for acute surprise. The two-timescale split
#   (acute/tonic) is not yet implemented; until it is, bump_p carries the tonic
#   role (persistence is what gem_gate reads). This is a declared bake-in.
#
# Ledger: open forks keyed by trajectory id, expiring after ttl in field-time.
#   n_arr = n_folded + n_discarded + n_open, exact (n_discarded includes expiry
#   and hallucinated-for-shit discard; the three discard paths are stated below).
#
# Antagonism: u_p = a_p + eta * n_p * (1 - pi_p);  pi = softmax(logits)
#   logits_p <- logits_p + kappa_pi (u_p - ubar)   (v7: logit space; was multiplicative)
#   a_p = mean agreement of settled class-p survivors; n_p = rate of novel-and-agreed.
#   The (1 - pi_p) rarity term is the two-class toy's form, carried over unchanged. For
#   three classes it is not zero-sum, so the two-class analytic threshold does not transfer.
#
# Breath phase (RoPE-style, header-only wire): with phase_on, pairs of hidden features
#   are rotated by angle phase * omega_i on the HOT path. R(a)^T R(b) = R(b - a), so
#   resonance depends only on relative phase. Conflict: a phase-dependent residual cannot
#   be folded into a phase-free base, so absorb() refuses while phase_on and D != 0.
#
# Bake-ins (DEFAULTS) are fixed constants that define the paths. Nothing here learns them.
#
# Rejection boundary (v8): non-finite x or y is rejected at fork, infer, observe,
#   settle, and ingest_corpus. settle validates BEFORE it consumes the ledger record,
#   so a rejected outcome leaves the fork open and the ledger balanced. A provenance
#   header naming a class not in PROVENANCE is rejected at fork.
#
# Settlement (v10): w = max(agree, surprise). Dissent is never quashed for observed
#   and inferred outcomes. The hallucinated class is subject to the declared valence:
#   persistent surprise is a gem (folded at full weight), one-off surprise is for-shit
#   (discarded at settlement). The valence is a bake-in (DEFAULTS["valence"]), stated
#   not derived. Discards come from three paths: expiry (field-time seizes), malformed
#   input (feeder's verifier), and hallucinated-for-shit (this valence). The first two
#   are structural; the third is the declared valence doing its job.
#
# Honored class (v10): fork() stores the class the fork was *asked to produce* on the
#   ledger record (from conditioning_tags['provenance'], defaulting to the fork's own
#   Proposal.provenance). settle() reads the honored class from the record; a caller
#   may pass an explicit override for tests and mislabeling probes, but there is no
#   silent "observed" default. The honored class is the sense-2 class (the accumulator
#   the outcome folds into); Proposal.provenance remains the sense-1 self-description.
#
# Consolidation loss (unchanged from v5, gradient-checked):
#   L = (1/B) sum_b w_b [ (1/F) sum_k 0.5(e^2 exp(-lv) + lv) + 0.5(e_s exp(lu) - lu) ]
#   e_s is a stop-gradient target. Rail veto is one-sided straight-through.
#
# Pickle note: run as a script, the class is also reachable as fork_generator.ForkGenerator.
# load() aliases __main__.ForkGenerator while unpickling.

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import pickle
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

if __name__ == "__main__":
    sys.modules.setdefault("fork_generator", sys.modules["__main__"])

# ---- numerical rails ----
MU_CLIP = 10.0
LV_MIN, LV_MAX = -10.0, 4.0
ERR_CLIP = 1e3
GRAD_CLIP = 1e3
DECAY_MAX = 0.9
LOGIT_CLIP = 30.0      # numerical rail on mixing logits

PROVENANCE = ("observed", "inferred", "hallucinated")
N_CLASS = len(PROVENANCE)

# ---- bake-ins: fixed, not learned ----
# Valence (declared, not derived): dissonance is pain. Pain is a hold signal for
# persistent surprise and an accelerate signal for acute surprise. The organism is
# born with this valence; it is not discovered from the firehose. The two-timescale
# split (acute/tonic) is not yet implemented; until it is, `bump` carries the tonic
# role (persistence is what the gem check reads).
DEFAULTS = dict(
    kappa=0.05,          # consolidation step size
    kappa_hot=0.5,       # hot pull rate (times dt; must keep kappa_hot*dt < 1)
    lam=0.01,            # evidence forgetting rate
    beta=1.0,            # ridge strength toward base
    s0=5.0,              # support half-saturation for the pull gate
    mu0=0.01,            # passive state decay with no support
    gamma=0.1,           # support resistance to decay
    lam_b=0.2,           # dissonance bump decay rate
    diss_gain=1.0,       # dissonance bump gain
    bump_max=1.0,        # cap on the total dissonance bump (rail; v7)
    eta=2.0,             # antagonism bonus
    kappa_pi=0.5,        # replicator rate
    tau_keep=0.2,        # agreement to count as kept (antagonism bookkeeping only)
    nov_thresh=0.5,      # novelty threshold for n_p
    ttl=50.0,            # open-fork lifetime in field-time
    phase_on=False,      # breath rotation on the hot path
    phase_base=10000.0,  # RoPE base frequency
    # Gem valence (hallucinated class only). Persistent surprise is a hold; one-off
    # surprise is for-shit and releases into discard. Thresholds are seeds for the
    # running aggregate baseline, not eternal constants.
    gem_bump=0.5,        # bump at or above this counts as persistent surprise
    gem_support=2.0,     # support mass at or above this counts as kept-settling
    valence="tonic_hold",  # declared valence; swapping this swaps the discard rule
)


# ---- wires ----
@dataclass(frozen=True)
class Token:
    origin_id: str
    field_time: float
    provenance: str
    tags: Tuple[str, ...]
    scale: float
    payload: object


@dataclass
class Proposal:
    trajectory_id: str
    proposal_weight: float
    latent_trajectory: np.ndarray
    conditioning_tags: Dict[str, float]
    uncertainty: float                   # exp(-lu): expected squared error of mu
    origin_id: str = "fork_generator"
    provenance: str = "inferred"         # sense-1 self-description (not the honored class)


# ---- helpers ----
def _one_sided(raw, grad, lo, hi):
    push_out_hi = (raw >= hi) & (grad < 0)
    push_out_lo = (raw <= lo) & (grad > 0)
    return np.where(push_out_hi | push_out_lo, 0.0, grad)


def _pearson(a, b):
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    if a.std() < 1e-12 or b.std() < 1e-12:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def _stack(corpus):
    X = np.stack([c[0] for c in corpus])
    Y = np.stack([c[1] for c in corpus])
    W = np.array([c[2] for c in corpus], dtype=float)
    return X, Y, W


# ---- model ----
class ForkGenerator:
    def __init__(self, d_delta=16, d_hum=8, d_state=8, d_fork=6, hidden=128,
                 seed=0, **overrides):
        unknown = set(overrides) - set(DEFAULTS)
        if unknown:
            raise ValueError(f"unknown config keys: {sorted(unknown)}")
        if hidden % 2:
            raise ValueError("hidden must be even: phase rotation acts on feature pairs")
        cfg = dict(DEFAULTS)
        cfg.update(overrides)
        if cfg["beta"] <= 0:
            raise ValueError("beta must be > 0")
        if cfg["valence"] not in ("tonic_hold",):
            # Only one valence is implemented. Adding a second is a deliberate
            # change to the discard rule in settle(); the string is the switch.
            raise ValueError(f"valence {cfg['valence']!r} not implemented; "
                             f"only 'tonic_hold' is available")
        cfg.update(d_delta=d_delta, d_hum=d_hum, d_state=d_state, d_fork=d_fork,
                   hidden=hidden, seed=seed)
        self.cfg = cfg
        self.rng = np.random.default_rng(seed)
        d_in = d_delta + d_hum + d_state
        H, F, C = hidden, d_fork, N_CLASS

        def g(shape, s=0.1):
            return self.rng.normal(0.0, s, size=shape)

        # consolidation parameters
        self.W1, self.b1 = g((d_in, H)), np.zeros(H)
        self.W2, self.b2 = g((H, H)), np.zeros(H)
        self.Wm, self.bm = g((H, F)), np.zeros(F)
        self.Wv, self.bv = g((H, F)), np.zeros(F)
        self.Wu, self.bu = g((H, 1)), np.zeros(1)

        # hot state
        self.logits = np.zeros(C)                    # pi = softmax(logits)
        self.D = np.zeros((C, F, H + 1))
        self.A = np.zeros((C, F, H + 1))
        self.B = np.zeros((C, H + 1, H + 1))
        self.M = np.zeros(C)
        self.bump = np.zeros(C)
        self.SN = np.zeros(C)
        self.SA = np.zeros(C)
        self.SNA = np.zeros(C)

        # ledger
        self.ledger: Dict[str, dict] = {}
        self.n_arr = 0
        self.n_folded = 0
        self.n_discarded = 0
        self.n_expired = 0

        # bookkeeping
        self.n_obs = 0
        self.field_time = 0.0
        self.step = 0
        self.absorbed = 0
        self.history: List[float] = []
        self.head_diag: List[Dict[str, float]] = []
        self.train_n = 0

    # ---- mixing weights: pi = softmax(logits) ----
    @property
    def pi(self) -> np.ndarray:
        z = self.logits - self.logits.max()
        e = np.exp(z)
        return e / e.sum()

    @pi.setter
    def pi(self, v):
        v = np.clip(np.asarray(v, float), 1e-12, None)
        self.logits = np.log(v)

    # ---- spaces ----
    def param_space(self) -> Dict[str, np.ndarray]:
        """Consolidation space only."""
        return dict(W1=self.W1, b1=self.b1, W2=self.W2, b2=self.b2,
                    Wm=self.Wm, bm=self.bm, Wv=self.Wv, bv=self.bv,
                    Wu=self.Wu, bu=self.bu)

    # ---- features ----
    def _hidden(self, X):
        z1 = np.tanh(X @ self.W1 + self.b1)
        z2 = np.tanh(z1 @ self.W2 + self.b2)
        return z1, z2

    def _rotate(self, z2, phase):
        """Breath rotation of feature pairs. Header-only: phase is a scalar clock value.
        Identity at phase 0 or when phase_on is False."""
        if not self.cfg["phase_on"] or phase == 0.0:
            return z2
        H = z2.shape[-1]
        i = np.arange(H // 2)
        ang = float(phase) * self.cfg["phase_base"] ** (-2.0 * i / H)
        c, s = np.cos(ang), np.sin(ang)
        a = z2[..., 0::2]
        b = z2[..., 1::2]
        out = np.empty_like(z2)
        out[..., 0::2] = a * c - b * s
        out[..., 1::2] = a * s + b * c
        return out

    def _lift(self, z2, phase):
        zr = self._rotate(z2, phase)
        return np.hstack([zr, np.ones((zr.shape[0], 1))])

    def _mix(self):
        return np.tensordot(self.pi, self.D, axes=1)          # (F, H+1)

    def _predict_raw(self, X, phase=0.0, hot=True):
        _, z2 = self._hidden(X)
        mu_raw = z2 @ self.Wm + self.bm
        if hot:
            G = self._lift(z2, phase)
            mu_raw = mu_raw + G @ self._mix().T
        lv_raw = z2 @ self.Wv + self.bv
        lu_raw = z2 @ self.Wu + self.bu
        return mu_raw, lv_raw, lu_raw

    def _novelty(self, g):
        """Mixed unfamiliarity of g: sum_p pi_p (1 - rho_p), rho_p = g^T B_p g / g^T (B_p + beta I) g."""
        gg = float(g @ g)
        beta = self.cfg["beta"]
        pi = self.pi
        out = 0.0
        for p in range(N_CLASS):
            bg = self.B[p] @ g
            rho = float(g @ bg) / (float(g @ bg) + beta * gg)
            out += pi[p] * (1.0 - rho)
        return float(out)

    # ---- consolidation ----
    def _consolidate_step(self, X, Y, w):
        Bn, F = X.shape[0], self.cfg["d_fork"]
        z1, z2 = self._hidden(X)
        mu_raw = z2 @ self.Wm + self.bm
        lv_raw = z2 @ self.Wv + self.bv
        lu_raw = z2 @ self.Wu + self.bu
        mu = np.clip(mu_raw, -MU_CLIP, MU_CLIP)
        lv = np.clip(lv_raw, LV_MIN, LV_MAX)
        lu = np.clip(lu_raw, LV_MIN, LV_MAX)

        diff = mu - Y
        ok = (np.abs(diff) < ERR_CLIP).astype(float)
        err = np.clip(diff, -ERR_CLIP, ERR_CLIP)
        prec = np.exp(-lv)
        wc = w[:, None]
        e_s = (err ** 2).mean(axis=1, keepdims=True)

        loss_mu = 0.5 * (err ** 2 * prec + lv)
        loss_u = 0.5 * (e_s * np.exp(lu) - lu)
        loss = float((loss_mu * wc).mean() + (loss_u * wc).mean())

        d_mu = err * prec * ok * wc / (Bn * F)
        d_lv = 0.5 * (1.0 - err ** 2 * prec) * ok * wc / (Bn * F)
        d_lu = 0.5 * (e_s * np.exp(lu) - 1.0) * wc / Bn

        d_mu = _one_sided(mu_raw, d_mu, -MU_CLIP, MU_CLIP)
        d_lv = _one_sided(lv_raw, d_lv, LV_MIN, LV_MAX)
        d_lu = _one_sided(lu_raw, d_lu, LV_MIN, LV_MAX)
        d_mu = np.clip(d_mu, -GRAD_CLIP, GRAD_CLIP)
        d_lv = np.clip(d_lv, -GRAD_CLIP, GRAD_CLIP)
        d_lu = np.clip(d_lu, -GRAD_CLIP, GRAD_CLIP)

        dz2 = (d_mu @ self.Wm.T + d_lv @ self.Wv.T + d_lu @ self.Wu.T) * (1.0 - z2 ** 2)
        dz1 = (dz2 @ self.W2.T) * (1.0 - z1 ** 2)
        grads = dict(
            W1=X.T @ dz1, b1=dz1.sum(0),
            W2=z1.T @ dz2, b2=dz2.sum(0),
            Wm=z2.T @ d_mu, bm=d_mu.sum(0),
            Wv=z2.T @ d_lv, bv=d_lv.sum(0),
            Wu=z2.T @ d_lu, bu=d_lu.sum(0),
        )
        return loss, grads, dict(mu=mu, lv=lv, lu=lu)

    def inhabit(self, grads):
        for k, P in self.param_space().items():
            P -= self.cfg["kappa"] * grads[k]
        self.step += 1

    def absorb(self):
        """Hot -> consolidation write. Prediction-preserving for phase-free readout."""
        pending = (np.any(self.D != 0.0) or np.any(self.A != 0.0)
                   or np.any(self.B != 0.0) or np.any(self.M != 0.0))
        if self.cfg["phase_on"] and pending:
            raise ValueError("phase-dependent residual cannot be folded into a phase-free base; "
                             "run with phase_on=False or clear the hot state first")
        Dm = self._mix()
        H = self.cfg["hidden"]
        self.Wm = self.Wm + Dm[:, :H].T
        self.bm = self.bm + Dm[:, H]
        self.D[:] = 0.0
        self.A[:] = 0.0
        self.B[:] = 0.0
        self.M[:] = 0.0
        self.bump[:] = 0.0
        self.absorbed += 1

    def train(self, corpus, batch=32, epochs=200, tol=1e-5, patience=10,
              verbose=True, diagnose=False):
        if not corpus:
            raise ValueError("empty corpus")
        self.absorb()
        X, Y, W = _stack(corpus)
        n = len(X)
        self.train_n = n
        best, stall = float("inf"), 0
        for ep in range(epochs):
            idx = self.rng.permutation(n)
            ep_loss, nb = 0.0, 0
            for i in range(0, n, batch):
                bi = idx[i:i + batch]
                loss, grads, _ = self._consolidate_step(X[bi], Y[bi], W[bi])
                self.inhabit(grads)
                ep_loss += loss
                nb += 1
            ep_loss /= max(nb, 1)
            self.history.append(ep_loss)
            if diagnose:
                st = self._head_stats(X)
                st.update(ep=ep, loss=ep_loss)
                self.head_diag.append(st)
                if verbose:
                    print(f"[train] ep={ep:3d} loss={ep_loss:.6f} |mu|={st['mean_abs_mu']:.4f} "
                          f"unc={st['mean_unc']:.4f} mu_rail={st['mu_at_rail']:.2f} "
                          f"lv_rail={st['lv_at_rail']:.2f} lu_rail={st['lu_at_rail']:.2f}")
            elif verbose and (ep % 10 == 0 or ep == epochs - 1):
                print(f"[train] ep={ep:3d} loss={ep_loss:.6f} step={self.step}")
            if ep_loss < best - tol:
                best, stall = ep_loss, 0
            else:
                stall += 1
            if stall >= patience:
                if verbose:
                    print(f"[train] early stop at ep={ep} (best={best:.6f})")
                break
        return self.history

    def _head_stats(self, X):
        mu_raw, lv_raw, lu_raw = self._predict_raw(X, hot=False)
        mu = np.clip(mu_raw, -MU_CLIP, MU_CLIP)
        lv = np.clip(lv_raw, LV_MIN, LV_MAX)
        lu = np.clip(lu_raw, LV_MIN, LV_MAX)
        return dict(
            mean_abs_mu=float(np.abs(mu).mean()),
            mean_abs_lv=float(np.abs(lv).mean()),
            mean_unc=float(np.exp(-lu).mean()),
            mu_at_rail=float((np.abs(mu) >= MU_CLIP - 1e-9).mean()),
            lv_at_rail=float(((lv <= LV_MIN + 1e-9) | (lv >= LV_MAX - 1e-9)).mean()),
            lu_at_rail=float(((lu <= LV_MIN + 1e-9) | (lu >= LV_MAX - 1e-9)).mean()),
        )

    # ---- hot clock ----
    def advance(self, dt):
        if dt < 0:
            raise ValueError("dt must be >= 0")
        c = self.cfg
        f = np.exp(-c["lam"] * dt)
        self.A *= f
        self.B *= f
        self.M *= f
        self.SN *= f
        self.SA *= f
        self.SNA *= f
        self.bump *= np.exp(-c["lam_b"] * dt)
        self.field_time += dt
        dead = [t for t, r in self.ledger.items() if self.field_time - r["t0"] > c["ttl"]]
        for t in dead:
            del self.ledger[t]
        self.n_expired += len(dead)
        self.n_discarded += len(dead)

    def observe(self, x, y, w, provenance="observed", phase=0.0):
        """Fold one survivor into the hot statistics. Returns its surprise z.
        Dissonance is computed against the readout BEFORE this survivor is folded."""
        p = PROVENANCE.index(provenance)
        w = float(w)
        if not (np.isfinite(w) and w >= 0):
            raise ValueError("w must be finite and >= 0")
        x = np.asarray(x, float).reshape(1, -1)
        y = np.asarray(y, float).reshape(-1)
        if not (np.all(np.isfinite(x)) and np.all(np.isfinite(y))):
            raise ValueError("observe: non-finite x or y rejected")
        _, z2 = self._hidden(x)
        base = z2 @ self.Wm + self.bm
        lu_base = float((z2 @ self.Wu + self.bu)[0, 0])
        G = self._lift(z2, phase)
        mu_hot = base + G @ self._mix().T
        unc = float(np.exp(-np.clip(lu_base, LV_MIN, LV_MAX)))
        e = np.clip(mu_hot[0] - y, -ERR_CLIP, ERR_CLIP)
        z = float(np.mean(e ** 2) / unc)
        resid = y - base[0]
        g = G[0]
        self.A[p] += w * np.outer(resid, g)
        self.B[p] += w * np.outer(g, g)
        self.M[p] += w
        inc = self.cfg["diss_gain"] * w * np.tanh(max(z - 1.0, 0.0))
        self.bump[p] = min(self.bump[p] + inc, self.cfg["bump_max"])
        self.n_obs += 1
        return z

    def relax(self, dt=1.0):
        c = self.cfg
        h0 = c["kappa_hot"] * dt
        if not (0.0 <= h0 < 1.0):
            raise ValueError(f"kappa_hot*dt={h0:.3f} outside [0, 1): relax step unstable")
        ridge = np.eye(self.B.shape[1]) * c["beta"]
        for p in range(N_CLASS):
            M = float(self.M[p])
            gate = M / (M + c["s0"]) if (M + c["s0"]) > 0 else 0.0
            target = np.linalg.solve(self.B[p] + ridge, self.A[p].T).T
            pull = h0 * gate
            rate = c["mu0"] / (1.0 + c["gamma"] * M) + self.bump[p]
            decay = min(rate * dt, DECAY_MAX)
            self.D[p] += pull * (target - self.D[p]) - decay * self.D[p]

    def update_pi(self):
        """Gradient ascent on expected settlement payoff, in logit space:
        pi = softmax(logits), logits_p += kappa_pi * (u_p - ubar).
        Simplex holds by construction. Interior fixed points match the replicator's
        (u_p equal over the support); the discrete trajectories differ."""
        c = self.cfg
        if np.any(self.SN <= 0):
            return self.pi
        pi = self.pi
        a = self.SA / self.SN
        n = self.SNA / self.SN
        u = a + c["eta"] * n * (1.0 - pi)
        ubar = float((pi * u).sum())
        self.logits = np.clip(self.logits + c["kappa_pi"] * (u - ubar), -LOGIT_CLIP, LOGIT_CLIP)
        return self.pi

    # ---- ledger ----
    def fork(self, x, phase=0.0, conditioning_tags=None):
        """Emit a proposal and open a ledger record. Counts as an arrival.

        conditioning_tags may carry 'provenance' as a header: the class the fork
        was *asked to produce*. The header is honored — it becomes the class the
        settlement folds into, unless settle() is given an explicit override.
        No default is invented; if no header, the fork's own provenance
        ('inferred') is the honored class. A header naming a class not in
        PROVENANCE is rejected at this boundary."""
        x = np.asarray(x, float).reshape(1, -1)
        if not np.all(np.isfinite(x)):
            raise ValueError("fork: non-finite input rejected")
        tags = dict(conditioning_tags or {})
        honored = tags.get("provenance", Proposal.provenance)
        if honored not in PROVENANCE:
            raise ValueError(f"fork: provenance header {honored!r} not in {PROVENANCE}")
        mu_raw, lv_raw, lu_raw = self._predict_raw(x, phase, hot=True)
        mu = np.clip(mu_raw[0], -MU_CLIP, MU_CLIP)
        lv = np.clip(lv_raw[0], LV_MIN, LV_MAX)
        lu = float(np.clip(lu_raw[0, 0], LV_MIN, LV_MAX))
        unc = float(np.exp(-lu))
        _, z2 = self._hidden(x)
        novelty = self._novelty(self._lift(z2, phase)[0])
        tid = "fork-" + hashlib.sha256(
            x.tobytes() + f"|{self.n_arr}|{self.field_time:.9f}|{float(phase):.9f}".encode()
        ).hexdigest()[:12]
        self.ledger[tid] = dict(x=x[0].copy(), phase=float(phase), mu=mu.copy(),
                                unc=unc, novelty=novelty, t0=self.field_time,
                                provenance=honored)
        self.n_arr += 1
        return Proposal(trajectory_id=tid,
                        proposal_weight=float(np.exp(-lv.mean())),
                        latent_trajectory=mu,
                        conditioning_tags=tags,
                        uncertainty=unc)

    def settle(self, tid, y, provenance=None):
        """Join an open fork with its outcome. Consumes the record.

        The class the outcome folds into is the class the fork was *asked to
        produce* (the honored header from fork()), unless the caller passes an
        explicit override. No silent default: a caller that passes nothing gets
        the fork's honored class. Dissent is never quashed for observed and
        inferred outcomes. The hallucinated class is subject to the declared
        valence: persistent surprise is a gem (fold), one-off surprise is for-shit
        (discard).

        Note on the discard rule (v10): the gem check is trembling-only — it reads
        bump[p] and M[p]. It cannot tell a tourmaline (persistently wrong) from a
        diamond (persistently early). The two-timescale split and the cross-class
        drift tracker are the machinery that separates them; until they land, this
        rule treats tourmalines as gems. The limitation is stated, not hidden."""
        if tid not in self.ledger:
            raise KeyError(f"no open fork {tid} (expired or already settled)")
        y = np.asarray(y, float).reshape(-1)
        if not np.all(np.isfinite(y)):
            raise ValueError("settle: non-finite outcome rejected; fork stays open")
        rec = self.ledger.pop(tid)
        c = self.cfg
        p_name = provenance if provenance is not None else rec["provenance"]
        if p_name not in PROVENANCE:
            raise ValueError(f"settle: override provenance {p_name!r} not in {PROVENANCE}")
        p = PROVENANCE.index(p_name)
        e = np.clip(rec["mu"] - y, -ERR_CLIP, ERR_CLIP)
        z = float(np.mean(e ** 2) / rec["unc"])
        agree = float(np.exp(-0.5 * z))
        surprise = 1.0 - agree
        kept = agree >= c["tau_keep"]
        w = max(agree, surprise)          # v9: dissent carries full weight
        novel = rec["novelty"] > c["nov_thresh"]
        self.SN[p] += 1.0
        self.SA[p] += agree
        if novel and kept:
            self.SNA[p] += 1.0

        # Valence: only the hallucinated class can be discarded at settlement.
        # The gem check uses the state the code carries: persistent surprise
        # (bump at or above gem_bump) OR repeated settling (support mass at or
        # above gem_support). Trembling-only approximation; drift tracker pending.
        if p_name == "hallucinated" and c["valence"] == "tonic_hold":
            gem = (self.bump[p] >= c["gem_bump"]) or (self.M[p] >= c["gem_support"])
            if not gem:
                self.n_discarded += 1
                return dict(agree=agree, w=w, z=z, novel=novel,
                            folded=False, discarded=True, gem=False)

        folded = w > 0.0
        if folded:
            self.observe(rec["x"], y, w, p_name, phase=rec["phase"])
            self.n_folded += 1
        else:
            self.n_discarded += 1
        return dict(agree=agree, w=w, z=z, novel=novel,
                    folded=folded, discarded=False, gem=True)

    def ledger_check(self):
        lhs = self.n_arr
        rhs = self.n_folded + self.n_discarded + len(self.ledger)
        return dict(n_arr=lhs, n_folded=self.n_folded, n_discarded=self.n_discarded,
                    n_expired=self.n_expired, n_open=len(self.ledger), balanced=(lhs == rhs))

    # ---- inference ----
    def infer(self, x, phase=0.0, conditioning_tags: Optional[Dict[str, float]] = None) -> Proposal:
        """Read-only proposal. Opens no ledger record."""
        x = np.asarray(x, float).reshape(1, -1)
        if not np.all(np.isfinite(x)):
            raise ValueError("infer: non-finite input rejected")
        mu_raw, lv_raw, lu_raw = self._predict_raw(x, phase, hot=True)
        mu = np.clip(mu_raw[0], -MU_CLIP, MU_CLIP)
        lv = np.clip(lv_raw[0], LV_MIN, LV_MAX)
        lu = float(np.clip(lu_raw[0, 0], LV_MIN, LV_MAX))
        key = f"|{self.step}|{self.n_arr}|{self.field_time:.9f}|{float(phase):.9f}".encode()
        tid = "read-" + hashlib.sha256(x.tobytes() + key).hexdigest()[:12]
        return Proposal(trajectory_id=tid,
                        proposal_weight=float(np.exp(-lv.mean())),
                        latent_trajectory=mu,
                        conditioning_tags=dict(conditioning_tags or {}),
                        uncertainty=float(np.exp(-lu)))

    def evaluate(self, corpus):
        X, Y, _ = _stack(corpus)
        mu_raw, _, lu_raw = self._predict_raw(X, 0.0, hot=True)
        mu = np.clip(mu_raw, -MU_CLIP, MU_CLIP)
        lu = np.clip(lu_raw[:, 0], LV_MIN, LV_MAX)
        err = mu - Y
        e_s = (err ** 2).mean(axis=1)
        unc = np.exp(-lu)
        mse = float((err ** 2).mean())
        base = float(((Y - Y.mean(axis=0)) ** 2).mean())
        return dict(
            n=len(X), mse=mse, baseline_mse=base,
            skill=1.0 - mse / max(base, 1e-12),
            mean_unc=float(unc.mean()),
            calibration_ratio=float(unc.mean() / max(e_s.mean(), 1e-12)),
            unc_err_pearson=_pearson(unc, e_s),
        )

    def _discard_rate(self) -> float:
        total = self.n_folded + self.n_discarded
        return float(self.n_discarded) / total if total else 0.0

    def diagnostics(self) -> dict:
        if not self.history:
            return {}
        out = dict(initial_loss=self.history[0], final_loss=self.history[-1],
                   improvement=self.history[0] - self.history[-1],
                   steps=self.step, train_n=self.train_n, absorbed=self.absorbed,
                   n_obs=self.n_obs, field_time=self.field_time,
                   valence=self.cfg["valence"],
                   discard_rate=self._discard_rate(),
                   pi=self.pi.round(4).tolist(),
                   ledger=self.ledger_check())
        if self.head_diag:
            out["heads_last"] = {k: v for k, v in self.head_diag[-1].items()
                                 if k not in ("ep", "loss")}
        return out

    # ---- ingest (pure) ----
    def ingest_corpus(self, raw) -> List[Tuple[np.ndarray, np.ndarray, float]]:
        d_in = self.cfg["d_delta"] + self.cfg["d_hum"] + self.cfg["d_state"]
        F = self.cfg["d_fork"]
        out = []
        for i, item in enumerate(raw):
            if len(item) == 4:
                x, y, _legacy_lv, w = item
            elif len(item) == 3:
                x, y, w = item
            else:
                raise ValueError(f"item {i}: expected (x, y, w) or legacy (x, y, lv, w)")
            x = np.asarray(x, float).reshape(-1)
            y = np.asarray(y, float).reshape(-1)
            w = float(w)
            if x.shape != (d_in,):
                raise ValueError(f"item {i}: x shape {x.shape} != ({d_in},)")
            if not (np.all(np.isfinite(x)) and np.all(np.isfinite(y))):
                raise ValueError(f"item {i}: non-finite x or y rejected")
            if y.shape != (F,):
                raise ValueError(f"item {i}: y shape {y.shape} != ({F},)")
            if not (np.isfinite(w) and w >= 0):
                raise ValueError(f"item {i}: w must be finite and >= 0")
            out.append((x, y, w))
        return out

    # ---- persistence ----
    def save(self, path: str):
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: str) -> "ForkGenerator":
        import __main__
        had = hasattr(__main__, "ForkGenerator")
        prev = getattr(__main__, "ForkGenerator", None)
        setattr(__main__, "ForkGenerator", cls)
        try:
            with open(path, "rb") as f:
                obj = pickle.load(f)
        finally:
            if had:
                setattr(__main__, "ForkGenerator", prev)
            else:
                try:
                    delattr(__main__, "ForkGenerator")
                except AttributeError:
                    pass
        assert isinstance(obj, cls)
        return obj

    @staticmethod
    def save_corpus(path: str, corpus):
        with open(path, "wb") as f:
            pickle.dump([(np.asarray(x, float), np.asarray(y, float), float(w))
                         for (x, y, w) in corpus], f)

    @staticmethod
    def load_corpus(path: str):
        with open(path, "rb") as f:
            return pickle.load(f)


# ---- synthetic corpus ----
def synth_corpus(n=2048, d_delta=16, d_hum=8, d_state=8, d_fork=6, seed=1):
    rng = np.random.default_rng(seed)
    basis = rng.normal(size=(3, d_delta))
    Wd = rng.normal(size=(d_delta, d_fork)) * 0.3
    Wh = rng.normal(size=(d_hum, d_fork)) * 0.2
    corpus = []
    for _ in range(n):
        coef = rng.normal(size=(3, 1)) * 0.5
        delta = (coef.T @ basis)[0] + 0.05 * rng.normal(size=d_delta)
        hum = rng.uniform(0, 1, size=d_hum)
        state = rng.normal(size=d_state) * 0.2
        mean = (delta @ Wd + np.tanh(hum @ Wh)) * (1.0 - 0.5 * hum[0])
        sigma = 0.02 + 0.2 * hum[1]
        traj = mean + sigma * rng.normal(size=d_fork)
        weight = float(0.5 + 0.5 * rng.uniform())
        x = np.concatenate([delta, hum, state])
        corpus.append((x, traj, weight))
    return corpus


def make_td_dat(path="td.dat", n=256, seed=42,
                d_delta=16, d_hum=8, d_state=8, d_fork=6):
    corpus = synth_corpus(n=n, seed=seed, d_delta=d_delta, d_hum=d_hum,
                          d_state=d_state, d_fork=d_fork)
    ForkGenerator.save_corpus(path, corpus)
    return path, len(corpus)


# ---- CLI ----
def _cmd_train(args):
    raw = (ForkGenerator.load_corpus(args.corpus) if args.corpus
           else synth_corpus(n=args.n, seed=args.seed))
    src = f"loaded {args.corpus}" if args.corpus else "synthesized"
    print(f"[corpus] {src} {len(raw)} items")
    m = ForkGenerator(seed=args.seed)
    corpus = m.ingest_corpus(raw)
    order = np.random.default_rng(args.seed).permutation(len(corpus))
    n_hold = int(round(len(corpus) * args.holdout))
    hold = [corpus[i] for i in order[:n_hold]]
    tr = [corpus[i] for i in order[n_hold:]]
    m.train(tr, epochs=args.epochs, diagnose=args.diagnose)
    if hold:
        print("[holdout]", json.dumps(m.evaluate(hold), indent=2))
    m.save(args.out)
    print("[save]", args.out)
    print("[diagnostics]", json.dumps(m.diagnostics(), indent=2))


def _cmd_infer(args):
    m = ForkGenerator.load(args.model)
    print(f"[infer] loaded model, steps={m.step}, absorbed={m.absorbed}")
    print('[infer] one JSON per line: {"delta": [...], "hum": [...], "state": [...], "phase": 0.0}')
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
            x = np.concatenate([
                np.array(obj["delta"], float),
                np.array(obj.get("hum", [0.5] * m.cfg["d_hum"]), float),
                np.array(obj.get("state", [0.0] * m.cfg["d_state"]), float),
            ])
            p = m.infer(x, phase=float(obj.get("phase", 0.0)))
            print(json.dumps(dict(trajectory_id=p.trajectory_id,
                                  proposal_weight=p.proposal_weight,
                                  latent_trajectory=p.latent_trajectory.tolist(),
                                  uncertainty=p.uncertainty), indent=2))
        except Exception as e:
            print("[infer][error]", e)


def _cmd_make_td(args):
    path, n = make_td_dat(args.out, n=args.n, seed=args.seed)
    print(f"[td.dat] wrote {n} items to {path}")


def _cmd_selftest(args):
    results = []

    def check(name, ok, detail=""):
        results.append(bool(ok))
        print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"  {detail}" if detail else ""))

    rng = np.random.default_rng(args.seed)
    raw = synth_corpus(n=1024, seed=args.seed)
    m = ForkGenerator(seed=args.seed)
    corpus = m.ingest_corpus(raw)
    tr, ho = corpus[:768], corpus[768:]
    m.train(tr, epochs=args.epochs, verbose=False)

    # 1-2 consolidation
    ev = m.evaluate(ho)
    check("consolidation learns: holdout skill > 0", ev["skill"] > 0, f"skill={ev['skill']:.3f}")
    pr = ev["unc_err_pearson"]
    check("uncertainty tracks held-out error: pearson > 0",
          np.isfinite(pr) and pr > 0, f"pearson={pr:.3f}")

    # 3 RoPE relative-phase identity: R(a)^T R(b) = R(b - a)
    rp = ForkGenerator(seed=args.seed, phase_on=True)
    H = rp.cfg["hidden"]
    z = rng.normal(size=(4, H))
    zz = rng.normal(size=(4, H))
    a, b = 0.37, 1.91
    lhs = np.sum(rp._rotate(z, a) * rp._rotate(zz, b), axis=1)
    rhs = np.sum(z * rp._rotate(zz, b - a), axis=1)
    gap = float(np.abs(lhs - rhs).max())
    check("RoPE: inner product depends only on relative phase", gap < 1e-9, f"gap={gap:.2e}")

    # 4 relaxation reaches ridge target when fully supported and undisturbed
    rd = ForkGenerator(seed=args.seed, mu0=0.0, diss_gain=0.0, s0=1e-9)
    for x, y, w in tr[:256]:
        rd.advance(0.1)
        rd.observe(x, y + 0.3, w)
    for _ in range(200):
        rd.relax(1.0)
    ridge = np.eye(rd.B.shape[1]) * rd.cfg["beta"]
    gap = max(float(np.abs(rd.D[p] - np.linalg.solve(rd.B[p] + ridge, rd.A[p].T).T).max())
              for p in range(N_CLASS))
    check("relaxation converges to ridge target A(B+beta I)^-1", gap < 1e-8, f"max_gap={gap:.2e}")

    # 5 ground state: no support -> residual returns to base
    gs = ForkGenerator(seed=args.seed, mu0=0.1)
    gs.D[:] = rng.normal(size=gs.D.shape) * 0.1
    for _ in range(200):
        gs.relax(1.5)
    left = float(np.abs(gs.D).max())
    check("ground state: no support, residual returns to base", left < 1e-9, f"|D|={left:.2e}")

    # 6 support resists decay (pure decay: target equals D, gate ~ 0)
    def retention(Mval):
        mm = ForkGenerator(seed=args.seed, mu0=0.1, diss_gain=0.0, s0=1e9)
        mm.D[:] = rng.normal(size=mm.D.shape) * 0.1
        Bp = rng.normal(size=(mm.B.shape[1], mm.B.shape[1]))
        for p in range(N_CLASS):
            Bs = Bp @ Bp.T / mm.B.shape[1]
            mm.B[p] = Bs
            mm.A[p] = mm.D[p] @ (Bs + np.eye(Bs.shape[0]) * mm.cfg["beta"])
        mm.M[:] = Mval
        before = mm.D.copy()
        mm.relax(1.0)
        return float(np.linalg.norm(mm.D) / np.linalg.norm(before))
    r0, r50 = retention(0.0), retention(50.0)
    check("support resists decay", r50 > r0, f"retain(M=0)={r0:.4f} retain(M=50)={r50:.4f}")

    # 7 dissonance bump: contradiction raises it, agreement does not, decay on field-time
    dm = ForkGenerator(seed=args.seed)
    x0, y0, w0 = tr[0]
    mu_pred = dm._predict_raw(x0[None, :], 0.0, hot=True)[0][0]
    b_agree0 = dm.bump[0]
    dm.observe(x0, mu_pred, 1.0)
    b_agree = dm.bump[0] - b_agree0
    z_c = dm.observe(x0, mu_pred + 3.0, 1.0)
    b_c = dm.bump[0]
    dm.advance(5.0)
    b_after = dm.bump[0]
    check("dissonance: agreement adds no bump, contradiction does",
          b_agree == 0.0 and b_c > 0 and z_c > 1, f"agree_bump={b_agree:.3f} contra_bump={b_c:.3f}")
    check("dissonance bump decays on field-time",
          b_after < 0.4 * b_c, f"before={b_c:.3f} after_dt5={b_after:.3f}")

    # 7b bounded: one survivor adds at most its own weight (times diss_gain) to the bump
    bd = ForkGenerator(seed=args.seed)
    bd.observe(x0, mu_pred + 1000.0, 1.0)
    bounded = float(bd.bump[0])
    check("dissonance bump bounded per survivor (<= w)", 0.0 < bounded <= 1.0 + 1e-12,
          f"bump={bounded:.6f}")

    # 7c bounded in total: a steady offset cannot push the bump past bump_max
    bt = ForkGenerator(seed=args.seed)
    for x, y, w in tr[:256]:
        bt.advance(0.1)
        bt.observe(x, y + 0.3, w)
    peak = float(bt.bump.max())
    check("dissonance bump bounded in total (<= bump_max)",
          peak <= bt.cfg["bump_max"] + 1e-12, f"peak={peak:.4f} cap={bt.cfg['bump_max']}")

    # 8 ledger conservation across fork / settle / expire / re-fork
    lg = ForkGenerator(seed=args.seed, ttl=3.0)
    provs = list(PROVENANCE)
    for i, (x, y, w) in enumerate(tr[:40]):
        lg.fork(x)
    tids = list(lg.ledger.keys())
    for i, t in enumerate(tids[:15]):
        lg.settle(t, tr[i][1], provs[i % 3])
    lg.advance(1.0)
    for i, (x, y, w) in enumerate(tr[40:50]):
        lg.fork(x)
    lg.advance(4.0)                       # expires the first batch (ttl=3)
    for i, (x, y, w) in enumerate(tr[50:60]):
        lg.fork(x)
    led = lg.ledger_check()
    check("ledger balances: n_arr = n_folded + n_discarded + n_open",
          led["balanced"] and led["n_expired"] > 0,
          f"arr={led['n_arr']} folded={led['n_folded']} discarded={led['n_discarded']} "
          f"open={led['n_open']} expired={led['n_expired']}")

    # 8b dissent folds at full weight for observed/inferred
    dv = ForkGenerator(seed=args.seed)
    dforks = [dv.fork(tr[i][0]).trajectory_id for i in range(20)]
    dres = [dv.settle(t_, tr[i][1] + 10.0, "observed") for i, t_ in enumerate(dforks)]
    min_w = min(r["w"] for r in dres)
    ok_d = all(r["folded"] for r in dres) and min_w >= 0.5 - 1e-12 and dv.n_discarded == 0
    check("observed dissent folds at full weight (w >= 0.5, never discarded)", ok_d,
          f"min_w={min_w:.3f} discarded={dv.n_discarded}")

    # 8c honored class: fork with header, settle without override -> folds as header
    hc = ForkGenerator(seed=args.seed)
    tid_h = hc.fork(tr[0][0], conditioning_tags={"provenance": "hallucinated"}).trajectory_id
    assert hc.ledger[tid_h]["provenance"] == "hallucinated"
    # for a for-shit hallucination (bump and M both zero), the gem check fails
    res_h = hc.settle(tid_h, tr[0][1] + 100.0)   # big surprise -> for-shit
    check("honored class: header routes settle; for-shit hallucinated discarded",
          res_h["discarded"] and not res_h["folded"] and not res_h["gem"]
          and hc.n_discarded == 1 and hc.SN[PROVENANCE.index("hallucinated")] == 1.0,
          f"res={res_h} SN={hc.SN.tolist()}")

    # 8d hallucinated gem: high bump (persistent surprise) OR high M (support) survives
    hg = ForkGenerator(seed=args.seed)
    hg.bump[PROVENANCE.index("hallucinated")] = hg.cfg["gem_bump"] + 0.1
    tid_g = hg.fork(tr[1][0], conditioning_tags={"provenance": "hallucinated"}).trajectory_id
    res_g = hg.settle(tid_g, tr[1][1])
    check("hallucinated gem: high bump survives, folds",
          res_g["gem"] and res_g["folded"] and not res_g["discarded"]
          and hg.n_folded == 1 and hg.n_discarded == 0,
          f"res={res_g}")

    # 8e override: settle() with explicit provenance overrides the honored class
    ov = ForkGenerator(seed=args.seed)
    tid_o = ov.fork(tr[0][0], conditioning_tags={"provenance": "hallucinated"}).trajectory_id
    res_o = ov.settle(tid_o, tr[0][1], provenance="observed")  # override to observed
    check("settle override: explicit provenance overrides honored class",
          res_o["folded"] and res_o["gem"] and ov.SN[PROVENANCE.index("observed")] == 1.0
          and ov.SN[PROVENANCE.index("hallucinated")] == 0.0,
          f"res={res_o} SN={ov.SN.tolist()}")

    # 8f bad header rejected at fork
    bh = ForkGenerator(seed=args.seed)
    try:
        bh.fork(tr[0][0], conditioning_tags={"provenance": "bogus"})
        bad_rejected = False
    except ValueError:
        bad_rejected = True
    check("fork rejects provenance header not in PROVENANCE", bad_rejected,
          f"n_arr={bh.n_arr}")

    # 9 absorb is prediction-preserving (phase-free)
    ab = ForkGenerator(seed=args.seed)
    ab.absorbed = 0
    for x, y, w in tr[:128]:
        ab.observe(x, y + 0.2, w)
    for _ in range(50):
        ab.relax(1.0)
    Xh, Yh, _ = _stack(ho)
    dnz = float(np.abs(ab.D).max())
    pre, _, _ = ab._predict_raw(Xh, 0.0, hot=True)
    ab.absorb()
    post, _, _ = ab._predict_raw(Xh, 0.0, hot=True)
    check("absorb preserves prediction", dnz > 0 and np.allclose(pre, post, atol=1e-9),
          f"|D|max={dnz:.3e}")

    # 10 antagonism replicator on three classes: eta=0 fixes on highest agreement;
    # eta>0 keeps a nonzero share of the rarer, novelty-settling class
    def replicator(eta, steps=20000):
        am = ForkGenerator(seed=args.seed, eta=eta)
        am.SN[:] = 1.0
        am.SA[:] = [0.8, 0.6, 0.5]
        am.SNA[:] = [0.04, 0.10, 0.20]
        for _ in range(steps):
            am.update_pi()
        return am.pi
    p0 = replicator(0.0)
    p2 = replicator(2.0)
    check("antagonism: eta=0 fixes on highest agreement", p0[0] > 0.99, f"pi={p0.round(4).tolist()}")
    check("antagonism: eta=2 keeps divergent class alive", p2[2] > 0.01, f"pi={p2.round(4).tolist()}")
    sx = ForkGenerator(seed=args.seed)
    sx.SN[:] = 1.0
    sx.SA[:] = [0.8, 0.6, 0.5]
    sx.SNA[:] = [0.04, 0.10, 0.20]
    for _ in range(50):
        sx.update_pi()
    pv = sx.pi
    check("pi is softmax(logits): positive, sums to 1",
          bool(np.all(pv > 0)) and abs(float(pv.sum()) - 1.0) < 1e-12,
          f"sum={pv.sum():.12f} min={pv.min():.2e}")

    # 11 relax guard
    try:
        rd.relax(dt=3.0)
        guard = False
    except ValueError:
        guard = True
    check("relax guard rejects kappa_hot*dt >= 1", guard)

    # 12 hot path learns a systematic residual shift (dissonance must not block it)
    hot = copy.deepcopy(m)
    hot.absorb()
    shift = 0.3
    Yref = Yh + shift

    def hot_mse(model):
        mu, _, _ = model._predict_raw(Xh, 0.0, hot=True)
        return float(((np.clip(mu, -MU_CLIP, MU_CLIP) - Yref) ** 2).mean())
    before = hot_mse(hot)
    for x, y, w in tr[:256]:
        hot.advance(0.1)
        hot.observe(x, y + shift, w)
    for _ in range(200):
        hot.relax(1.0)
    after = hot_mse(hot)
    check("hot path learns a systematic residual shift", after < before,
          f"before={before:.4f} after={after:.4f}")

    # 13 harness rejects non-finite values at every entry point; ledger stays balanced
    nf = ForkGenerator(seed=args.seed)
    xbad = corpus[0][0].copy()
    xbad[3] = np.nan

    def raised(fn):
        try:
            fn()
            return False
        except ValueError:
            return True
    ok_fork = raised(lambda: nf.fork(xbad))
    ok_ing = raised(lambda: nf.ingest_corpus([(xbad, corpus[0][1], 1.0)]))
    prop = nf.fork(corpus[0][0])
    ybad = corpus[0][1].copy()
    ybad[0] = np.nan
    ok_settle = raised(lambda: nf.settle(prop.trajectory_id, ybad, "observed"))
    still_open = prop.trajectory_id in nf.ledger
    balanced = nf.ledger_check()["balanced"]
    check("harness rejects non-finite at fork/ingest/settle; fork stays open",
          ok_fork and ok_ing and ok_settle and still_open and balanced,
          f"fork={ok_fork} ingest={ok_ing} settle={ok_settle} open={still_open} balanced={balanced}")

    ok = all(results)
    print(f"[selftest] {sum(results)}/{len(results)} passed")
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser(prog="fork_generator")
    sub = ap.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("train")
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--n", type=int, default=2048)
    t.add_argument("--epochs", type=int, default=200)
    t.add_argument("--holdout", type=float, default=0.2)
    t.add_argument("--out", default="fork_gen.pkl")
    t.add_argument("--corpus", default=None)
    t.add_argument("--diagnose", action="store_true")
    t.set_defaults(fn=_cmd_train)

    i = sub.add_parser("infer")
    i.add_argument("--model", default="fork_gen.pkl")
    i.set_defaults(fn=_cmd_infer)

    mk = sub.add_parser("make-td")
    mk.add_argument("--out", default="td.dat")
    mk.add_argument("--n", type=int, default=256)
    mk.add_argument("--seed", type=int, default=42)
    mk.set_defaults(fn=_cmd_make_td)

    st = sub.add_parser("selftest")
    st.add_argument("--seed", type=int, default=0)
    st.add_argument("--epochs", type=int, default=60)
    st.set_defaults(fn=_cmd_selftest)

    args = ap.parse_args()
    rc = args.fn(args)
    sys.exit(rc if isinstance(rc, int) else 0)


if __name__ == "__main__":
    main()
