# /home/tdrsvr/Desktop/slop/gradcheck.py
import numpy as np, fork_generator as fg
m = fg.ForkGenerator(seed=0, hidden=8, d_delta=4, d_hum=2, d_state=2, d_fork=3)
rng = np.random.default_rng(1)
X = rng.normal(size=(5, 8)) * 0.5
Y = rng.normal(size=(5, 3)) * 0.3
W = rng.uniform(0.5, 1.0, 5)
loss, grads, aux = m._consolidate_step(X, Y, W)
# stop-gradient target e_s, fixed at base point
err0 = np.clip(aux["mu"] - Y, -fg.ERR_CLIP, fg.ERR_CLIP)
e_s = (err0 ** 2).mean(1, keepdims=True)
def L():
    z1 = np.tanh(X @ m.W1 + m.b1); z2 = np.tanh(z1 @ m.W2 + m.b2)
    mu = np.clip(z2 @ m.Wm + m.bm, -fg.MU_CLIP, fg.MU_CLIP)
    lv = np.clip(z2 @ m.Wv + m.bv, fg.LV_MIN, fg.LV_MAX)
    lu = np.clip(z2 @ m.Wu + m.bu, fg.LV_MIN, fg.LV_MAX)
    e = np.clip(mu - Y, -fg.ERR_CLIP, fg.ERR_CLIP)
    lm = 0.5 * (e ** 2 * np.exp(-lv) + lv)
    lu_ = 0.5 * (e_s * np.exp(lu) - lu)
    wc = W[:, None]
    return (lm * wc).mean() + (lu_ * wc).mean()
print("loss analytic vs recomputed:", abs(loss - L()))
eps = 1e-6; worst = 0.0; r = np.random.default_rng(2)
for k, P in m.param_space().items():
    for _ in range(6):
        idx = tuple(int(r.integers(0, s)) for s in P.shape)
        old = P[idx]
        P[idx] = old + eps; lp = L()
        P[idx] = old - eps; lm_ = L()
        P[idx] = old
        worst = max(worst, abs((lp - lm_) / (2 * eps) - grads[k][idx]))
print("max |numeric - analytic| over sampled entries:", worst)
