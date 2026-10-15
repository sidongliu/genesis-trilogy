"""c3: Theorem R3 (sign inversions) checked against the Bayes-optimal policy computed by exact backward induction.

Cycle: inversion of sigma w.p. eta and bath flip of B w.p. d; reading y ~ (1+sigma*g*y*B)/2; action a in {0,1}
(attempt a flip of B, success s).  sigma_0=+1 and B_0=+1 known.  Loss after cycle T: P(B_T=-1).
The optimal policy over ALL adaptive policies minimizes this exactly; R3 says it is >= 1/2(1-(1-2 eta)^T) (eta<=d)
and >= 1/2(1-varrho^T) always.  Also checks P(sigma_T != sigma_0 | record) <= 1/2 for every record (eta<=d)
and its mean equals the prior.
"""
import json
import numpy as np
from functools import lru_cache

def run(g, d, s, eta, T):
    # hidden index: (sigma, B) -> 0:(+,+) 1:(+,-) 2:(-,+) 3:(-,-)
    sig = np.array([1, 1, -1, -1]); Bv = np.array([1, -1, 1, -1])
    # start-of-cycle transition: inversion (flip sigma) and bath (flip B)
    Tm = np.zeros((4, 4))
    for i in range(4):
        for di in (0, 1):
            for db in (0, 1):
                sg = sig[i] * (-1) ** di; b = Bv[i] * (-1) ** db
                j = [k for k in range(4) if sig[k] == sg and Bv[k] == b][0]
                Tm[j, i] += (eta if di else 1 - eta) * (d if db else 1 - d)
    Fl = np.zeros((4, 4))
    for i in range(4):
        j = [k for k in range(4) if sig[k] == sig[i] and Bv[k] == -Bv[i]][0]
        Fl[j, i] += s; Fl[i, i] += 1 - s
    Py = {y: (1 + sig * g * y * Bv) / 2 for y in (1, -1)}

    def V(w, k):
        if k == T:
            return w[1] + w[3]
        w1 = Tm @ w
        tot = 0.0
        for y in (1, -1):
            wy = Py[y] * w1
            tot += min(V(wy, k + 1), V(Fl @ wy, k + 1))
        return tot

    # posterior statistics under the natural corrective policy (flip iff y=-1)
    post_max = 0.0; post_mean = 0.0
    def walk(w, k):
        nonlocal post_max, post_mean
        if k == T:
            z = w.sum()
            if z > 0:
                ps = (w[2] + w[3]) / z
                post_max = max(post_max, ps)
                post_mean += z * ps
            return
        w1 = Tm @ w
        for y in (1, -1):
            wy = Py[y] * w1
            walk(Fl @ wy if y == -1 else wy, k + 1)
    w0 = np.array([1.0, 0, 0, 0])
    opt = V(w0, 0)
    walk(w0, 0)
    q1 = eta * (1 - d) / (eta * (1 - d) + d * (1 - eta)); q0 = eta * d / (eta * d + (1 - eta) * (1 - d))
    vr = max(abs(1 - 2 * q0), abs(1 - 2 * q1))
    return dict(T=T, opt_loss=opt, bound_eta=0.5 * (1 - (1 - 2 * eta) ** T), bound_general=0.5 * (1 - vr ** T),
                post_max_natural=post_max, post_mean_natural=post_mean, prior_odd=0.5 * (1 - (1 - 2 * eta) ** T))

res = {}
for (g, d, s, eta) in [(0.9, 0.1, 0.95, 0.05), (0.6, 0.2, 0.8, 0.1), (0.95, 0.05, 0.99, 0.02), (0.9, 0.05, 0.95, 0.1)]:
    rows = [run(g, d, s, eta, T) for T in range(1, 9)]
    res[f"g={g},d={d},s={s},eta={eta}"] = rows
    print(f"g={g} d={d} s={s} eta={eta}")
    for r in rows:
        print(f"  T={r['T']}  opt={r['opt_loss']:.6f}  bound(eta<=d)={r['bound_eta']:.6f}  bound_gen={r['bound_general']:.6f}"
              f"  post_max={r['post_max_natural']:.4f}  post_mean={r['post_mean_natural']:.6f} prior={r['prior_odd']:.6f}")

# long-run simulation of the natural corrective agent: loss -> 1/2 with eta>0, stays low with eta=0
rng = np.random.default_rng(7)
def sim(g, d, s, eta, T, ntraj=4000):
    sigma = np.ones(ntraj); B = np.ones(ntraj); out = []
    for t in range(T):
        sigma *= np.where(rng.random(ntraj) < eta, -1, 1)
        B *= np.where(rng.random(ntraj) < d, -1, 1)
        y = np.where(rng.random(ntraj) < (1 + sigma * g * B) / 2, 1, -1)
        fl = (y == -1) & (rng.random(ntraj) < s)
        B = np.where(fl, -B, B)
        out.append(np.mean(B == -1))
    return np.array(out)
lr = {}
for eta in (0.0, 0.002, 0.01):
    L = sim(0.9, 0.05, 0.95, eta, 3000)
    lr[eta] = dict(loss_t100=float(L[90:110].mean()), loss_t500=float(L[490:510].mean()), loss_t2000=float(L[1990:2010].mean()),
                   loss_t3000=float(L[-20:].mean()), bound_t500=0.5 * (1 - (1 - 2 * eta) ** 500), bound_t2000=0.5 * (1 - (1 - 2 * eta) ** 2000))
    print("eta", eta, lr[eta])
res["long_run_natural"] = lr
json.dump(res, open("checks/out/c3_sign_inversion.json".replace("checks/", ""), "w"), indent=1, default=float)
