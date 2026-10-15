"""c2: gauge theorem R2 checked by exact enumeration (binary body loop) and Kalman likelihoods (linear loop).

Binary: body B, bath flip d, instrument P(y|B)=(1+sigma*g*y*B)/2 (setting matched), agent action a_t in {0,1}
(attempt flip) chosen by an arbitrary deterministic function of the record history and current reading;
flip succeeds with prob s.  The record law under (sigma=+1, P(B0=+1)=p0) must equal that under
(sigma=-1, P(B0=+1)=1-p0) for EVERY policy.  With a reference read (known input +1) appended each step,
the two laws differ (gauge broken).
"""
import json, itertools
import numpy as np

rng = np.random.default_rng(20261007)

def record_law(policy, sigma, p0, g, d, s, T, ref=False):
    """Exact law of records (y_1,a_1,[z_1],...,y_T,a_T,[z_T]) as dict."""
    law = {}
    # state: (B, record tuple, prob)
    states = [(1, (), p0), (-1, (), 1 - p0)]
    for t in range(T):
        new = []
        for B, rec, pr in states:
            for y in (1, -1):
                py = (1 + sigma * g * y * B) / 2
                a = policy(rec, y)
                refs = ((1, (1 + sigma * g) / 2), (-1, (1 - sigma * g) / 2)) if ref else ((None, 1.0),)
                for z, pz in refs:
                    rec2 = rec + ((y, a, z) if ref else (y, a))
                    # flip
                    outs = [(B, 1.0)] if a == 0 else [(-B, s), (B, 1 - s)]
                    for B1, pf in outs:
                        for flipb, pb in ((1, 1 - d), (-1, d)):
                            new.append((B1 * flipb, rec2, pr * py * pz * pf * pb))
        # merge
        agg = {}
        for B, rec, pr in new:
            agg[(B, rec)] = agg.get((B, rec), 0.0) + pr
        states = [(B, rec, pr) for (B, rec), pr in agg.items()]
    for B, rec, pr in states:
        law[rec] = law.get(rec, 0.0) + pr
    return law

def random_policy(seed):
    r = np.random.default_rng(seed)
    table = {}
    def pol(rec, y):
        key = (rec, y)
        if key not in table:
            table[key] = int(r.integers(0, 2))
        return table[key]
    return pol

res = {"binary": []}
for trial in range(20):
    g = rng.uniform(0.2, 0.95); d = rng.uniform(0.01, 0.3); s = rng.uniform(0.5, 1.0); p0 = rng.uniform(0.05, 0.95)
    T = 5
    pol = random_policy(1000 + trial)
    L1 = record_law(pol, +1, p0, g, d, s, T)
    L2 = record_law(pol, -1, 1 - p0, g, d, s, T)
    keys = set(L1) | set(L2)
    tv = 0.5 * sum(abs(L1.get(k, 0) - L2.get(k, 0)) for k in keys)
    pol2 = random_policy(5000 + trial)
    R1 = record_law(pol2, +1, p0, g, d, s, 3, ref=True)
    R2 = record_law(pol2, -1, 1 - p0, g, d, s, 3, ref=True)
    keys = set(R1) | set(R2)
    tv_ref = 0.5 * sum(abs(R1.get(k, 0) - R2.get(k, 0)) for k in keys)
    res["binary"].append(dict(g=g, d=d, s=s, p0=p0, TV_no_ref=tv, TV_with_ref=tv_ref))
res["binary_max_TV_no_ref"] = max(r["TV_no_ref"] for r in res["binary"])
res["binary_min_TV_with_ref"] = min(r["TV_with_ref"] for r in res["binary"])

# loss identity R2(b): P_theta(B_t=-1)+P_gamma theta(B_t=-1)=1
def loss_curve(policy, sigma, p0, g, d, s, T):
    states = [(1, (), p0), (-1, (), 1 - p0)]
    out = []
    for t in range(T):
        new = []
        for B, rec, pr in states:
            for y in (1, -1):
                py = (1 + sigma * g * y * B) / 2
                a = policy(rec, y)
                rec2 = rec + ((y, a),)
                outs = [(B, 1.0)] if a == 0 else [(-B, s), (B, 1 - s)]
                for B1, pf in outs:
                    for flipb, pb in ((1, 1 - d), (-1, d)):
                        new.append((B1 * flipb, rec2, pr * py * pf * pb))
        states = new
        out.append(sum(pr for B, rec, pr in states if B == -1))
    return out
lc = []
for trial in range(5):
    g, d, s, p0 = 0.8, 0.05, 0.9, rng.uniform(0.1, 0.9)
    pol = lambda rec, y: 1 if y == -1 else 0   # natural corrective policy
    a1 = loss_curve(pol, +1, p0, g, d, s, 6)
    a2 = loss_curve(pol, -1, 1 - p0, g, d, s, 6)
    lc.append(dict(p0=p0, loss_sigma_plus=a1, loss_sigma_minus=a2, sums=[x + y for x, y in zip(a1, a2)]))
res["loss_identity"] = lc

# Linear loop: Kalman log-likelihood of y given causal adaptive policy with dither, under theta and g theta
def kalman_loglik(y, u, a, kap, f, sw, g, b, se, m0, v0):
    m, v, ll = m0, v0, 0.0
    for t in range(len(y)):
        S = g * g * v + se ** 2
        r = y[t] - (g * m + b)
        ll += -0.5 * (np.log(2 * np.pi * S) + r * r / S)
        K = v * g / S
        m = m + K * r; v = (1 - K * g) * v
        m = a * m + kap * u[t] + f; v = a * a * v + sw ** 2
    return ll

lin = []
for trial in range(20):
    a = rng.uniform(0.3, 0.95); kap = rng.uniform(0.2, 1.5); f = rng.normal(); sw = rng.uniform(0.1, 1)
    g = rng.uniform(0.3, 2); b = rng.normal(); se = rng.uniform(0.1, 1); m0 = rng.normal(); v0 = rng.uniform(0.1, 2)
    T = 200
    # simulate under theta with an adaptive nonlinear policy + known dither
    x = m0 + np.sqrt(v0) * rng.normal(); ys, us = [], []
    dith = rng.normal(size=T)
    for t in range(T):
        yt = g * x + b + se * rng.normal()
        Kt = 0.4 + 0.3 * np.tanh(ys[-1]) if ys else 0.4
        ut = -Kt * yt + 0.5 * dith[t]
        ys.append(yt); us.append(ut)
        x = a * x + kap * ut + f + sw * rng.normal()
    lam = rng.choice([-1, 1]) * rng.uniform(0.2, 5); mu = rng.normal() * 3
    ll1 = kalman_loglik(ys, us, a, kap, f, sw, g, b, se, m0, v0)
    ll2 = kalman_loglik(ys, us, a, lam * kap, lam * f + mu * (1 - a), abs(lam) * sw, g / lam, b - g * mu / lam, se,
                        lam * m0 + mu, lam ** 2 * v0)
    # a non-gauge change (kappa only) for contrast
    ll3 = kalman_loglik(ys, us, a, 1.3 * kap, f, sw, g, b, se, m0, v0)
    lin.append(dict(lam=lam, mu=mu, dLL_gauge=ll1 - ll2, dLL_kappa_only=ll1 - ll3))
res["linear_max_abs_dLL_gauge"] = max(abs(r["dLL_gauge"]) for r in lin)
res["linear_min_abs_dLL_nongauge"] = min(abs(r["dLL_kappa_only"]) for r in lin)
res["linear"] = lin
json.dump(res, open("out/c2_gauge_enum.json", "w"), indent=1, default=float)
print("binary max TV (no ref):", res["binary_max_TV_no_ref"])
print("binary min TV (with ref):", res["binary_min_TV_with_ref"])
print("loss identity sums:", [ [round(z,15) for z in r["sums"]] for r in lc][:2])
print("linear max |dLL| gauge:", res["linear_max_abs_dLL_gauge"], " min |dLL| kappa-only:", res["linear_min_abs_dLL_nongauge"])
