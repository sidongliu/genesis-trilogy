"""Closed-loop comparison for the TD-II candidate (agent D).

Budgeted reader: D channels x ~ N(0, I_D) each step, outcome y = rho u.x + xi, xi ~ N(0, s^2).
The coupling direction u drifts by an isotropic step u' = (u + eta g_perp)/|...|.
Every agent may make b linear reads per step; y is revealed after the prediction.
All agents receive the same n0 full probes (n0*D reads) at the start.

Agents (common random numbers: same u_t, x_t, xi_t for all):
  zero  : no memory, yhat = 0
  frozen: g0 from the start-up probes, never revised, yhat = g0.x (one read along g0)
  noframe: fixed channels S = first b coordinates; coefficients learned by EMA
  fixed : frame v0 = g0/|g0| plus b-1 fixed orthonormal directions; coefficients learned by EMA
  calib : g in R^D; one task read along g (yhat = g.x); b-1 probe reads of uniformly chosen
          channels, each probed coordinate of g updated by EMA (the task read is not used for learning)
Risk is evaluated exactly given the state: s^2 + |rho u - c|^2, c the agent's effective vector.
Learning rates are set from the surrogate formulas (not tuned on the simulation output).
Writes td2_loop_output.json.
"""
import json, math, os, sys
import numpy as np
from scipy import integrate, stats

def step_factors(D, eta):
    """Exact decay factors of the isotropic step: lam1 = E[u'.u], lam2 for the degree-2 harmonic."""
    chi = stats.chi2(D - 1)
    f1 = lambda r: (1 + eta ** 2 * r) ** -0.5 * chi.pdf(r)
    f2 = lambda r: (1 + eta ** 2 * r) ** -1.0 * chi.pdf(r)
    e1 = integrate.quad(f1, 0, np.inf, limit=200)[0]
    e2 = integrate.quad(f2, 0, np.inf, limit=200)[0]
    lam2 = (D * e2 - 1) / (D - 1)
    return e1, lam2

def steady(alpha, p, lam1, rho, s, D, m):
    """Exact stationary moments of the EMA tracker of theta restricted to m coordinates
    (m = D for the calibrated frame with probe probability p; m = b, p = 1 for fixed channels).
    Returns (A, c, T) with A = E|e|^2, c = E[e.theta_m], T = E|theta_m|^2 = rho^2 m/D."""
    kap = 1 - lam1
    a = p * alpha * (2 - alpha)
    Nsum = m * (rho ** 2 + s ** 2 + rho ** 2 / D)
    T = rho ** 2 * m / D
    c = -kap * T / (1 - lam1 * (1 - p * alpha))
    A = (p * alpha ** 2 * Nsum + 2 * kap * T + 2 * kap * (1 - p * alpha) * c) / a
    return A, c, T

def risks(alpha, p, lam1, rho, s, D, m):
    """Stationary risk with coefficient gamma=1 and with the optimal shrink gamma*."""
    A, c, T = steady(alpha, p, lam1, rho, s, D, m)
    out_of_frame = rho ** 2 - T
    r1 = s ** 2 + out_of_frame + A
    gam = (T + c) / (T + 2 * c + A)
    rs = s ** 2 + rho ** 2 - (T + c) ** 2 / (T + 2 * c + A)
    return r1, rs, gam

def best_alpha(p, lam1, rho, s, D, m, which, floor=0.002):
    grid = np.geomspace(1e-4, 0.999, 4000)
    v = [risks(a, p, lam1, rho, s, D, m)[which] for a in grid]
    return max(float(grid[int(np.argmin(v))]), floor)

def run(D, b, rho, s, eta, n0, T, burn, ntraj, seed):
    rng_w = np.random.default_rng(seed)          # world stream
    rng_p = np.random.default_rng(seed + 7919)   # probe-choice stream
    lam1, lam2 = (1.0, 1.0) if eta == 0 else step_factors(D, eta)
    q = 2 * rho ** 2 * (1 - lam1)
    pC = (b - 1) / D
    aC = best_alpha(pC, lam1, rho, s, D, D, 0)
    aN = best_alpha(1.0, lam1, rho, s, D, b, 0)
    aCs = best_alpha(pC, lam1, rho, s, D, D, 1)
    aNs = best_alpha(1.0, lam1, rho, s, D, b, 1)
    gCs = risks(aCs, pC, lam1, rho, s, D, D)[2]
    gNs = risks(aNs, 1.0, lam1, rho, s, D, b)[2]
    # initial coupling and start-up probes
    U = rng_w.normal(size=(ntraj, D)); U /= np.linalg.norm(U, axis=1, keepdims=True)
    X0 = rng_w.normal(size=(ntraj, n0, D))
    y0 = rho * np.einsum('nkd,nd->nk', X0, U) + s * rng_w.normal(size=(ntraj, n0))
    g0 = np.einsum('nkd,nk->nd', X0, y0) / n0
    # agents' states
    S = np.arange(b)
    betaN = g0[:, S].copy()
    v0 = g0 / np.linalg.norm(g0, axis=1, keepdims=True)
    W = np.zeros((ntraj, D, b))
    for n in range(ntraj):
        M = np.column_stack([v0[n], rng_w.normal(size=(D, b - 1))])
        Q, _ = np.linalg.qr(M)
        Q[:, 0] *= np.sign(Q[:, 0] @ v0[n])
        W[n] = Q
    betaF = np.einsum('ndb,nd->nb', W, g0)
    gC = g0.copy()
    gCs_state = g0.copy()
    betaNs = g0[:, S].copy()
    acc = {k: np.zeros(ntraj) for k in ["zero", "frozen", "noframe", "fixed", "calib", "noframe_shrunk", "calib_shrunk"]}
    alignC = np.zeros(ntraj); alignF = np.zeros(ntraj)
    curve_t = []; curve = {k: [] for k in acc}
    cnt = 0
    for t in range(T):
        if t >= burn or t % 500 == 0:
            cN = np.zeros((ntraj, D)); cN[:, S] = betaN
            cF = np.einsum('ndb,nb->nd', W, betaF)
            th = rho * U
            r = {"zero": s ** 2 + np.sum(th ** 2, 1),
                 "frozen": s ** 2 + np.sum((th - g0) ** 2, 1),
                 "noframe": s ** 2 + np.sum((th - cN) ** 2, 1),
                 "fixed": s ** 2 + np.sum((th - cF) ** 2, 1),
                 "calib": s ** 2 + np.sum((th - gC) ** 2, 1)}
            cNs = np.zeros((ntraj, D)); cNs[:, S] = gNs * betaNs
            r["noframe_shrunk"] = s ** 2 + np.sum((th - cNs) ** 2, 1)
            r["calib_shrunk"] = s ** 2 + np.sum((th - gCs * gCs_state) ** 2, 1)
            if t % 500 == 0:
                curve_t.append(t)
                for k in acc:
                    curve[k].append(float(r[k].mean()))
            if t >= burn:
                for k in acc:
                    acc[k] += r[k]
                vC = gC / np.linalg.norm(gC, axis=1, keepdims=True)
                alignC += np.sum(U * vC, 1) ** 2
                alignF += np.sum(U * v0, 1) ** 2
                cnt += 1
        # one world step: input, outcome
        x = rng_w.normal(size=(ntraj, D))
        y = rho * np.sum(U * x, 1) + s * rng_w.normal(size=ntraj)
        # learning
        betaN += aN * (y[:, None] * x[:, S] - betaN)
        betaF += aN * (y[:, None] * np.einsum('ndb,nd->nb', W, x) - betaF)
        probe = np.argsort(rng_p.random((ntraj, D)), axis=1)[:, :b - 1]
        rows = np.arange(ntraj)[:, None]
        gC[rows, probe] += aC * (y[:, None] * x[rows, probe] - gC[rows, probe])
        gCs_state[rows, probe] += aCs * (y[:, None] * x[rows, probe] - gCs_state[rows, probe])
        betaNs += aNs * (y[:, None] * x[:, S] - betaNs)
        # drift
        if eta > 0:
            gpert = rng_w.normal(size=(ntraj, D))
            gpert -= np.sum(gpert * U, 1, keepdims=True) * U
            U = U + eta * gpert
            U /= np.linalg.norm(U, axis=1, keepdims=True)
    out = {}
    for k in acc:
        per = acc[k] / cnt
        out[k] = dict(mean=float(per.mean()), se=float(per.std(ddof=1) / math.sqrt(ntraj)))
    # paired difference calib - noframe
    dif = (acc["calib"] - acc["noframe"]) / cnt
    out["calib_minus_noframe"] = dict(mean=float(dif.mean()), se=float(dif.std(ddof=1) / math.sqrt(ntraj)))
    dif2 = (acc["calib_shrunk"] - acc["noframe_shrunk"]) / cnt
    out["calib_shrunk_minus_noframe_shrunk"] = dict(mean=float(dif2.mean()), se=float(dif2.std(ddof=1) / math.sqrt(ntraj)))
    theory = dict(lam1=lam1, lam2=lam2, q=q, sigma_cont=-math.log(lam1) / (D - 1) if eta > 0 else 0.0,
                  alpha_calib=aC, alpha_noframe=aN, alpha_calib_shrunk=aCs, alpha_noframe_shrunk=aNs,
                  gamma_calib=gCs, gamma_noframe=gNs,
                  calib=risks(aC, pC, lam1, rho, s, D, D)[0],
                  noframe=risks(aN, 1.0, lam1, rho, s, D, b)[0],
                  calib_shrunk=risks(aCs, pC, lam1, rho, s, D, D)[1],
                  noframe_shrunk=risks(aNs, 1.0, lam1, rho, s, D, b)[1],
                  zero=s ** 2 + rho ** 2,
                  frozen_limit_formula="s^2 + rho^2 + E|g0|^2 - 2 rho E[g0.u0] lam1^t")
    return dict(params=dict(D=D, b=b, rho=rho, s=s, eta=eta, n0=n0, T=T, burn=burn, ntraj=ntraj, seed=seed),
                sim=out, theory=theory,
                align_calib=float(alignC.mean() / cnt), align_fixed=float(alignF.mean() / cnt),
                curve=dict(t=curve_t, **curve))

if __name__ == "__main__":
    res = []
    D, b, rho, s, n0 = 8, 3, 1.0, 0.5, 64
    for eta in [0.0, 0.002, 0.005, 0.01, 0.02, 0.04, 0.06, 0.08, 0.12]:
        r = run(D, b, rho, s, eta, n0, T=24000, burn=8000, ntraj=200, seed=4100 + int(eta * 1e4))
        res.append(r)
        print(eta, {k: round(v["mean"], 4) for k, v in r["sim"].items()}, flush=True)
        print("   theory", {k: (round(v, 4) if isinstance(v, float) else v) for k, v in r["theory"].items() if k in ("calib", "noframe", "calib_shrunk", "noframe_shrunk", "sigma_cont", "gamma_calib")},
              "align", round(r["align_calib"], 3), flush=True)
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "td2_loop_output.json")
    with open(path, "w") as f:
        json.dump(res, f, indent=1)
