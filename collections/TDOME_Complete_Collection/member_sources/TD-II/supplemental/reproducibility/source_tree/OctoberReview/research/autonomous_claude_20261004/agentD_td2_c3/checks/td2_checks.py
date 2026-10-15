"""Checks for the TD-II candidate (agent D).

Deterministic.  Writes td2_checks_output.json next to this file.
Sections:
  A  capability of a budgeted reader (exact enumeration over a finite prior)
  B  frame-information rate Lambda*_{D,k}: null law, mgf, Legendre transform, large-D limit,
     D=2 von Mises form, probe-count corollary
  C  in-frame identifiability and the D=2 even residual (Fisher information formula)
  D  drift: discrete isotropic rotations, decay factors, frozen / fixed-frame risks
"""
import json, math, os
import numpy as np
import mpmath as mp
from scipy import stats

OUT = {}
rng = np.random.default_rng(20261005)
mp.mp.dps = 40


def unit(v):
    return v / np.linalg.norm(v)


# ---------------------------------------------------------------- A
def check_A():
    D, k, rho, s = 4, 2, 1.3, 0.6
    nU, nM = 20, 5
    U = np.array([unit(rng.normal(size=D)) for _ in range(nU)])
    prior = rng.dirichlet(np.ones(nU))
    chan = rng.dirichlet(np.ones(nM), size=nU)          # P(M=m | U=i)
    joint = prior[:, None] * chan                        # (i,m)
    pm = joint.sum(0)
    post = joint / pm                                     # P(i | m)
    mvec = post.T @ U                                     # E[u | m], shape (nM, D)
    # an arbitrary foreground for each memory value
    Ps = []
    for m in range(nM):
        Q, _ = np.linalg.qr(rng.normal(size=(D, k)))
        Ps.append(Q @ Q.T)
    # risk of linear predictor yhat = c(m).(P x): s^2 + E|rho u - P c|^2
    def risk(C):
        r = s ** 2
        for i in range(nU):
            for m in range(nM):
                d = rho * U[i] - Ps[m] @ C[m]
                r += joint[i, m] * d @ d
        return r
    Copt = np.array([rho * Ps[m] @ mvec[m] for m in range(nM)])
    r_opt = risk(Copt)
    formula = s ** 2 + rho ** 2 * (1 - sum(pm[m] * mvec[m] @ Ps[m] @ mvec[m] for m in range(nM)))
    worst_gain = min(risk(Copt + 1e-3 * rng.normal(size=Copt.shape)) - r_opt for _ in range(200))
    # Jensen: E|Pm|^2 <= E|Pu|^2
    EPm = sum(pm[m] * mvec[m] @ Ps[m] @ mvec[m] for m in range(nM))
    EPu = sum(joint[i, m] * U[i] @ Ps[m] @ U[i] for i in range(nU) for m in range(nM))
    # top-k optimality for a general matrix task: risk = nu + E tr W - E tr(P What P)
    p = 3
    A_list = [rng.normal(size=(p, D)) for _ in range(nU)]
    Abar = [sum(post[i, m] * A_list[i] for i in range(nU)) for m in range(nM)]
    best = 0.0
    for m in range(nM):
        What = Abar[m].T @ Abar[m]
        ev = np.sort(np.linalg.eigvalsh(What))[::-1]
        topk = ev[:k].sum()
        # random rank-k projections never exceed topk
        mx = max(np.trace((lambda Q: Q @ Q.T)(np.linalg.qr(rng.normal(size=(D, k)))[0]) @ What)
                 for _ in range(2000))
        best = max(best, mx - topk)
    OUT["A"] = dict(risk_opt=r_opt, risk_formula=formula, residual=abs(r_opt - formula),
                    min_perturbation_gain=worst_gain, E_Pm2=EPm, E_Pu2=EPu,
                    jensen_holds=bool(EPm <= EPu + 1e-15),
                    max_random_projection_minus_topk=best)


# ---------------------------------------------------------------- B
def logmgf(kappa, D, k):
    return float(mp.log(mp.hyp1f1(mp.mpf(k) / 2, mp.mpf(D) / 2, kappa)))


def mean_t(kappa, D, k):
    a, b = mp.mpf(k) / 2, mp.mpf(D) / 2
    return float((a / b) * mp.hyp1f1(a + 1, b + 1, kappa) / mp.hyp1f1(a, b, kappa))


def Lstar(c, D, k):
    """Legendre transform of the log-mgf of Beta(k/2,(D-k)/2) at c >= k/D (nats)."""
    if c <= k / D:
        return 0.0, 0.0
    lo, hi = 0.0, 1.0
    while mean_t(hi, D, k) < c:
        hi *= 2
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if mean_t(mid, D, k) < c:
            lo = mid
        else:
            hi = mid
    kap = 0.5 * (lo + hi)
    return kap * c - logmgf(kap, D, k), kap


def check_B():
    res = {}
    # null law of |P u|^2 for Haar u is Beta(k/2,(D-k)/2)
    ks_p = {}
    for D, k in [(3, 1), (8, 2), (16, 1)]:
        u = rng.normal(size=(40000, D))
        u /= np.linalg.norm(u, axis=1, keepdims=True)
        t = (u[:, :k] ** 2).sum(1)
        ks_p[f"D{D}k{k}"] = float(stats.kstest(t, stats.beta(k / 2, (D - k) / 2).cdf).pvalue)
        # mgf at kappa=1.7 : Monte Carlo vs 1F1
        ks_p[f"D{D}k{k}_mgf_mc"] = float(np.mean(np.exp(1.7 * t)))
        ks_p[f"D{D}k{k}_mgf_1F1"] = float(mp.hyp1f1(k / 2, D / 2, 1.7))
    res["null_law"] = ks_p
    # table of Lambda* (nats and bits), k=1
    table = []
    for D in [2, 3, 4, 8, 16, 64, 256]:
        row = {"D": D}
        for c in [0.5, 0.9, 0.99]:
            L, kap = Lstar(c, D, 1)
            row[f"c{c}"] = L
            row[f"c{c}_bits"] = L / math.log(2)
            row[f"c{c}_per_half_log"] = L / (0.5 * D * math.log(1 / (1 - c)))
        table.append(row)
    res["Lstar_k1"] = table
    # D=2 closed form with von Mises: sup_kappa [kappa(2c-1) - log I0(kappa)]
    d2 = []
    for c in [0.6, 0.75, 0.9, 0.99]:
        def neg(kp):
            return -(kp * (2 * c - 1) - float(mp.log(mp.besseli(0, kp))))
        grid = np.linspace(0, 200, 400001)
        vals = [-(neg(x)) for x in grid[::400]]
        kp0 = grid[::400][int(np.argmax(vals))]
        # refine
        from scipy.optimize import minimize_scalar
        r = minimize_scalar(neg, bracket=(max(kp0 - 1, 0), kp0, kp0 + 1))
        d2.append(dict(c=c, vonmises=-r.fun, legendre=Lstar(c, 2, 1)[0]))
    res["D2_vonmises_vs_legendre"] = d2
    # k>1 examples
    res["Lstar_k2_D8_c0.9"] = Lstar(0.9, 8, 2)[0]
    res["Lstar_k4_D64_c0.9"] = Lstar(0.9, 64, 4)[0]
    # equality case: tilted law Q(dt) ∝ e^{kappa t} Beta has mean c and KL = Lambda*(c)
    D, k, c = 8, 1, 0.9
    L, kap = Lstar(c, D, k)
    f = lambda t: mp.exp(kap * t) * t ** (k / 2 - 1) * (1 - t) ** ((D - k) / 2 - 1) / mp.beta(k / 2, (D - k) / 2)
    Z = mp.quad(f, [0, 1])
    meanQ = mp.quad(lambda t: t * f(t), [0, 1]) / Z
    KL = mp.quad(lambda t: f(t) / Z * mp.log(mp.exp(kap * t) / Z), [0, 1])
    res["tilted_equality"] = dict(Lstar=L, KL=float(KL), meanQ=float(meanQ))
    # probe corollary: overlap after n full probes with the correlation estimator vs bound
    D, rho, s = 8, 1.0, 0.5
    C1 = 0.5 * math.log(1 + rho ** 2 / s ** 2)
    rows = []
    for n in [4, 16, 64, 256]:
        ov = []
        for _ in range(4000):
            u = unit(rng.normal(size=D))
            X = rng.normal(size=(n, D))
            y = rho * X @ u + s * rng.normal(size=n)
            g = X.T @ y / n
            ov.append((u @ g) ** 2 / (g @ g))
        c = float(np.mean(ov))
        rows.append(dict(n=n, overlap=c, Lstar=Lstar(c, D, 1)[0],
                         info_ceiling=n * C1, bound_holds=bool(Lstar(c, D, 1)[0] <= n * C1)))
    res["probe_bound"] = dict(C1_nats_per_step=C1, rows=rows)
    OUT["B"] = res


# ---------------------------------------------------------------- C
def check_C():
    rho, s = 1.0, 0.5
    def cov(theta):
        a = rho * math.cos(theta)
        return np.array([[1.0, a], [a, rho ** 2 + s ** 2]])
    def fisher_numeric(theta, h=1e-5):
        S = cov(theta); Si = np.linalg.inv(S)
        dS = (cov(theta + h) - cov(theta - h)) / (2 * h)
        return 0.5 * np.trace(Si @ dS @ Si @ dS)
    def fisher_formula(theta):
        return (rho ** 2 * math.sin(theta) ** 2 * (rho ** 2 * math.cos(theta) ** 2 + rho ** 2 + s ** 2)
                / (s ** 2 + rho ** 2 * math.sin(theta) ** 2) ** 2)
    rows = []
    for th in [0.01, 0.05, 0.2, 0.6, 1.2]:
        rows.append(dict(theta=th, numeric=fisher_numeric(th), formula=fisher_formula(th),
                         small_angle=rho ** 2 * th ** 2 * (2 * rho ** 2 + s ** 2) / s ** 4,
                         even=abs(fisher_formula(th) - fisher_formula(-th))))
    # identifiability: two directions with same Pu give the same Gaussian law of (Px,y)
    D, k = 6, 2
    P = np.zeros((D, D)); P[:k, :k] = np.eye(k)
    u1 = unit(rng.normal(size=D))
    w = rng.normal(size=D - k); w = w / np.linalg.norm(w) * np.linalg.norm(u1[k:])
    u2 = np.concatenate([u1[:k], w])
    def joint_cov(u, rho_, s_):
        top = np.eye(k); off = rho_ * u[:k]
        return np.block([[top, off[:, None]], [off[None, :], np.array([[rho_ ** 2 + s_ ** 2]])]])
    same_dir = np.abs(joint_cov(u1, rho, s) - joint_cov(u2, rho, s)).max()
    # attribution ambiguity: (rho', s', u') with rho' u'_fg = rho u_fg and rho'^2+s'^2 = rho^2+s^2
    b = rho * u1[:k]; V = rho ** 2 + s ** 2
    rho2 = 0.5 * (np.linalg.norm(b) + math.sqrt(V))   # any value in [|b|, sqrt V]
    s2 = math.sqrt(V - rho2 ** 2)
    ufg = b / rho2
    ubg = unit(rng.normal(size=D - k)) * math.sqrt(1 - ufg @ ufg)
    u3 = np.concatenate([ufg, ubg])
    attrib = np.abs(joint_cov(u1, rho, s) - joint_cov(u3, rho2, s2)).max()
    OUT["C"] = dict(fisher_rows=rows, same_Pu_law_difference=float(same_dir),
                    attribution_law_difference=float(attrib),
                    attribution_pair=dict(rho=rho, s=s, rho_alt=rho2, s_alt=s2,
                                          misalignment=float(1 - u1[:k] @ u1[:k]),
                                          misalignment_alt=float(1 - ufg @ ufg)))


# ---------------------------------------------------------------- D
def random_rotation_small(D, eps):
    O = rng.normal(size=(D, D)) * eps
    Om = (O - O.T) / math.sqrt(2)
    w, V = np.linalg.eig(Om)            # skew: expm via eig
    R = (V @ np.diag(np.exp(w)) @ np.linalg.inv(V)).real
    return R


def check_D():
    res = {}
    for D in [2, 8]:
        eps = 0.05
        # decay factors of a conjugation-invariant step distribution
        e = np.zeros(D); e[0] = 1
        R_samples = [random_rotation_small(D, eps) for _ in range(20000)]
        d1 = np.array([R[:, 0] @ e for R in R_samples])     # e.R e
        lam1 = float(d1.mean())
        lam2 = float((D * np.mean(d1 ** 2) - 1) / (D - 1))
        # simulate trajectories
        T, ntraj = 120, 3000
        u0 = np.zeros(D); u0[0] = 1
        U = np.tile(u0, (ntraj, 1))
        m1, m2 = [], []
        idx = rng.integers(0, len(R_samples), size=(T, ntraj))
        for t in range(T):
            m1.append(float(U[:, 0].mean())); m2.append(float((U[:, 0] ** 2).mean()))
            U = np.einsum('nij,nj->ni', np.array([R_samples[j] for j in idx[t]]), U)
        tt = np.arange(T)
        pred1 = lam1 ** tt
        pred2 = 1 / D + (1 - 1 / D) * lam2 ** tt
        se1 = float(np.max(np.abs(np.array(m1) - pred1)))
        se2 = float(np.max(np.abs(np.array(m2) - pred2)))
        # continuous-time comparison: lam1 ~ exp(-(D-1) sigma), lam2 ~ exp(-2 D sigma)
        sigma_from_lam1 = -math.log(lam1) / (D - 1)
        sigma_from_lam2 = -math.log(lam2) / (2 * D)
        rho, s, b = 1.0, 0.5, 2 if D > 2 else 1
        frozen = [s ** 2 + 2 * rho ** 2 * (1 - x) for x in pred1]
        tstar = math.log(2) / (-math.log(lam1))
        fixed = [s ** 2 + rho ** 2 * (1 - b / D) * (1 - lam2 ** t) for t in tt]
        res[f"D{D}"] = dict(lam1=lam1, lam2=lam2, max_dev_m1=se1, max_dev_m2=se2,
                            sigma_from_lam1=sigma_from_lam1, sigma_from_lam2=sigma_from_lam2,
                            frozen_t0=frozen[0], frozen_t60=frozen[60], tstar_steps=tstar,
                            fixed_t60=fixed[60], noframe_level=s ** 2 + rho ** 2 * (1 - b / D))
    OUT["D"] = res


if __name__ == "__main__":
    check_A(); check_B(); check_C(); check_D()
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "td2_checks_output.json")
    with open(path, "w") as f:
        json.dump(OUT, f, indent=1)
    print(json.dumps(OUT, indent=1)[:6000])
