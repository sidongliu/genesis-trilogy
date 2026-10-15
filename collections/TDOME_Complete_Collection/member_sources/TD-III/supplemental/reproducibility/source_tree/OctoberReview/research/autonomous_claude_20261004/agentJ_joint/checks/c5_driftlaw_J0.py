"""c5: Proposition R6 (drift law with free loop information), self-label price ratios, reading-tax factor,
and free calibration of the magnitude by the body loop (tracking slope)."""
import json
import numpy as np
from scipy.optimize import minimize_scalar
from common import h, LN2

res = {}
# (a) floor of TD-III Thm with J0, units w=J=c=1
def floor(nu, J0, w=1.0, J=1.0, c=1.0):
    def hP(P):
        return (w / J) * (nu ** 2 / (P * (P + nu ** 2)) - J0) + c * P
    if J0 > 0:
        Pf = (-nu ** 2 + np.sqrt(nu ** 4 + 4 * nu ** 2 / J0)) / 2
    else:
        Pf = 1e6
    Ps = np.geomspace(1e-9, Pf, 4000)
    v = hP(Ps); i = int(np.argmin(v))
    lo = Ps[max(i - 1, 0)]; hi = Ps[min(i + 1, len(Ps) - 1)]
    r = minimize_scalar(hP, bounds=(lo, hi), method="bounded", options={"xatol": 1e-14})
    best = min(r.fun, v.min())
    return best, Pf
rows = []
nus = np.geomspace(1e-6, 1e-1, 26)
for J0 in (0.0, 0.01, 0.1, 1.0):
    vals = []
    for nu in nus:
        fl, Pf = floor(nu, J0)
        nux = 2 * 1.0 * J0 ** 1.5 / (1.0 * 1.0) if J0 > 0 else 0.0
        vals.append(dict(nu=nu, floor=fl, cPf=(Pf if J0 > 0 else np.nan), nu_x_asym=nux))
    # local slopes
    f = np.array([v["floor"] for v in vals])
    sl = np.diff(np.log(f)) / np.diff(np.log(nus))
    rows.append(dict(J0=J0, rows=vals, slope_small_nu=float(sl[:5].mean()), slope_large_nu=float(sl[-3:].mean())))
    print(f"J0={J0}: slope at small nu {sl[:5].mean():.4f}, at large nu {sl[-3:].mean():.4f}; nu_x(asym)={2*J0**1.5 if J0>0 else 0:.4g}")
    if J0 > 0:
        dev = [abs(v["floor"] - v["cPf"]) / v["cPf"] for v in vals if v["nu"] < 0.5 * v["nu_x_asym"]]
        print(f"   max rel. deviation floor vs c*P_f for nu<nu_x/2: {max(dev) if dev else float('nan'):.2e}")
        rows[-1]["max_rel_dev_floor_vs_cPf_below_half_nux"] = max(dev) if dev else None
res["floor_J0"] = rows

# (b) self-label price ratio and reading-tax factor
pr = {}
for g in (0.55, 0.9448):
    hh = h((1 + g) / 2)
    pr[g] = dict(tax=hh, ratio_label_vs_ref=(hh / (LN2 + hh)) ** (1 / 3), tax_factor_on_TDIII=((LN2 + hh) / LN2) ** (1 / 3))
    print("g", g, pr[g])
res["self_label"] = pr

# (c) body loop supplies free information about c g^2 (R5 moments); persistent tracking slope
rng = np.random.default_rng(3)
c_b, s_b, hgate = 0.9, 0.95, 0.5
def track(nu, alpha, T=60000, ntraj=400):
    g = np.full(ntraj, 0.7); B = np.ones(ntraj); kap_hat = np.full(ntraj, c_b * 0.49)
    yprev = None; Aprev = None; err2 = 0.0; n = 0
    for t in range(T):
        g = np.clip(g + nu * rng.normal(size=ntraj), 0.3, 0.98)
        B *= np.where(rng.random(ntraj) < (1 - c_b) / 2, -1, 1)
        y = np.where(rng.random(ntraj) < (1 + g * B) / 2, 1, -1)
        if yprev is not None:
            m = Aprev == 0
            kap_hat = np.where(m, (1 - alpha) * kap_hat + alpha * (y * yprev), kap_hat)
        A = (rng.random(ntraj) < hgate).astype(int)
        fl = (A == 1) & (y == -1) & (rng.random(ntraj) < s_b)
        B = np.where(fl, -B, B)
        yprev, Aprev = y, A
        if t > 5000:
            err2 += np.mean((kap_hat - c_b * g ** 2) ** 2); n += 1
    return err2 / n
tr = []
for nu in (1e-4, 3e-4, 1e-3, 3e-3):
    best = None
    for alpha in np.geomspace(2e-4, 0.3, 14):
        e = track(nu, alpha, T=30000, ntraj=300)
        if best is None or e < best[0]:
            best = (e, alpha)
    tr.append(dict(nu=nu, mse_kappa=best[0], alpha=best[1]))
    print(f"nu={nu}: best MSE of c g^2 estimate {best[0]:.3e} at alpha={best[1]:.3g}")
xs = np.log([t["nu"] for t in tr]); ys = np.log([t["mse_kappa"] for t in tr])
res["body_loop_tracking"] = dict(rows=tr, slope_mse_vs_nu=float(np.polyfit(xs, ys, 1)[0]))
print("log-log slope of MSE vs nu (body loop, free info):", res["body_loop_tracking"]["slope_mse_vs_nu"])
json.dump(res, open("out/c5_driftlaw_J0.json", "w"), indent=1, default=float)
