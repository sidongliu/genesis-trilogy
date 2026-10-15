"""Achievability for Corollary (detection before failure): residual-variance test under spherical Brownian drift, D=2.
In-frame residual e_m = y_m - rho v.x_m ~ N(0, s^2 + 2 rho^2 (1 - cos theta_m)); under calibration N(0, s^2).
Statistic S = mean of e_m^2 over the window; threshold set for type I error alpha by the exact chi-square law.
For each sigma find the smallest T (grid) with type II error <= beta; report T * sigma^{2/3} (should be ~constant)
and the ratio to the necessary time of Proposition detect (exact sums). Output: check_variance_test_output.json
"""
import json, math
import numpy as np
from scipy import stats
rng = np.random.default_rng(7)
rho, s, r = 1.0, 0.5, 1.0
alpha = beta = 0.05
out = {}
bound = {1e-2: 17, 1e-3: 68, 1e-4: 282}   # from check_detection_output.json
for sigma in [1e-2, 1e-3, 1e-4]:
    found = None
    for T in np.unique(np.round(np.geomspace(5, 40000, 160)).astype(int)):
        n = int(T*r)
        thr = s**2*stats.chi2.ppf(1 - alpha, n)/n
        nrep = 4000
        th = np.cumsum(math.sqrt(2*sigma/r)*rng.standard_normal((nrep, n)), axis=1)
        var = s**2 + 2*rho**2*(1 - np.cos(th))
        e2 = var*rng.chisquare(1, size=(nrep, n))
        Sstat = e2.mean(axis=1)
        b = float(np.mean(Sstat <= thr))
        if b <= beta:
            found = dict(T=int(T), type2=b, T_times_sigma23=float(T*sigma**(2/3)), ratio_to_bound=float(T/bound[sigma]))
            break
    out[str(sigma)] = found
json.dump(out, open('check_variance_test_output.json', 'w'), indent=1)
print(json.dumps(out, indent=1))
