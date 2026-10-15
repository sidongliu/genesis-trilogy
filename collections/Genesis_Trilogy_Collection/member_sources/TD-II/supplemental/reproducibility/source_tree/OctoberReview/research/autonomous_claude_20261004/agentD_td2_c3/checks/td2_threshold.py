"""Benefit threshold of the calibrated frame over the no-frame reader, from the exact
stationary formulas of td2_loop.py (no simulation).  Drift enters only through
kappa = 1 - lambda_1 = 1 - E[u_{t+1}.u_t].  Writes td2_threshold_output.json."""
import json, os, math
import numpy as np
from scipy.optimize import brentq
from td2_loop import risks, best_alpha

def gap(kap, D, b, rho, s, which):
    lam1 = 1 - kap
    aC = best_alpha((b - 1) / D, lam1, rho, s, D, D, which, floor=1e-4)
    aN = best_alpha(1.0, lam1, rho, s, D, b, which, floor=1e-4)
    return risks(aC, (b - 1) / D, lam1, rho, s, D, D)[which] - risks(aN, 1.0, lam1, rho, s, D, b)[which]

rho, s = 1.0, 0.5
out = []
for D in [4, 8, 16, 32]:
    for b in [2, 3, 5]:
        if b >= D:
            continue
        row = dict(D=D, b=b)
        for which, name in [(0, "gamma1"), (1, "shrunk")]:
            grid = np.geomspace(1e-7, 0.5, 120)
            g = [gap(k, D, b, rho, s, which) for k in grid]
            kc = None
            for i in range(len(grid) - 1):
                if g[i] < 0 <= g[i + 1]:
                    kc = brentq(lambda k: gap(k, D, b, rho, s, which), grid[i], grid[i + 1], xtol=1e-12)
                    break
            row[name + "_kappa_c"] = kc
            row[name + "_gap_at_1e-6"] = gap(1e-6, D, b, rho, s, which)
            row[name + "_scaled"] = None if kc is None else kc * D ** 2 / (b - 1)
        out.append(row)
        print(row, flush=True)
json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "td2_threshold_output.json"), "w"), indent=1)
