"""c7: Proposition R5 by exact enumeration: closed-loop moments identify c, g^2, s; frame mismatch kills them."""
import json, itertools
import numpy as np

def moments(g, sigma, d, s, h, pB, matched=True):
    c = 1 - 2 * d
    def py(y, B):
        return (1 + sigma * g * y * B) / 2 if matched else 0.5
    # enumerate B_t, Y_t, A_t, flip coin, bath, Y_{t+1}, A_{t+1}, flip, bath, Y_{t+2}
    E = {"YY_A0": [0, 0], "YY_A1": [0, 0], "YY2_00": [0, 0], "EY": 0}
    for B in (1, -1):
        pb = pB if B == 1 else 1 - pB
        for Y in (1, -1):
            p1 = pb * py(Y, B)
            E["EY"] += p1 * Y
            for A in (0, 1):
                pa = h if A else 1 - h
                for phi in (0, 1):
                    if A == 1 and Y == -1:
                        pf = s if phi else 1 - s
                    else:
                        if phi: continue
                        pf = 1.0
                    B1 = -B if phi else B
                    for xi in (1, -1):
                        px = 1 - d if xi == 1 else d
                        B2 = B1 * xi
                        for Y2 in (1, -1):
                            p2 = p1 * pa * pf * px * py(Y2, B2)
                            key = "YY_A1" if A else "YY_A0"
                            E[key][0] += p2 * Y2 * Y; E[key][1] += p2
                            if A == 0:
                                for A2 in (0, 1):
                                    if A2 == 1: continue
                                    pa2 = 1 - h
                                    for xi2 in (1, -1):
                                        px2 = 1 - d if xi2 == 1 else d
                                        B3 = B2 * xi2
                                        for Y3 in (1, -1):
                                            p3 = p2 * pa2 * px2 * py(Y3, B3)
                                            E["YY2_00"][0] += p3 * Y3 * Y; E["YY2_00"][1] += p3
    out = dict(EY=E["EY"], YY_A0=E["YY_A0"][0] / E["YY_A0"][1], YY_A1=E["YY_A1"][0] / E["YY_A1"][1], YY2_00=E["YY2_00"][0] / E["YY2_00"][1])
    pred = dict(YY_A0=c * g * g, YY_A1=c * g * g * (1 - s) + c * s * E["EY"], YY2_00=c * c * g * g)
    return out, pred

rng = np.random.default_rng(5)
rows = []; maxdev = 0.0
for k in range(30):
    g, d, s, h, pB = rng.uniform(0.1, 0.95), rng.uniform(0.01, 0.4), rng.uniform(0.3, 1), rng.uniform(0.1, 0.9), rng.uniform(0, 1)
    sigma = rng.choice([-1, 1])
    o, p = moments(g, sigma, d, s, h, pB)
    dev = max(abs(o[k2] - p[k2]) for k2 in p)
    maxdev = max(maxdev, dev)
    # inversion of the identities: recover c, g^2, s
    c_hat = o["YY2_00"] / o["YY_A0"]; g2_hat = o["YY_A0"] / c_hat
    s_hat = (o["YY_A0"] - o["YY_A1"]) / (o["YY_A0"] - c_hat * o["EY"])
    rows.append(dict(g=g, sigma=int(sigma), d=d, s=s, h=h, pB=pB, dev=dev, c_err=abs(c_hat - (1 - 2 * d)), g2_err=abs(g2_hat - g * g), s_err=abs(s_hat - s)))
    om, _ = moments(g, sigma, d, s, h, pB, matched=False)
    rows[-1]["mismatched"] = om
res = dict(max_dev_identities=maxdev, max_c_err=max(r["c_err"] for r in rows), max_g2_err=max(r["g2_err"] for r in rows),
           max_s_err=max(r["s_err"] for r in rows), max_abs_mismatched_moment=max(max(abs(r["mismatched"][k]) for k in ("YY_A0", "YY2_00")) for r in rows), rows=rows)
print({k: v for k, v in res.items() if k != "rows"})
json.dump(res, open("out/c7_closedloop_moments.json", "w"), indent=1, default=float)
