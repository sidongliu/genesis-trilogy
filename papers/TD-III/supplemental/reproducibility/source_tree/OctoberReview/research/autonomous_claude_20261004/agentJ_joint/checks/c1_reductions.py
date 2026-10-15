"""c1: reductions of the joint model to TD-I, QR-B, IMP-CHAIN; flip-then-harvest income; copy vs blank purity."""
import json, itertools
import numpy as np
from common import *

out = {}
# TD-I cycle numbers (Prop cycle): five- and seven-memory fragments, c=0.8, Helstrom reads
for n in (1, 5, 7):
    r = 0.8 ** n
    g = helstrom_gain(r)
    I = info_XM(g)
    chi = chi_M_G(g, r)
    out[f"TD-I n={n}"] = dict(r=r, g=g, I_XM=I, chi_X_EF=Hp(0.5, r), chi_M_EG=chi,
                              net_with_copy=I - (LN2 - chi), net_without=I - LN2)
# expected: n=1 I=0.1927; n=5 I=0.5668 chi=0.5231 net 0.3968 / -0.1263; n=7 net 0.5511 / -0.0611

# flip-then-harvest: exact enumeration of P(X'=+1) and income ln2-h(q) vs I(X:M)
rows = []
for g in (0.3, 0.55, 0.9448):
    for s in (1.0, 0.99, 0.9, 0.7):
        p = 0.0
        for x, m, fl in itertools.product((1, -1), (1, -1), (0, 1)):
            px = 0.5
            pm = (1 + g * m * x) / 2
            if m == 1:
                if fl == 1:
                    continue
                pf = 1.0; xp = x
            else:
                pf = s if fl == 1 else 1 - s
                xp = -x if fl == 1 else x
            if xp == 1:
                p += px * pm * pf
        q = (1 + s * g) / 2
        rows.append(dict(g=g, s=s, q_enum=p, q_formula=q, income=LN2 - h(q), I_XM=info_XM(g)))
out["flip_then_harvest"] = rows
out["flip_then_harvest_maxdev"] = max(abs(r["q_enum"] - r["q_formula"]) for r in rows)

# QR-B coherent drive failures and persistence loss (Table: nu=0.8,c=0.9,h=0.4,q32=0.0404888)
qrb = {}
for N in (1, 4, 16, 64):
    e = eps_coherent(N); s = 1 - e
    nu, c, hh, q = 0.8, 0.9, 0.4, 0.0404888
    Lam = c * (1 - hh * s); B = c * hh * s * nu
    qrb[N] = dict(eps=e, L_inf_q32=0.5 - (1 - q) * B / (2 * (1 - Lam)), L_inf_q_half=0.5 - 0.5 * B / (2 * (1 - Lam)),
                  eps_opt_bound=eps_optimal_bound(N), eps_gauss=eps_gaussian(N))
out["QR-B"] = qrb
# expected eps: 0.354394 0.132699 0.037104 0.009546 ; L(q32): 0.231654 0.209300 0.202127 0.200259

# IMP-CHAIN exact-flip calibrated loss 0.199631 (q32) and known-interface 0.186957
nu, c, hh = 0.8, 0.9, 0.4
lam = c * (1 - hh); b = c * hh * nu
out["IMP-CHAIN"] = dict(L_q32=0.5 - (1 - 0.040489) * b / (2 * (1 - lam)), L_known=0.5 - b / (2 * (1 - lam)))

# copy vs blank purity (R8)
r5 = 0.8 ** 5
out["R8 copy vs blank"] = dict(copy_chi_M_EG=chi_M_G(helstrom_gain(r5), r5), copy_memory_purity=5 * LN2,
                               blank_absorbs=LN2, ratio_per_memory=(LN2) / (chi_M_G(helstrom_gain(r5), r5) / 5))

json.dump(out, open("out/c1_reductions.json", "w"), indent=1, default=float)
for k, v in out.items():
    print(k, v if not isinstance(v, list) else f"[{len(v)} rows]")
