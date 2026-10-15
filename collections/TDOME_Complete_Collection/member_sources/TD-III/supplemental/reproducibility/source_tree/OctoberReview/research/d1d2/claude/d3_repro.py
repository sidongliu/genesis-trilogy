import sys, json, numpy as np
from multiprocessing import Pool
from platform_e0 import *
REPS, EPOCHS, BURN = 160, 1200, 200
def job(args):
    regime, speed, seed = args
    # shared initial calibration across the three policies, as in the spec
    g0, b0 = sensor(0, regime, speed); p0, m0 = ref_counts(np.random.default_rng(seed + 999), g0, b0, CFG['init_ref'], REPS)
    init = (quant((p0 - m0)/CFG['init_ref']), quant((p0 + m0)/CFG['init_ref'] - 1)); out = {}
    for pol in ('fixed', 'blockwise', 'persistent'):
        r, _ = run_d3(regime, pol, REPS, EPOCHS, BURN, seed + {'fixed': 1, 'blockwise': 2, 'persistent': 3}[pol], speed, init)
        out[pol] = {k: (float(v.mean()), se(v)) for k, v in r.items()}; out[pol]['_raw_world'] = r['world'].tolist()
    return (regime, speed), out
if __name__ == '__main__':
    jobs = [('drift', 1, 710001), ('static', 1, 710002), ('drift', 8, 710003), ('drift', 2, 710004), ('drift', 4, 710005), ('drift', 16, 710006)]
    with Pool(6) as P: res = P.map(job, jobs)
    json.dump({f"{k[0]}x{k[1]}": v for k, v in res}, open('d3_repro_output.json', 'w'))
    codex = {('drift', 1): (.0052832, .0036639, .0035985), ('static', 1): (.0034094, .0035859, .0035357), ('drift', 8): (.0045886, .0036229, .0038173)}
    print("regime(speed)   fixed                 blockwise             persistent           | Codex: fixed / blockwise / persistent")
    for (reg, sp), o in res:
        f = lambda p: f"{o[p]['world'][0]:.7f}+-{o[p]['world'][1]:.7f}"
        cx = codex.get((reg, sp)); cxs = " / ".join(f"{v:.7f}" for v in cx) if cx else "(not in Codex's table)"
        print(f"{reg:7s}x{sp:<3d}    {f('fixed')}   {f('blockwise')}   {f('persistent')}   | {cxs}")
    print("\npersistent - blockwise world loss (paired by trajectory is not available across independent streams; difference of means with combined s.e.):")
    for (reg, sp), o in res:
        dm = o['persistent']['world'][0] - o['blockwise']['world'][0]; ds = np.hypot(o['persistent']['world'][1], o['blockwise']['world'][1])
        print(f"   {reg}x{sp}: {dm:+.7f} +- {ds:.7f}   ({'persistent better' if dm < -2*ds else 'blockwise better' if dm > 2*ds else 'indistinguishable'})")
    print("\nestimation loss (xhat-x)^2/4 for drift x1:", {p: round(res[0][1][p]['est'][0], 7) for p in ('fixed', 'blockwise', 'persistent')}, " Codex: .0110985, .0074456, .0045691 (fixed, blockwise, persistent)")
