"""Claude 2026-10-04. Independent implementation of the shared E0 platform (spec from Codex's d3/codex/v0/report.md; the report and then its code were consulted, so independence is partial: the world here is simulated BIT BY BIT (4096 explicit +-1 bits per trajectory, explicit random read
indices), not by the exchangeable count compression, which also tests that compression).
World: M bits z_j in {-1,+1}, x = mean z. Instrument: read a random bit j (index not returned), outcome y=+-1 with P(y=+1) = (1 + g z_j + b)/2.
Reference reads: known inputs z=+-1 through the same instrument. Actuator/bath channel on every bit: keep with prob a, refresh to tau_u with prob c,
refresh to tau_d with prob r (tau_v = (I + vZ)/2). The policy only sees counts and its own stored (quantised) estimates."""
import numpy as np
CFG = dict(M=4096, a=0.65, c=0.25, r=0.10, h=0.8, g_min=0.30, nq=1024, init_ref=2048, ns_fixed=512, ns_adapt=256, nr_adapt=128, alpha_persist=0.25)
def quant(z, nq=CFG['nq']): return -1 + 2*np.rint((np.clip(z, -1, 1) + 1)*nq/2)/nq
def sensor(k, regime, speed=1):
    if regime == 'static': return 0.55, 0.0
    kk = k*speed
    return (0.55 + 0.17*np.sin(2*np.pi*kk/420) + 0.03*np.sin(2*np.pi*kk/67), 0.14*np.sin(2*np.pi*kk/300) + 0.025*np.sin(2*np.pi*kk/43))
def bath(k): return 0.60 + 0.25*np.sin(2*np.pi*k/177)
def ref_counts(rng, g, b, n, size):
    return rng.binomial(n, (1 + g + b)/2, size=size), rng.binomial(n, (1 - g + b)/2, size=size)
def read_counts(rng, bits, g, b, ns):
    reps, M = bits.shape; idx = rng.integers(0, M, size=(reps, ns)); z = bits[np.arange(reps)[:, None], idx]
    return (rng.random(z.shape) < (1 + g*z + b)/2).sum(axis=1)
def world_step(rng, bits, u, d, cfg=CFG):
    reps, M = bits.shape; U = rng.random((reps, M)); V = rng.random((reps, M))
    pu = ((1 + u)/2)[:, None]; pd = (1 + d)/2
    ctrl = (U >= cfg['a']) & (U < cfg['a'] + cfg['c']); env = U >= cfg['a'] + cfg['c']
    return np.where(ctrl, np.where(V < pu, 1, -1), np.where(env, np.where(V < pd, 1, -1), bits)).astype(np.int8)
def run_d3(regime, policy, reps, epochs, burn, seed, speed=1, init=None, cfg=CFG):
    rng = np.random.default_rng(seed); M = cfg['M']
    bits = np.ones((reps, M), np.int8); bits[:, int(0.6*M):] = -1
    if init is None:
        g0, b0 = sensor(0, regime, speed); p0, m0 = ref_counts(np.random.default_rng(seed + 999), g0, b0, cfg['init_ref'], reps)
        gh = quant((p0 - m0)/cfg['init_ref']); bh = quant((p0 + m0)/cfg['init_ref'] - 1)
    else: gh, bh = init[0].copy(), init[1].copy()
    acc = dict(world=np.zeros(reps), est=np.zeros(reps), gerr=np.zeros(reps), berr=np.zeros(reps), ctrl=np.zeros(reps))
    for k in range(epochs):
        g, b = sensor(k, regime, speed); d = bath(k); x = bits.mean(axis=1)
        if policy == 'fixed': ns = cfg['ns_fixed']
        else:
            ns = cfg['ns_adapt']; nr = cfg['nr_adapt']; p, m = ref_counts(rng, g, b, nr, reps)
            al = 1.0 if policy == 'blockwise' else cfg['alpha_persist']
            gh = quant((1 - al)*gh + al*(p - m)/nr); bh = quant((1 - al)*bh + al*((p + m)/nr - 1))
        cnt = read_counts(rng, bits, g, b, ns); obs = 2*cnt/ns - 1
        xh = quant((obs - bh)/np.maximum(gh, cfg['g_min'])); u = -cfg['h']*xh
        if k >= burn:
            acc['world'] += x*x/4; acc['est'] += (xh - x)**2/4; acc['gerr'] += (gh - g)**2; acc['berr'] += (bh - b)**2; acc['ctrl'] += u*u
        bits = world_step(rng, bits, u, d, cfg)
    n = epochs - burn
    return {kk: vv/n for kk, vv in acc.items()}, (gh, bh)
def se(v): return float(np.std(v, ddof=1)/np.sqrt(len(v)))
