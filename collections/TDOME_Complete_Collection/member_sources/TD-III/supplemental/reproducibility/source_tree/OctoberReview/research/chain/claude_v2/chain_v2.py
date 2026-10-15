"""Claude 2026-10-04. chain_v2: the four-cell comparison {unweighted, weighted alignment} x {fixed w=1/2, convex authority} on ONE command grid. Correction identified by Codex: the command is u_i = -h * xhat_i with xhat_i already on the 1025-point grid, so |u_i| <= h exactly
 (chain_v1 used Q(-h xhat), which can round to -0.80078 at xhat=1). The joint command is u = w u_1 + (1-w) u_2, w = 1/2 (fixed) or s_1/(s_1+s_2) with s_i = clip(ghat_i/.4,0,1)^2 (convex; 1/2 if both are 0), so |u| <= h by construction. Peak |u| and the number of epochs with |u| > h are recorded in EVERY run (fault, no-fault, baseline).
chain_v1 = chain_v0 (frozen in ../claude/, unchanged) plus the strictly cap-conserving convex authority policy of Codex (research/synthesis/codex/round20_v1/review.md): after ONE exchange of xhat and ghat both observers compute the common weight
 w = s_1/(s_1+s_2), s_i = clip(ghat_i/.4,0,1)^2 (w=1/2 if both are 0) and the joint command is u = w u_1 + (1-w) u_2 with u_i = quant(clip(-h xhat_i, -1, 1)), so |u| <= h by construction. This implementation uses Codex's rule and the chain_v0 simulator; it is not independent code, and the max |u| actually used is recorded.
Two seed blocks are run to check stability. Original docstring of chain_v0 follows.
Claude 2026-10-04. Two-observer closed loop on the shared E0 world, following Codex's chain_spec.md (research/synthesis/codex/round19_v1/).
World: M=4096 bits, x = mean of bits. Joint transition (exact binomial per bit, exchangeable count form): P(+1 | prev=+1) = (1 + a + c u + r d)/2, P(+1 | prev=-1) = (1 - a + c u + r d)/2,
  with (a,c,r)=(.65,.25,.10), bath d_k = 0.6 c_k driven by the public external event c_k (flips w.p. p_sw=.01 per epoch), u = w u1 + (1-w) u2, w=1/2 fixed in all main comparisons.
Each observer i has its own legal instrument P(y=+1|x) = (1 + g_i x + b_i)/2 (|g|+|b|<=1), reads n_s world bits, n_r + n_r reference reads, n_s + 2 n_r = 512 per epoch; initial calibration 2048 reference pairs each (one-off).
Order per epoch (spec §'每轮顺序'): (1) memory fixes the reference share and rules; (2) instruments frozen, both read world and references on the SAME pre-action world, per-trajectory sampling (shape asserted);
(3) own references update own calibration, xhat_i = quant((o_i - bhat_i)/max(ghat_i, g_min)); (4) one synchronous exchange of xhat (and ghat for the weighted rule); the agreement update for NEXT epoch uses the pre-exchange values; the action u_i = clip(-h xhat_i, -1, 1) (quantised) uses own pre-exchange xhat;
(5) score the pre-action state; execute the joint channel. Truth (x, c, g, b, fault labels) goes to the scorer only.
Random streams are ADDRESSED by (seed, epoch, event) so that a fault in one instrument cannot shift the exogenous randomness of later epochs (c flips and world draws use their own stream per epoch). Feedback through the real world means fault and no-fault trajectories legitimately differ afterwards; matched pairs share only the exogenous draws.
Scoring (spec §'正确性必须跟随任务定义'): world loss x^2/4; estimation loss (xhat_i - x)^2/4; disagreement (xhat_1 - xhat_2)^2/4; error covariance E[e1 e2] (common mode = E[(e1+e2)^2]/4); binary public content = sign(x) on the covered set |x|>=delta (delta=.05) with coverage reported and common error rate.
External-event readout (spec §'保留原外部事件任务'): label c_k, available at epoch k+1: naive sign(xhat_{k+1}); record decoder sign(xhat_{k+1} - a xhat_k - c utilde_k) with the public joint command (11 bits/epoch per observer to publish) or own command only; and EWMA-pooled versions (beta=.1).
Communication ledger: 11 bits per transmitted scalar per epoch (1025-value grid); commands 11 bits each; weighted rule sends xhat and ghat (22 bits)."""
import numpy as np, json, sys
from multiprocessing import Pool
M, a, c_, r_, NQ = 4096, 0.65, 0.25, 0.10, 1024
def quant(z): return -1 + 2*np.rint((np.clip(z, -1, 1) + 1)*NQ/2)/NQ
PH = dict(g=(0.0, 1.7), b=(0.4, 2.9))
def legal(g, b, cap=0.98): return g, np.clip(b, -(cap - np.abs(g)), cap - np.abs(g))
def instr(k, i, kind, on, who=1, cs=1.0):
    g = 0.55 + 0.17*np.sin(2*np.pi*k/420 + PH['g'][i]) + 0.03*np.sin(2*np.pi*k/67 + PH['g'][i])
    b = cs*(0.10*np.sin(2*np.pi*k/250) + 0.03*np.sin(2*np.pi*k/53)) + 0.10*np.sin(2*np.pi*k/300 + PH['b'][i]) + 0.025*np.sin(2*np.pi*k/43 + PH['b'][i])
    if i == who and np.any(on):
        onf = np.asarray(on, float)
        if kind == 'bias': b = b + 0.30*onf
        elif kind == 'lowgain': g = np.where(onf > 0, 0.05, g)
        elif kind == 'zerogain': g = np.where(onf > 0, 0.0, g)
    return legal(g, b)
def stream(seed, k, s): return np.random.default_rng([seed, k + 1, s])
def refdraw(rng, n, g, b, reps):
    p = rng.binomial(n, (1 + g + b)/2, size=reps); m = rng.binomial(n, (1 - g + b)/2, size=reps)
    assert p.shape == (reps,) and m.shape == (reps,); return p, m
def simulate(rule, seed, kind=None, window=None, reps=800, epochs=3000, burn=200, p_sw=0.01, nr=(0, 0), lam=0.25, fault_on=True, who=1, h=0.8, w=0.5, gmin=0.30, delta=0.05, beta=0.1, act_mode=None):
    umax = np.zeros(1); uexc = np.zeros(1)
    ns = [512 - 2*nr[0], 512 - 2*nr[1]]
    w0 = np.random.default_rng([seed, 0, 9]).integers(800, 2400, size=reps) if window == 'random' else np.full(reps, 10**9)
    wlen = 100 if window == 'random' else 0
    rinit = stream(seed, -1, 0); n_plus = np.full(reps, M//2); c = np.where(rinit.random(reps) < 0.5, 1, -1)
    gh = [None, None]; bh = [None, None]
    for i in range(2):
        g, b = instr(0, i, kind, False, who); p, m = refdraw(stream(seed, -1, 1 + i), 2048, g, b, reps); gh[i] = quant((p - m)/2048); bh[i] = quant((p + m)/2048 - 1)
    NM = 10; tot = np.zeros((reps, NM)); win = np.zeros((reps, NM)); post = np.zeros((reps, NM)); ntot = 0
    ND = 7; dec = np.zeros((reps, ND, 2)); ndec = 0
    gprev = [gh[0].copy(), gh[1].copy()]; prev = None; ewn = [np.zeros(reps), np.zeros(reps)]; ewr = [np.zeros(reps), np.zeros(reps)]; ewa = [np.zeros(reps), np.zeros(reps)]
    for k in range(epochs):
        rc = stream(seed, k, 0); flip = rc.random(reps) < p_sw; c = np.where(flip, -c, c); d = 0.60*c; x = 2*n_plus/M - 1
        inwin = (k >= w0) & (k < w0 + wlen); inpost = (k >= w0 + wlen) & (k < w0 + wlen + 100); on = inwin & fault_on
        obs = []; xh = []
        for i in range(2):
            ri = stream(seed, k, 1 + i); g, b = instr(k, i, kind, on, who); assert np.all(np.abs(g) + np.abs(b) <= 1.0)
            cnt = ri.binomial(ns[i], (1 + g*x + b)/2); assert cnt.shape == (reps,); o = 2*cnt/ns[i] - 1; obs.append(o)
            if nr[i] > 0:
                p, m = refdraw(ri, nr[i], g, b, reps)
                gh[i] = quant(0.75*gh[i] + 0.25*(p - m)/nr[i]); bh[i] = quant(0.75*bh[i] + 0.25*((p + m)/nr[i] - 1))
            xh.append(quant((o - bh[i])/np.maximum(gh[i], gmin)))
        # action policies. None: spec baseline u_i = -h xhat_i. 'gate': own authority discounted by OWN reported gain. 'realloc': authority shared in proportion to reliability s_i = clip(ghat_i/.4,0,1)^2, own current and the partner's gain
        # from the PREVIOUS exchange (causal), so that a dead partner's authority passes to the healthy observer: u_i = -h xhat_i * 2 s_i/(s_i + s_j). Both are different policies from the spec baseline, reported separately.
        s_own = [np.clip(gh[i]/0.40, 0, 1)**2 for i in range(2)]
        if act_mode == 'gate': fa = s_own
        elif act_mode == 'realloc': fa = [2*s_own[i]/np.maximum(s_own[i] + np.clip(gprev[1 - i]/0.40, 0, 1)**2, 1e-9) for i in range(2)]
        else: fa = [1.0, 1.0]
        u = [-h*fa[i]*xh[i] for i in range(2)]                       # xh is on the 1025-point grid; no further rounding, |u_i| <= h
        if act_mode == 'convex':
            den_ = s_own[0] + s_own[1]; wv = np.where(den_ > 0, s_own[0]/np.maximum(den_, 1e-12), 0.5); uj = wv*u[0] + (1 - wv)*u[1]
        else: uj = w*u[0] + (1 - w)*u[1]
        umax[0] = max(umax[0], float(np.abs(uj).max())); uexc[0] += float((np.abs(uj) > h + 1e-12).sum())
        # external-event readouts for the PREVIOUS epoch's event, available now (decoders: 0 naive sign, 1 record(pub), 2 record(own), 3 EWMA naive, 4 EWMA record(pub), 5 EWMA 'knows a but ignores actions', 6 one-step 'knows a but ignores actions')
        if prev is not None and k > burn:
            cprev, xprev, ujprev, uprev = prev
            for i in range(2):
                I_pub = xh[i] - a*xprev[i] - c_*ujprev; I_own = xh[i] - a*xprev[i] - c_*uprev[i]; I_noact = xh[i] - a*xprev[i]
                ewn[i] = (1 - beta)*ewn[i] + beta*xh[i]; ewr[i] = (1 - beta)*ewr[i] + beta*I_pub; ewa[i] = (1 - beta)*ewa[i] + beta*I_noact
                outs = [np.sign(xh[i]), np.sign(I_pub), np.sign(I_own), np.sign(ewn[i]), np.sign(ewr[i]), np.sign(ewa[i]), np.sign(I_noact)]
                for j, s_ in enumerate(outs): dec[:, j, i] += (np.where(s_ == 0, 1, s_) == cprev)
            ndec += 1
        prev = (c.copy(), [xh[0].copy(), xh[1].copy()], uj.copy(), [u[0].copy(), u[1].copy()])
        ngh, nbh = list(gh), list(bh)
        if rule != 'none' and rule != 'ref':
            for i in range(2):
                src = xh[1 - i]; e = obs[i] - (gh[i]*src + bh[i])
                if rule == 'mixedw': nbh[i] = quant(bh[i] + lam*np.clip(gh[1 - i]/0.40, 0, 1)**2*e)
                else: nbh[i] = quant(bh[i] + lam*e)
        gh, bh = ngh, nbh; gprev = [gh[0].copy(), gh[1].copy()]
        if k >= burn:
            e1 = xh[0] - x; e2 = xh[1] - x; cov = np.abs(x) >= delta; s = np.where(x >= 0, 1, -1); c1 = np.where(xh[0] >= 0, 1, -1); c2 = np.where(xh[1] >= 0, 1, -1)
            vec = np.stack([x*x/4, e1*e1/4, e2*e2/4, (xh[0] - xh[1])**2/4, e1*e2/4, cov, cov & (c1 == s), cov & (c2 == s), cov & (c1 == c2) & (c1 != s), x], axis=1).astype(float)
            tot += vec; ntot += 1; win += vec*inwin[:, None]; post += vec*inpost[:, None]
        pp = (1 + a + c_*uj + r_*d)/2; pm = (1 - a + c_*uj + r_*d)/2; assert np.all((pp >= 0) & (pp <= 1) & (pm >= 0) & (pm <= 1))
        rw = stream(seed, k, 3); n_plus = rw.binomial(n_plus, pp) + rw.binomial(M - n_plus, pm)
    out = dict(tot=tot/ntot, dec=dec/max(ndec, 1), umax=float(umax[0]), uexc=float(uexc[0]))
    if window is not None: out['win'] = win/wlen; out['post'] = post/100.0
    return out
def run(args):
    key, kw = args; return key, simulate(**kw)
def se(v): return float(np.std(v, ddof=1)/np.sqrt(len(v)))
def ratio(num, den):
    r = num.sum()/max(den.sum(), 1e-12); return float(r), float(np.std(num - r*den, ddof=1)/(max(den.mean(), 1e-12)*np.sqrt(len(num))))
NAMES = ['world loss', 'est loss 1', 'est loss 2', 'disagreement', 'error cov']
RULES = {'unweighted alignment, fixed w=1/2': ('mixed', dict(nr=(16, 16))),
         'unweighted alignment, convex authority': ('mixed', dict(nr=(16, 16), act_mode='convex')),
         'weighted alignment, fixed w=1/2': ('mixedw', dict(nr=(16, 16))),
         'weighted alignment, convex authority': ('mixedw', dict(nr=(16, 16), act_mode='convex')),
         'references only, fixed w=1/2 (no comm.)': ('ref', dict(nr=(16, 16)))}
REPS = 800
if __name__ == '__main__':
    jobs = []
    for blk, off in (('A', 0), ('B', 1000), ('C', 2000)):
        for i, (name, (rule, kw)) in enumerate(RULES.items()):
            jobs.append((('base', blk, name), dict(rule=rule, seed=2500000 + off + 10*i, reps=REPS, **kw)))
            for j, kd in enumerate(('bias', 'lowgain', 'zerogain')):
                sd = 2600000 + off + 100*i + 7*j
                jobs.append((('fault', blk, name, kd), dict(rule=rule, seed=sd, kind=kd, window='random', reps=REPS, **kw)))
                jobs.append((('nofault', blk, name, kd), dict(rule=rule, seed=sd, kind=kd, window='random', fault_on=False, reps=REPS, **kw)))
    with Pool(8) as P: res = dict(P.map(run, jobs))
    BL = 'ABC'; L = []
    def Pr(s=""): L.append(s)
    Pr(f"# chain_v2 output ({REPS} trajectories per run, three seed blocks A/B/C, 3000 epochs, burn-in 200, fault on observer 2 for 100 epochs at a per-trajectory random onset in [800,2400)); differences are fault - no fault inside the window; units 1e-3")
    Pr("\n## command cap h=0.8: largest |u| applied and number of (trajectory, epoch) pairs with |u| > h, over ALL runs of each policy (baseline, fault, no-fault)")
    out = dict(base={}, cap={}, faults={}, interaction={})
    for name in RULES:
        runs = [v for k, v in res.items() if k[0] in ('base', 'fault', 'nofault') and k[2 if k[0] == 'base' else 2] == name]
        um = max(v['umax'] for v in runs); ex = sum(v['uexc'] for v in runs)
        out['cap'][name] = dict(max_abs_u=um, exceed_count=ex, runs=len(runs))
        Pr(f"{name:48s} max|u| {um:.6f}   exceed count {int(ex)}   runs {len(runs)}")
    Pr("\n## no-fault world loss (mean over trajectories; s.e. about 3e-6 per block)")
    for name in RULES:
        wl = [res[('base', b, name)]['tot'][:, 0].mean() for b in BL]; out['base'][name] = wl
        Pr(f"{name:48s} " + "  ".join(f"{b}: {w:.6f}" for b, w in zip(BL, wl)))
    Pr("\n## faults: dWorld (x1e-3) and dEst1 (x1e-3, healthy observer) per block with s.e.")
    Pr(f"{'policy':48s} {'fault':9s} " + " ".join(f"{'dWorld '+b:>14s}" for b in BL) + " " + " ".join(f"{'dEst1 '+b:>14s}" for b in BL))
    mean_dw = {}
    for name in RULES:
        for kd in ('bias', 'lowgain', 'zerogain'):
            row = {}
            for b in BL:
                f = res[('fault', b, name, kd)]['win']; n = res[('nofault', b, name, kd)]['win']
                row['dW' + b] = (float((f[:, 0] - n[:, 0]).mean()), se(f[:, 0] - n[:, 0])); row['dE' + b] = (float((f[:, 1] - n[:, 1]).mean()), se(f[:, 1] - n[:, 1]))
            out['faults'][f"{name}|{kd}"] = row; mean_dw[(name, kd)] = sum(row['dW' + b][0] for b in BL)/3
            Pr(f"{name:48s} {kd:9s} " + " ".join(f"{1e3*row['dW'+b][0]:+7.3f}±{1e3*row['dW'+b][1]:.3f}" for b in BL) + " " + " ".join(f"{1e3*row['dE'+b][0]:+7.3f}±{1e3*row['dE'+b][1]:.3f}" for b in BL))
    Pr("\n## two conditional effects and their interaction on the 2x2 cells (mean dWorld over the three blocks, x1e-3)")
    cells = {('u', 'f'): 'unweighted alignment, fixed w=1/2', ('u', 'c'): 'unweighted alignment, convex authority', ('w', 'f'): 'weighted alignment, fixed w=1/2', ('w', 'c'): 'weighted alignment, convex authority'}
    for kd in ('bias', 'lowgain', 'zerogain'):
        m = {k: 1e3*mean_dw[(v, kd)] for k, v in cells.items()}
        authority_given_weighted = m[('w', 'c')] - m[('w', 'f')]; authority_given_unweighted = m[('u', 'c')] - m[('u', 'f')]
        weighting_given_fixed = m[('w', 'f')] - m[('u', 'f')]; weighting_given_convex = m[('w', 'c')] - m[('u', 'c')]
        inter = authority_given_weighted - authority_given_unweighted
        out['interaction'][kd] = dict(cells={f'{k[0]}/{k[1]}': v for k, v in m.items()}, authority_given_weighted=authority_given_weighted, authority_given_unweighted=authority_given_unweighted, weighting_given_fixed=weighting_given_fixed, weighting_given_convex=weighting_given_convex, interaction=inter)
        Pr(f"{kd:9s} cells: u/f {m[('u','f')]:+.3f}  u/c {m[('u','c')]:+.3f}  w/f {m[('w','f')]:+.3f}  w/c {m[('w','c')]:+.3f} | authority effect: weighted {authority_given_weighted:+.3f}, unweighted {authority_given_unweighted:+.3f} | weighting effect: fixed {weighting_given_fixed:+.3f}, convex {weighting_given_convex:+.3f} | interaction {inter:+.3f}")
    open('chain_v2_output.txt', 'w').write("\n".join(L) + "\n"); json.dump(out, open('chain_v2_output.json', 'w')); print("\n".join(L))
