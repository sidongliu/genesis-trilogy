"""Claude 2026-10-04. D2 v3 corrects the online-reference sampling defect identified by Codex in d1d2/claude/d2_public.py and d1d2/claude_v2/d2_rules.py.
Sampling defect: in those files the instrument parameters (g,b) of the D2 world are scalars, so `binomial(nr, p)` without `size` returned ONE positive and ONE negative reference
count per epoch, broadcast to all trajectories: every trajectory with online references shared the same calibration noise path, so the 400 trajectories were not independent and the reported
standard errors were not unconditional errors. (D1 is unaffected: its instrument parameters are per-trajectory arrays; D3 platform_e0 passes size=reps.) Fix: every online reference draw has shape (reps,),
asserted in code. Everything else is as in d2_rules.py (rules, independent per-observer random streams, matched controls) except:
 * fault 'dead' is split into 'lowgain' (g=0.05) and 'zerogain' (g=0); decoding floor GMIN is a parameter (default 0.30 = D3 platform) with an ablation at 0.05;
 * the anchor position is identified separately: references at (n1,n2) in {(4,0),(0,4),(4,4)};
 * communication ledger: each transmitted scalar field is 11 bits (1025 output values); bias/gated/soft/leader send xhat (11 bits), gainbias and mixedw send xhat and ghat (22 bits);
 * the v1 headline table (reference-budget scan, mutual, mixed, common-contamination scan) is recomputed here with correct sampling;
 * 800 independent trajectories per configuration.
FAULT WINDOW (changed after the first v3 run): 100 epochs starting at a per-trajectory random epoch in [800,2400), the same across fault / no-fault / isolated runs, so results average over the phases of the drifting instruments. The first v3 run used the fixed window [1500,1600) (d2_v3_fixedwindow.py, d2_v3_output_fixedwindow.txt); there the who=1 and who=0 results differ strongly because the instruments are in different drift phases in that window.
Resources per observer per epoch: n_s + 2 n_r = 512 reads; initial calibration 2048 reference pairs per observer (one-off, 4096 reads).
DECOMPOSITION (added after the first v3 run; the first run's numbers are unchanged, same seeds): dC = C_other(no fault) - C_other(fault) splits exactly into
  loss of support  = C_other(no fault) - C_other(fault, receiver ignores the partner inside the window)   [the partner no longer helps]
  harm of coupling = C_other(fault, receiver ignores the partner) - C_other(fault, coupled)              [listening to the faulted partner is worse than ignoring it].
Contagion = C_other(no fault) - C_other(fault), matched by seed, for the observer that is NOT faulted. In this no-control world the faulted observer's instrument cannot change the world, and the observers' random streams are separate,
so paired differences for rules without a path from the faulted observer to the other are exactly zero by construction (as before); once actions feed back (chain), that exactness is not expected."""
import numpy as np, json, sys
from multiprocessing import Pool
M, a, r_, NQ = 4096, 0.65, 0.10, 1024
def quant(z): return -1 + 2*np.rint((np.clip(z, -1, 1) + 1)*NQ/2)/NQ
PH = dict(g=(0.0, 1.7), b=(0.4, 2.9))
def legal(g, b, cap=0.98): return g, np.clip(b, -(cap - np.abs(g)), cap - np.abs(g))
def instr(k, i, kind, on, who=1, cs=1.0):
    g = 0.55 + 0.17*np.sin(2*np.pi*k/420 + PH['g'][i]) + 0.03*np.sin(2*np.pi*k/67 + PH['g'][i])
    b = cs*(0.10*np.sin(2*np.pi*k/250) + 0.03*np.sin(2*np.pi*k/53)) + 0.10*np.sin(2*np.pi*k/300 + PH['b'][i]) + 0.025*np.sin(2*np.pi*k/43 + PH['b'][i])
    if i == who and np.any(on):
        onf = np.asarray(on, float)
        if kind == 'bias': b = b + 0.30*onf
        elif kind == 'gain': g = g*(1 - 0.5*onf)
        elif kind == 'lowgain': g = np.where(onf > 0, 0.05, g)
        elif kind == 'zerogain': g = np.where(onf > 0, 0.0, g)
    return legal(g, b)
def refdraw(rng, n, g, b, reps):
    p = rng.binomial(n, (1 + g + b)/2, size=reps); m = rng.binomial(n, (1 - g + b)/2, size=reps)
    assert p.shape == (reps,) and m.shape == (reps,); return p, m
def simulate(rule, seed, kind=None, window=None, reps=800, epochs=1800, burn=200, p_sw=0.01, nr=(0, 0), lam=0.25, lamg=0.25, tau=0.12, fault_on=True, who=1, cs=1.0, gmin=0.30, isolate=False):
    ns = [512 - 2*nr[0], 512 - 2*nr[1]]
    ss = np.random.SeedSequence(seed).spawn(4); rw = np.random.default_rng(ss[0]); ro = [np.random.default_rng(ss[1]), np.random.default_rng(ss[2])]
    if window == 'random': w0 = np.random.default_rng(ss[3]).integers(800, 2400, size=reps)      # per-trajectory fault onset, identical across fault / no-fault / isolated runs
    elif window is not None: w0 = np.full(reps, window[0])
    else: w0 = np.full(reps, 10**9)
    wlen = 100 if window == 'random' else (window[1] - window[0] if window is not None else 0)
    n_plus = np.full(reps, M//2); c = np.where(rw.random(reps) < 0.5, 1, -1)
    gh = [None, None]; bh = [None, None]
    for i in range(2):
        g, b = instr(0, i, kind, False, who, cs); p, m = refdraw(ro[i], 2048, g, b, reps)
        gh[i] = quant((p - m)/2048); bh[i] = quant((p + m)/2048 - 1)
    tot = np.zeros((reps, 4)); ntot = 0; win = np.zeros((reps, 4)); nwin = 0; post = np.zeros((reps, 4)); npost = 0
    for k in range(epochs):
        flip = rw.random(reps) < p_sw; c = np.where(flip, -c, c); d = 0.60*c; x = 2*n_plus/M - 1
        inwin = (k >= w0) & (k < w0 + wlen); inpost = (k >= w0 + wlen) & (k < w0 + wlen + 100); on = inwin & fault_on; obs = []; xh = []
        for i in range(2):
            g, b = instr(k, i, kind, on, who, cs); assert np.all(np.abs(g) + np.abs(b) <= 1.0)
            cnt = ro[i].binomial(ns[i], (1 + g*x + b)/2); assert cnt.shape == (reps,); o = 2*cnt/ns[i] - 1; obs.append(o)
            if nr[i] > 0:
                p, m = refdraw(ro[i], nr[i], g, b, reps)
                gh[i] = quant(0.75*gh[i] + 0.25*(p - m)/nr[i]); bh[i] = quant(0.75*bh[i] + 0.25*((p + m)/nr[i] - 1))
            xh.append(quant((o - bh[i])/np.maximum(gh[i], gmin)))
        ngh, nbh = list(gh), list(bh)
        for i in range(2):
            if rule == 'ref' or (rule == 'leader' and i == 0): continue
            src = xh[0] if rule == 'leader' else xh[1 - i]
            e = obs[i] - (gh[i]*src + bh[i])
            if rule in ('bias', 'mixed', 'leader'): nbh[i] = quant(bh[i] + lam*e)
            elif rule == 'mixedw': nbh[i] = quant(bh[i] + lam*np.clip(gh[1 - i]/0.40, 0, 1)**2*e)
            elif rule == 'gated': nbh[i] = quant(bh[i] + lam*e*(np.abs(e) <= tau))
            elif rule == 'soft': nbh[i] = quant(bh[i] + lam*e/(1 + (e/tau)**2))
            elif rule == 'gainbias':
                den = src**2 + 1.0; nbh[i] = quant(bh[i] + lam*e/den); ngh[i] = quant(gh[i] + lamg*e*src/den)
        if isolate:                                                       # receiver ignores the partner while the (per-trajectory) fault window is open
            r = 1 - who; nbh[r] = np.where(inwin, bh[r], nbh[r]); ngh[r] = np.where(inwin, gh[r], ngh[r])
        gh, bh = ngh, nbh
        ch = [np.where(xh[i] >= 0, 1, -1) for i in range(2)]
        if k >= burn:
            vec = np.stack([ch[0] == ch[1], ch[0] == c, ch[1] == c, (ch[0] == ch[1]) & (ch[0] != c)], axis=1).astype(float)
            tot += vec; ntot += 1
            win += vec*inwin[:, None]; post += vec*inpost[:, None]
        n_plus = rw.binomial(n_plus, (1 + a + r_*d)/2) + rw.binomial(M - n_plus, (1 - a + r_*d)/2)
    out = dict(tot=tot/ntot)
    if window is not None: out['win'] = win/wlen; out['post'] = post/100.0
    return out
def run(args):
    key, kw = args; return key, simulate(**kw)
def se(v): return float(np.std(v, ddof=1)/np.sqrt(len(v)))
W = 'random'; REPS = 800      # fault window: 100 epochs starting at a per-trajectory random epoch in [800, 2400)
BITS = dict(bias=11, gated=11, soft=11, leader=11, mixed=11, gainbias=11, mixedw=22, ref=0)   # bits per transmitting observer per epoch
CONFIGS = {
 'bias (mutual, no refs)': ('bias', dict()), 'gated tau=.06': ('gated', dict(tau=0.06)), 'gated tau=.12': ('gated', dict(tau=0.12)), 'soft tau=.12': ('soft', dict(tau=0.12)),
 'gainbias (lamg=.02)': ('gainbias', dict(lamg=0.02)),
 'ref n_r=64 (no comm.)': ('ref', dict(nr=(64, 64))), 'leader (n_r=64)': ('leader', dict(nr=(64, 0))),
 'mixed (4,4)': ('mixed', dict(nr=(4, 4))), 'mixed (16,16)': ('mixed', dict(nr=(16, 16))), 'mixed (64,64)': ('mixed', dict(nr=(64, 64))),
 'mixed (4,0)': ('mixed', dict(nr=(4, 0))), 'mixed (0,4)': ('mixed', dict(nr=(0, 4))),
 'mixedw (4,4)': ('mixedw', dict(nr=(4, 4))), 'mixedw (16,16)': ('mixedw', dict(nr=(16, 16))), 'mixedw (64,64)': ('mixedw', dict(nr=(64, 64))),
 'mixedw (4,0)': ('mixedw', dict(nr=(4, 0))), 'mixedw (0,4)': ('mixedw', dict(nr=(0, 4))),
}
KINDS = ('bias', 'gain', 'lowgain', 'zerogain')
if __name__ == '__main__':
    jobs = []
    for i, (name, (rule, kw)) in enumerate(CONFIGS.items()):
        jobs.append((('base', name), dict(rule=rule, seed=71000 + 10*i, epochs=3000, reps=REPS, **kw)))   # (3000 epochs; the fault runs below use 3000 epochs as well so every random window plus its 100 post epochs fits)
        for j, kd in enumerate(KINDS):
            for who in (1, 0):
                sd = 81000 + 100*i + 7*j + 3*who
                jobs.append((('fault', name, kd, who), dict(rule=rule, seed=sd, kind=kd, window=W, who=who, reps=REPS, epochs=3000, **kw)))
                jobs.append((('nofault', name, kd, who), dict(rule=rule, seed=sd, kind=kd, window=W, fault_on=False, who=who, reps=REPS, epochs=3000, **kw)))
                jobs.append((('isolated', name, kd, who), dict(rule=rule, seed=sd, kind=kd, window=W, who=who, reps=REPS, epochs=3000, isolate=True, **kw)))
    # decoding-floor ablation (lowgain / zerogain, floor 0.05 instead of 0.30)
    for name in ('mixed (16,16)', 'mixedw (16,16)'):
        rule, kw = CONFIGS[name]
        for j, kd in enumerate(('lowgain', 'zerogain')):
            sd = 91000 + 100*list(CONFIGS).index(name) + 7*j
            jobs.append((('ablate_fault', name, kd), dict(rule=rule, seed=sd, kind=kd, window=W, who=1, gmin=0.05, reps=REPS, epochs=3000, **kw)))
            jobs.append((('ablate_nofault', name, kd), dict(rule=rule, seed=sd, kind=kd, window=W, fault_on=False, who=1, gmin=0.05, reps=REPS, epochs=3000, **kw)))
    # v1 headline table recomputed with correct sampling: reference-budget scan, mutual, mixed, common-contamination scan
    for nr in (0, 4, 16, 32, 64, 128): jobs.append((('v1scan', 'ref', nr), dict(rule='ref', seed=72000 + nr, epochs=3000, reps=REPS, nr=(nr, nr))))
    for nr in (4, 16, 32): jobs.append((('v1scan', 'mixed', nr), dict(rule='mixed', seed=73000 + nr, epochs=3000, reps=REPS, nr=(nr, nr))))
    jobs.append((('v1scan', 'mutual', 0), dict(rule='bias', seed=74000, epochs=3000, reps=REPS)))
    for cs in (0.0, 1.0, 2.0):
        jobs.append((('v1common', 'mutual', cs), dict(rule='bias', seed=75000 + int(10*cs), epochs=3000, reps=REPS, cs=cs)))
        jobs.append((('v1common', 'ref64', cs), dict(rule='ref', seed=76000 + int(10*cs), epochs=3000, reps=REPS, nr=(64, 64), cs=cs)))
    with Pool(8) as P: res = dict(P.map(run, jobs))
    L = []
    def P_(s=""): L.append(s)
    P_(f"# D2 v3 output ({REPS} independent trajectories per configuration; fault window = 100 epochs at a per-trajectory random onset in [800,2400); correct online reference sampling; independent per-observer random streams; matched no-fault controls). Per observer per epoch n_s+2n_r=512 reads; initial 2048 reference pairs per observer (one-off); communication: 11 bits per transmitted scalar per transmitting observer per epoch ('pair bits' = total for the pair per epoch).")
    P_("\n## baseline (no fault), 3000 epochs, burn-in 200: mean (s.e. over trajectories)")
    P_(f"{'configuration':26s} {'pair bits':>9s} {'A12':>8s} {'C1':>16s} {'C2':>16s} {'E':>16s}")
    base = {}
    for name, (rule, kw) in CONFIGS.items():
        t = res[('base', name)]['tot']; base[name] = {k: float(t[:, j].mean()) for j, k in enumerate(('A12', 'C1', 'C2', 'E'))}
        P_(f"{name:26s} {BITS[rule]*(2 if rule not in ('leader','ref') else (1 if rule=='leader' else 0)):9d} {t[:,0].mean():8.4f} {t[:,1].mean():8.4f}({se(t[:,1]):.4f}) {t[:,2].mean():8.4f}({se(t[:,2]):.4f}) {t[:,3].mean():8.4f}({se(t[:,3]):.4f})")
    P_("\n## v1 headline table recomputed with correct sampling (3000 epochs)")
    P_(f"{'configuration':22s} {'A12':>8s} {'C1':>8s} {'C2':>8s} {'E':>8s} {'C1 s.e.':>8s}")
    v1 = {}
    for key in [k for k in res if k[0] == 'v1scan']:
        t = res[key]['tot']; v1['|'.join(map(str, key[1:]))] = {k: float(t[:, j].mean()) for j, k in enumerate(('A12', 'C1', 'C2', 'E'))}
        P_(f"{key[1] + ' nr=' + str(key[2]):22s} {t[:,0].mean():8.4f} {t[:,1].mean():8.4f} {t[:,2].mean():8.4f} {t[:,3].mean():8.4f} {se(t[:,1]):8.4f}")
    P_("\ncommon contamination scale x0/x1/x2:")
    v1c = {}
    for key in [k for k in res if k[0] == 'v1common']:
        t = res[key]['tot']; v1c['|'.join(map(str, key[1:]))] = {k: float(t[:, j].mean()) for j, k in enumerate(('A12', 'C1', 'C2', 'E'))}
        P_(f"{key[1] + ' x' + str(key[2]):22s} A12={t[:,0].mean():.4f} C1={t[:,1].mean():.4f} E={t[:,3].mean():.4f}")
    P_("\n## contagion: dC = C_other(no fault) - C_other(fault), window = the 100 fault epochs (post = the next 100); who=1: fault on observer 2; who=0: fault on observer 1 (leader in the leader rule). Cf = correctness of the faulted observer in the fault run (window).")
    P_(f"{'configuration':26s} {'fault':8s} {'who':>3s} {'dC window':>17s} {'dC post':>17s} {'Cf win':>8s} {'Cf post':>8s} {'A12 win':>8s}")
    summary = {}
    for name in CONFIGS:
        for kd in KINDS:
            for who in (1, 0):
                f = res[('fault', name, kd, who)]; n = res[('nofault', name, kd, who)]; oc = 1 if who == 1 else 2; fc = 2 if who == 1 else 1
                dw = n['win'][:, oc] - f['win'][:, oc]; dp = n['post'][:, oc] - f['post'][:, oc]
                summary[f"{name}|{kd}|{who}"] = dict(dC_win=float(dw.mean()), se_win=se(dw), dC_post=float(dp.mean()), se_post=se(dp), Cf_win=float(f['win'][:, fc].mean()), Cf_post=float(f['post'][:, fc].mean()), A12_win=float(f['win'][:, 0].mean()))
                P_(f"{name:26s} {kd:8s} {who:3d} {dw.mean():+8.4f}+-{se(dw):.4f}  {dp.mean():+8.4f}+-{se(dp):.4f}  {f['win'][:,fc].mean():8.4f} {f['post'][:,fc].mean():8.4f} {f['win'][:,0].mean():8.4f}")
    P_("\n## decomposition of dC (window): support loss = C(no fault) - C(isolated), harm of coupling = C(isolated) - C(coupled fault); dC = support + harm")
    P_(f"{'configuration':26s} {'fault':8s} {'who':>3s} {'dC total':>9s} {'support loss':>17s} {'harm of coupling':>19s}")
    decomp = {}
    for name in CONFIGS:
        for kd in KINDS:
            for who in (1, 0):
                f = res[('fault', name, kd, who)]; n = res[('nofault', name, kd, who)]; iso = res[('isolated', name, kd, who)]; oc = 1 if who == 1 else 2
                sup = n['win'][:, oc] - iso['win'][:, oc]; harm = iso['win'][:, oc] - f['win'][:, oc]; tot_ = sup + harm
                decomp[f"{name}|{kd}|{who}"] = dict(total=float(tot_.mean()), support=float(sup.mean()), se_support=se(sup), harm=float(harm.mean()), se_harm=se(harm))
                P_(f"{name:26s} {kd:8s} {who:3d} {tot_.mean():+9.4f} {sup.mean():+9.4f}+-{se(sup):.4f} {harm.mean():+9.4f}+-{se(harm):.4f}")
    P_("\n## decoding-floor ablation (GMIN=0.05 instead of 0.30), fault on observer 2")
    abl = {}
    for name in ('mixed (16,16)', 'mixedw (16,16)'):
        for kd in ('lowgain', 'zerogain'):
            f = res[('ablate_fault', name, kd)]; n = res[('ablate_nofault', name, kd)]; dw = n['win'][:, 1] - f['win'][:, 1]
            abl[f"{name}|{kd}"] = dict(dC_win=float(dw.mean()), se_win=se(dw)); P_(f"{name:26s} {kd:8s} dC window {dw.mean():+8.4f}+-{se(dw):.4f}")
    open('d2_v3_output.txt', 'w').write("\n".join(L) + "\n")
    json.dump(dict(baseline=base, contagion=summary, decomposition=decomp, v1scan=v1, v1common=v1c, ablation=abl, reps=REPS, bits=BITS), open('d2_v3_output.json', 'w'))
    print("\n".join(L))
