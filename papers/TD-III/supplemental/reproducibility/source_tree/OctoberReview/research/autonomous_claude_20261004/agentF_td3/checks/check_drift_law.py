"""Checks for the drift law (TD-III Section 5).

Theorem: scalar self-parameter, Gaussian random walk theta_k = theta_{k-1} + w_k, Var w = nu^2; reference reads with
expected Fisher information <= J each; rho_k >= 0 reads in use k (any schedule, may depend on past records), cost w each;
loss at use k with E loss_k >= l(P_k), P_k the posterior Cramer-Rao bound. Then
   liminf (1/K) sum_k (w rho_k + loss_k) >= min_P h(P),  h(P) = (w/J) nu^2/(P(P+nu^2)) + l(P).
Quadratic loss l = cP:   C - c nu^2 <= min h <= C,  C = 3*2^{-2/3} c^{2/3} (w nu^2/J)^{1/3}
(with c = kT G/2:  C = 3*4^{-2/3} w^{1/3} (kT G nu)^{2/3} J^{-1/3}).

Checks:
 (a) min h numerically vs the two closed-form brackets, several nu.
 (b) 20000 random time-varying schedules (periodic and aperiodic) through the exact information recursion: none beats min h.
 (c) average-cost value iteration on a P grid (optimal control over all schedules): equals min h.
 (d) Gaussian Kalman simulation with constant rho*: realized average cost -> min h.
 (e) Binomial references on the gain-bias channel (gain drifts, b = 0): persistent estimator (EMA with optimized alpha and
     reads), TD-I loss metric G = 1/(1-g^2): realized cost vs the floor computed with E[J(g_k)].
 (f) C-III relative-entropy loss l = 1/2 ln(1+P/v): floor for all schedules vs C-III's stationary constant-rate optimum.
 (g) Linear (deterministic) drift with a constant-weight memory: optimal cost exponent 2/5.
 (h) Thermodynamic example numbers: w = kT ln 2, g = 0.55, b = 0.
Output: check_drift_law_output.json
"""
import json, math
import numpy as np
from scipy import optimize

rng = np.random.default_rng(5)
out = {}

def h_quad(P, w, J, nu, c):
    return (w/J)*nu**2/(P*(P+nu**2)) + c*P

def minh(w, J, nu, c, loss=None):
    if loss is None:
        f = lambda lp: h_quad(math.exp(lp), w, J, nu, c)
    else:
        f = lambda lp: (w/J)*nu**2/(math.exp(lp)*(math.exp(lp)+nu**2)) + loss(math.exp(lp))
    r = optimize.minimize_scalar(f, bounds=(math.log(1e-12), math.log(1e3)), method='bounded', options={'xatol': 1e-12})
    return r.fun, math.exp(r.x)

def Cconst(w, J, nu, c):
    return 3*2**(-2/3)*c**(2/3)*(w*nu**2/J)**(1/3)

# (a)
w, J, c = 1.0, 1.0, 1.0
tab = {}
for nu in [1e-1, 3e-2, 1e-2, 1e-3, 1e-4]:
    m, P = minh(w, J, nu, c)
    C = Cconst(w, J, nu, c)
    tab[str(nu)] = dict(min_h=m, P_star=P, C=C, C_minus_c_nu2=C - c*nu**2, ratio=m/C,
                        rho_star=nu**2/(P*(P+nu**2))/J, rho_leading=(2*c*nu/(4*w*math.sqrt(J)))**(2/3))
out['a_brackets'] = tab
assert all(v['C_minus_c_nu2'] <= v['min_h'] + 1e-12 <= v['C'] + 2e-12 for v in tab.values())

# (b) random schedules through the exact information recursion
def avg_cost_schedule(rhos, w, J, nu, c, P0=1.0, K=4000):
    Q = 1.0/P0; tot = 0.0; L = len(rhos)
    for k in range(K):
        rho = rhos[k % L]
        Q = 1.0/(1.0/Q + nu**2) + J*rho
        tot += w*rho + c/Q
    return tot/K

nu = 1e-2
mstar, Pstar = minh(w, J, nu, c)
rho_star = nu**2/(Pstar*(Pstar+nu**2))/J
viol = 0; best = float('inf')
for trial in range(20000):
    L = int(rng.integers(1, 40))
    kind = trial % 3
    if kind == 0:
        rhos = rng.exponential(rho_star, size=L)
    elif kind == 1:
        rhos = np.zeros(L); rhos[0] = rho_star*L*rng.uniform(0.3, 3)   # bursts
    else:
        rhos = rho_star*rng.uniform(0, 3)*np.ones(L)*(rng.uniform(size=L) < rng.uniform(0.1, 1))
    val = avg_cost_schedule(rhos, w, J, nu, c, K=3000)
    best = min(best, val)
    if val < mstar - 1e-9: viol += 1
out['b_random_schedules'] = dict(n=20000, violations=viol, best_found=best, floor=mstar, best_over_floor=best/mstar)

# (c) average-cost value iteration on a log grid of the information Q
Qs = np.exp(np.linspace(math.log(1/Pstar)-2.5, math.log(1/Pstar)+2.5, 700))
psi = 1.0/(1.0/Qs + nu**2)
Vf = np.zeros_like(Qs)
for it in range(1500):
    # next Q' >= psi(Q); choose Q' on grid; cost w (Q'-psi(Q))/J + c/Q'
    cost = w*np.maximum(Qs[None, :] - psi[:, None], 0)/J + c/Qs[None, :]
    feas = Qs[None, :] >= psi[:, None] - 1e-15
    tot = np.where(feas, cost + Vf[None, :], np.inf)
    Vn = tot.min(axis=1)
    g = Vn.min()
    Vf = Vn - g
out['c_value_iteration_gain'] = dict(average_cost=float(g), floor=mstar, ratio=float(g)/mstar)

# (d) Gaussian Kalman with constant rho*, 4000 independent trajectories
ntr = 4000; K = 12000; burn = 6000
theta = np.zeros(ntr); m = np.zeros(ntr); P = 1.0; tot = 0.0; err2 = 0.0
R = 1.0/(J*rho_star)
for k in range(K):
    theta += nu*rng.standard_normal(ntr)
    Pp = P + nu**2
    y = theta + math.sqrt(R)*rng.standard_normal(ntr)
    Kg = Pp/(Pp + R)
    m = m + Kg*(y - m); P = (1 - Kg)*Pp
    if k >= burn:
        tot += w*rho_star + c*np.mean((m - theta)**2)
        err2 += np.mean((m - theta)**2)
nn = K - burn
out['d_kalman_sim'] = dict(avg_cost=tot/nn, floor=mstar, ratio=tot/nn/mstar, mse=err2/nn, P_star=Pstar, P_final=P,
                           kalman_gain_alpha=Pstar*J*rho_star, alpha_leading=nu*math.sqrt(rho_star*J))

# (e) Binomial references on the gain (b = 0), EMA persistent estimator, TD-I metric G = 1/(1-g^2)
# loss per use = (1/2) G (ghat - g)^2 (units kT); reference cost w per read; n_pair reads of each sign per use.
def sim_binomial(nu_g, n_pair, alpha, w_ref, T=4000, burn=1500, ntraj=1000, g0=0.55, seed=11):
    rr = np.random.default_rng(seed)
    g = np.full(ntraj, g0); ghat = np.full(ntraj, g0)
    tot = 0.0; cnt = 0; Jsum = 0.0; Gsum = 0.0
    for k in range(T):
        g = np.clip(g + nu_g*rr.standard_normal(ntraj), 0.05, 0.95)
        p = rr.binomial(n_pair, (1+g)/2); mneg = rr.binomial(n_pair, (1-g)/2)
        gt = (p - mneg)/n_pair
        ghat = (1-alpha)*ghat + alpha*gt
        if k >= burn:
            G = 1/(1-g**2)
            tot += np.sum(0.5*G*(ghat-g)**2) + ntraj*w_ref*2*n_pair
            Jsum += np.sum(1/(1-g**2)); cnt += ntraj
    return tot/cnt, Jsum/cnt

nu_g = 0.003; w_e = 1e-4
gref = 0.55; Gref = 1/(1-gref**2); Jref = Gref
rho_lead = (Gref*nu_g/(4*w_e*math.sqrt(Jref)))**(2/3)
alpha_lead = nu_g*math.sqrt(rho_lead*Jref)
grid = {}
best_e = None
for n_pair in [1, 2, 3, 4]:
    for alpha in [0.004, 0.006, 0.008, 0.010, 0.013, 0.017]:
        val, EJ = sim_binomial(nu_g, n_pair, alpha, w_e)
        grid[f"{n_pair},{alpha}"] = val
        if best_e is None or val < best_e[0]:
            best_e = (val, n_pair, alpha, EJ)
val, n_pair, alpha, EJ = best_e
fl_EJ, _ = minh(w_e, EJ, nu_g, 0.5*Gref)
out['e_binomial_EMA'] = dict(w_ref=w_e, nu=nu_g, best_cost=val, n_pair=n_pair, reads_per_use=2*n_pair, alpha=alpha,
                            E_J=EJ, floor_with_EJ_and_Gref=fl_EJ, ratio=val/fl_EJ, rho_leading=rho_lead,
                            alpha_leading=alpha_lead, grid=grid,
                            note='G evaluated at g=0.55; loss uses G(g_k); floor uses E[J(g_k)]')

# (f) C-III relative-entropy loss (Gaussian self-state N(theta, v), readings with variance r): all schedules
v = 1.0; rvar = 1.0; q = 1e-4; wC = 1e-3
nuC = math.sqrt(q); JC = 1/rvar
flC, PC = minh(wC, JC, nuC, None, loss=lambda P: 0.5*math.log(1+P/v))
# C-III stationary constant-rate optimum (exact discrete cost minimized over N)
def stat_cost(N):
    P = (-q + math.sqrt(q*q + 4*q*rvar/N))/2
    return 0.5*math.log(1+P/v) + wC*N
rN = optimize.minimize_scalar(lambda lN: stat_cost(math.exp(lN)), bounds=(-10, 10), method='bounded')
out['f_CIII_KL_loss'] = dict(floor_all_policies=flC, stationary_constant_rate_optimum=rN.fun, N_star=math.exp(rN.x),
                            CIII_leading_3wN=3*wC*(math.sqrt(q*rvar)/(4*v*wC))**(2/3))

# (g) linear drift, constant-weight memory: cost exponent
def lin_cost(nu_l, w, J, c):
    def f(x):
        a = 1/(1+math.exp(-x[0])); rho = math.exp(x[1])
        s2 = 1/(rho*J)
        mse = nu_l**2*(1-a)**2/a**2 + a*s2/(2-a)
        return w*rho + c*mse
    best = None
    for a0 in [-6, -4, -2, 0]:
        for r0 in [-6, -3, 0, 3]:
            r = optimize.minimize(f, [a0, r0], method='Nelder-Mead', options={'xatol': 1e-10, 'fatol': 1e-14, 'maxiter': 20000})
            if best is None or r.fun < best.fun: best = r
    return best.fun
nus = [1e-3, 1e-4, 1e-5, 1e-6]
lc = [lin_cost(x, 1.0, 1.0, 1.0) for x in nus]
slopes = [math.log(lc[i+1]/lc[i])/math.log(nus[i+1]/nus[i]) for i in range(len(nus)-1)]
rw = [minh(1.0, 1.0, x, 1.0)[0] for x in nus]
rw_slopes = [math.log(rw[i+1]/rw[i])/math.log(nus[i+1]/nus[i]) for i in range(len(nus)-1)]
def lin_lead(nu_l, w=1.0, J=1.0, c=1.0):
    A = 3*4**(-2/3)*c*nu_l**(2/3)*J**(-2/3)
    rho = (2*A/(3*w))**0.6
    return 2.5*w*rho
out['g_linear_drift_EMA'] = dict(nus=nus, cost=lc, slopes=slopes, leading_formula=[lin_lead(x) for x in nus],
                                 random_walk_floor=rw, rw_slopes=rw_slopes)

# (h) thermodynamic example: gain drifts, b = 0, w_ref = kT ln 2, loss metric G = J = 1/(1-g^2) (units kT)
ex = {}
for nu_h in [1e-2, 3e-3, 1e-3, 1e-4]:
    Cth = 3*4**(-2/3)*math.log(2)**(1/3)*(Gref*nu_h)**(2/3)*Jref**(-1/3)
    mth, Pth = minh(math.log(2), Jref, nu_h, 0.5*Gref)
    rho_h = nu_h**2/(Pth*(Pth+nu_h**2))/Jref
    ex[str(nu_h)] = dict(C=Cth, exact_floor=mth, lower=Cth - 0.5*Gref*nu_h**2, rho_star=rho_h,
                         alpha_star=Pth*Jref*rho_h, record_value=0.16)
out['h_thermo_example_g055'] = ex
out['constant_3_4^-2/3'] = 3*4**(-2/3)

json.dump(out, open('check_drift_law_output.json', 'w'), indent=1)
print(json.dumps(out, indent=1))
