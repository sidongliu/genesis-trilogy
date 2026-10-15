"""Extra checks for TD-III.
(i)  Off-frame probe record Fisher metric F_probe(theta) = rho^2 cos^2 (rho^2 sin^2 + V)/(V - rho^2 sin^2)^2  (>= rho^2 cos^2/V)
     and the in-frame natural-gradient magnitude |grad L|/F near calibration (L = s^2 + rho^2 sin^2 theta).
(ii) Drift per use at which the price of keeping up equals the record value 0.16 kT (g = 0.55, w = kT ln 2).
(iii) Smoother drift (m-fold integrated random walk, m = 1, 2, 3): steady-state Kalman error of the position scales as
     nu^{1/m} R^{(2m-1)/(2m)}, and the optimal reference cost as nu^{2/(4m-1)} (stationary constant-rate policies only).
Output: check_extra_output.json
"""
import json, math
import numpy as np
from scipy import optimize

out = {}
rho, s = 1.0, 0.5; V = rho**2 + s**2
def cov_probe(th):
    a = rho*np.sin(th); return np.array([[1.0, a], [a, V]])
def cov_in(th):
    a = rho*np.cos(th); return np.array([[1.0, a], [a, V]])
def fisher_cov(covf, th, h=1e-6):
    S = covf(th); Si = np.linalg.inv(S); dS = (covf(th+h) - covf(th-h))/(2*h)
    return 0.5*np.trace(Si@dS@Si@dS)
def Fprobe(th):
    return rho**2*np.cos(th)**2*(rho**2*np.sin(th)**2 + V)/(V - rho**2*np.sin(th)**2)**2
out['probe_metric_formula_residual'] = max(abs(Fprobe(t) - fisher_cov(cov_probe, t)) for t in np.linspace(0, 1.4, 30))
out['natgrad_inframe_over_1/theta'] = {str(t): (2*rho**2*math.sin(t)*math.cos(t)/fisher_cov(cov_in, t))*t for t in [1e-3, 1e-2, 0.1]}
out['natgrad_limit_2rho2_over_kappa'] = 2*rho**2/(rho**2*(2*rho**2 + s**2)/s**4)

# (ii)
g = 0.55; G = 1/(1 - g*g); J = G; w = math.log(2)
def floor(nu):
    f = lambda lp: (w/J)*nu**2/(math.exp(lp)*(math.exp(lp) + nu**2)) + 0.5*G*math.exp(lp)
    return optimize.minimize_scalar(f, bounds=(-30, 5), method='bounded', options={'xatol': 1e-12}).fun
nu_eq = optimize.brentq(lambda lnu: floor(math.exp(lnu)) - 0.16, math.log(1e-4), math.log(0.5))
out['nu_where_floor_equals_record_value'] = math.exp(nu_eq)

# (iii) m-fold integrated random walk, observe position with noise variance R (one use per step)
def dare_P11(m, nu, R):
    if m == 1:  # closed form (posterior variance of the scalar random walk)
        q = nu**2
        return (-q + math.sqrt(q*q + 4*q*R))/2
    A = np.eye(m) + np.diag(np.ones(m-1), 1)   # x_{k+1} = x_k + x'_k ..., last component random walk
    Qn = np.zeros((m, m)); Qn[-1, -1] = nu**2
    H = np.zeros((1, m)); H[0, 0] = 1.0
    P = np.eye(m)
    for it in range(200000):
        Pp = A@P@A.T + Qn
        S = H@Pp@H.T + R
        K = Pp@H.T/S
        Pn = (np.eye(m) - K@H)@Pp
        if np.max(np.abs(Pn - P)) < 1e-18*max(1, np.max(np.abs(P))): P = Pn; break
        P = Pn
    return P[0, 0]
res = {}
for m in [1, 2, 3]:
    nus = [1e-3, 1e-4, 1e-5] if m < 3 else [1e-4, 1e-5, 1e-6]
    # error scaling at fixed R
    R = 1.0
    P = [dare_P11(m, nu, R) for nu in nus]
    sl_nu = [math.log(P[i+1]/P[i])/math.log(nus[i+1]/nus[i]) for i in range(2)]
    # cost optimum over rho (R = 1/rho), w = c = J = 1
    costs = []
    for nu in nus:
        f = lambda lr: math.exp(lr) + dare_P11(m, nu, 1/math.exp(lr))
        rr = optimize.minimize_scalar(f, bounds=(-12, 8), method='bounded', options={'xatol': 1e-6})
        costs.append(rr.fun)
    sl_cost = [math.log(costs[i+1]/costs[i])/math.log(nus[i+1]/nus[i]) for i in range(2)]
    res[str(m)] = dict(nus=nus, P11=P, slope_P_vs_nu=sl_nu, predicted_P_slope=1/m, cost=costs,
                       slope_cost=sl_cost, predicted_cost_slope=2/(4*m - 1))
out['smooth_drift'] = res
json.dump(out, open('check_extra_output.json', 'w'), indent=1)
print(json.dumps(out, indent=1))
