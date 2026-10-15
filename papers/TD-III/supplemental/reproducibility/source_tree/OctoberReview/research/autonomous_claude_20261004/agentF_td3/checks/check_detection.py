"""Checks for TD-III Section 3 (even residual models, detection times) and the worked example.

R1  Even families: F(0)=0, F(theta)=kappa theta^2+O(theta^4), kappa=E_0[(d^2_theta log p)^2],
    KL(p_theta||p_0)=kappa theta^4/8+O(theta^6).
    Instances: Gaussian mean model N(a theta^2, se^2); TD-II two-channel reader (in-frame record (v.x, y)).
    Off-frame probe record (v_perp.x, y): KL=-1/2 ln(1-rho^2 sin^2/V), F(0)=rho^2/V.
R2  Detection lower bound: any test with errors (alpha,beta) needs sum_m KL >= d(1-beta||alpha).
    Linear drift and Brownian drift; comparison with TD-II frozen-model failure time t*=ln2/((D-1)sigma).
    Monte Carlo of the likelihood-ratio test (simple vs simple) to show the bound is of the right order.
Deterministic (fixed seeds).  Output: check_detection_output.json
"""
import json, math
import numpy as np
from scipy import integrate, optimize

out = {}
rng = np.random.default_rng(20261005)

# ---------------- R1: TD-II two-channel reader ----------------
rho, s = 1.0, 0.5
V = rho**2 + s**2

def cov_in(th):
    a = rho*np.cos(th)
    return np.array([[1.0, a], [a, V]])

def cov_probe(th):
    a = rho*np.sin(th)
    return np.array([[1.0, a], [a, V]])

def kl_gauss(S1, S0):
    """KL(N(0,S1)||N(0,S0))"""
    S0i = np.linalg.inv(S0)
    k = S1.shape[0]
    return 0.5*(np.trace(S0i@S1) - k - np.log(np.linalg.det(S1)/np.linalg.det(S0)))

def fisher_cov(covf, th, h=1e-6):
    S = covf(th); Si = np.linalg.inv(S)
    dS = (covf(th+h) - covf(th-h))/(2*h)
    return 0.5*np.trace(Si@dS@Si@dS)

kappa_formula = rho**2*(2*rho**2 + s**2)/s**4
ths = [1e-3, 1e-2, 0.05, 0.1]
out['reader_kappa_formula'] = kappa_formula
out['reader_F_over_theta2'] = {str(t): fisher_cov(cov_in, t)/t**2 for t in ths}

def kl_in_exact(th):
    return rho**2*(1-np.cos(th))/s**2 - 0.5*np.log(1 + rho**2*np.sin(th)**2/s**2)

def kl_probe_exact(th):
    return -0.5*np.log(1 - rho**2*np.sin(th)**2/V)

res_in = max(abs(kl_in_exact(t) - kl_gauss(cov_in(t), cov_in(0.0))) for t in np.linspace(0.01, 1.5, 40))
res_pr = max(abs(kl_probe_exact(t) - kl_gauss(cov_probe(t), cov_probe(0.0))) for t in np.linspace(0.01, 1.5, 40))
out['reader_KL_exact_formula_residual'] = res_in
out['probe_KL_exact_formula_residual'] = res_pr
out['reader_KL_over_theta4_vs_kappa_over_8'] = {str(t): [kl_in_exact(t)/t**4, kappa_formula/8] for t in [1e-2, 3e-2, 0.1]}
out['probe_F0_numeric'] = fisher_cov(cov_probe, 0.0)
out['probe_F0_formula'] = rho**2/V

# kappa as E_0[(d^2 log p)^2] by Monte Carlo for the reader (independent route)
N = 400000
z = rng.multivariate_normal([0, 0], cov_in(0.0), size=N)
def logp(th, z):
    S = cov_in(th); Si = np.linalg.inv(S)
    return -0.5*np.einsum('ni,ij,nj->n', z, Si, z) - 0.5*np.log(np.linalg.det(S))
h = 1e-3
d2 = (logp(h, z) - 2*logp(0.0, z) + logp(-h, z))/h**2
out['reader_kappa_MC_E0_d2logp_sq'] = float(np.mean(d2**2))
out['reader_kappa_MC_se'] = float(np.std(d2**2)/math.sqrt(N))

# Gaussian mean model (July residual model of Section 3)
a, se = 1.2, 0.35
out['gauss_mean_kappa'] = 4*a**2/se**2
out['gauss_mean_KL_theta4_coeff'] = a**2/(2*se**2)  # exact, no remainder
out['gauss_mean_kappa_over_8'] = (4*a**2/se**2)/8

# Monte Carlo score-squared for the reader at a few angles (direct Fisher test)
mc = {}
for t in [0.05, 0.1, 0.2, 0.4]:
    zz = rng.multivariate_normal([0, 0], cov_in(t), size=200000)
    hh = 1e-5
    sc = (logp(t+hh, zz) - logp(t-hh, zz))/(2*hh)
    mc[str(t)] = [float(np.mean(sc**2)), float(np.std(sc**2)/math.sqrt(len(sc))), fisher_cov(cov_in, t)]
out['reader_score_squared_MC_vs_exact'] = mc

# ---------------- R2: detection lower bounds ----------------
def dbin(p, q):
    return p*math.log(p/q) + (1-p)*math.log((1-p)/(1-q))
alpha = beta = 0.05
dreq = dbin(1-beta, alpha)
out['d_required_alpha_beta_0.05'] = dreq

r = 1.0  # records per unit time
def T_lin_exact(eps, klf, p=1.0):
    """smallest T with r*p*sum over records of KL(theta_m) >= dreq, theta_m = eps*m/r"""
    tot = 0.0; m = 0
    while tot < dreq:
        m += 1
        tot += p*klf(eps*m/r)
        if m > 10**8: return float('inf')
    return m/r

lin = {}
for eps in [1e-2, 3e-3, 1e-3]:
    Te = T_lin_exact(eps, kl_in_exact)
    Tasym_even = (40*dreq/(kappa_formula*r*eps**4))**0.2
    F0 = rho**2/V
    Tp = T_lin_exact(eps, kl_probe_exact)
    Tasym_probe = (6*dreq/(F0*r*eps**2))**(1/3)
    lin[str(eps)] = dict(T_even_exact=Te, T_even_asym=Tasym_even, T_probe_exact=Tp, T_probe_asym=Tasym_probe)
out['linear_drift_detection_lower_bounds'] = lin

# Brownian drift on the circle (D=2): theta_t ~ wrapped N(0, 2 sigma t); expected sum of KL
def E_kl_wrapped(var, klf):
    sd = math.sqrt(var)
    f = lambda th: klf(th)*sum(math.exp(-(th+2*math.pi*k)**2/(2*var)) for k in range(-3, 4))/math.sqrt(2*math.pi*var)
    val, _ = integrate.quad(f, -math.pi, math.pi, limit=200)
    return val

def T_brown_exact(sigma, klf):
    tot = 0.0; m = 0
    while tot < dreq:
        m += 1
        tot += E_kl_wrapped(2*sigma*m/r, klf)
        if m > 10**6: return float('inf')
    return m/r

br = {}
for sigma in [1e-2, 1e-3, 1e-4]:
    Tb = T_brown_exact(sigma, kl_in_exact)
    Tasym = (2*dreq/(kappa_formula*r*sigma**2))**(1/3)
    tstar = math.log(2)/sigma
    br[str(sigma)] = dict(T_even_exact=Tb, T_even_asym=Tasym, frozen_failure_tstar=tstar)
out['brownian_detection_vs_frozen_failure'] = br
out['brownian_sigma_threshold_formula'] = kappa_formula*r*math.log(2)**3/(2*dreq)

# Likelihood-ratio test (simple vs simple, known linear drift), Monte Carlo errors at T = c * T_lower
def llr_errors(eps, T, nrep=4000):
    ms = np.arange(1, int(round(T*r))+1)
    th = eps*ms/r
    S1 = np.array([cov_in(t) for t in th]); S0 = cov_in(0.0)
    S1i = np.linalg.inv(S1); S0i = np.linalg.inv(S0)
    logdet1 = np.log(np.linalg.det(S1)); logdet0 = np.log(np.linalg.det(S0))
    def llr(Z):  # Z: nrep x m x 2
        q1 = np.einsum('rmi,mij,rmj->rm', Z, S1i, Z)
        q0 = np.einsum('rmi,ij,rmj->rm', Z, S0i, Z)
        return np.sum(-0.5*q1 + 0.5*q0 - 0.5*logdet1 + 0.5*logdet0, axis=1)
    L1 = np.linalg.cholesky(S1); L0 = np.linalg.cholesky(S0)
    g1 = rng.standard_normal((nrep, len(ms), 2)); g0 = rng.standard_normal((nrep, len(ms), 2))
    Z1 = np.einsum('mij,rmj->rmi', L1, g1); Z0 = np.einsum('ij,rmj->rmi', L0, g0)
    l1 = llr(Z1); l0 = llr(Z0)
    thr = 0.0
    # choose the threshold that equalizes the two errors
    cand = np.quantile(np.concatenate([l0, l1]), np.linspace(0.01, 0.99, 197))
    best = min(cand, key=lambda c: abs(np.mean(l0 > c) - np.mean(l1 <= c)))
    return float(np.mean(l0 > best)), float(np.mean(l1 <= best))

eps = 1e-2
Tlow = lin[str(eps)]['T_even_exact']
llr_tab = {}
for c in [1.0, 1.5, 2.0]:
    e0, e1 = llr_errors(eps, c*Tlow)
    llr_tab[str(c)] = dict(T=c*Tlow, type1=e0, type2=e1)
out['LLR_test_eps_0.01_equal_error'] = llr_tab

json.dump(out, open('check_detection_output.json', 'w'), indent=1)
print(json.dumps(out, indent=1))
