"""Independent recheck of the analytic numbers kept in TD-III Section 9 (no rerun of the large simulations).
- critical drift nu* of Proposition memory_drift for Model A
- X_sharp limsup bounds for gain and bias, decoded-state bound 0.4518 and unrolled form 0.5123
- Proposition two_optima: exact enumeration of the four error branches and both binomial draws (M = 20)
- TD-I calibrated loss for the Model A reference budget (n_r = 128 per sign, g = 0.55, b = 0)
Output: check_section9_output.json
"""
import json, math
from math import comb

out = {}
# nu*
alpha = 0.25; nr = 128; g = 0.55; b = 0.0
s2 = (1 - g*g - b*b)/(2*nr)
nu_star = math.sqrt(2*alpha**2*s2/((1-alpha)*(2-alpha)))
out['sigma2'] = s2; out['nu_star'] = nu_star

# X_sharp
def xsharp(nu, s2, alpha, Delta):
    beta = 1 - alpha
    return (Delta/2 + beta**2*nu + math.sqrt(beta**2*(Delta/2 + nu)**2 + (1 - beta**2)*alpha**2*s2))/(1 - beta**2)
Delta = 2/1024; s2u = 1/(2*nr)
nu_g = 0.17*2*math.pi/420 + 0.03*2*math.pi/67
nu_b = 0.14*2*math.pi/300 + 0.025*2*math.pi/43
eg = xsharp(nu_g, s2u, alpha, Delta); eb = xsharp(nu_b, s2u, alpha, Delta)
ns = 256; gmin = 0.30
dec = (ns**-0.5 + eg + eb)/gmin + Delta/2
def unrolled(nu, s2, alpha, Delta):
    beta = 1 - alpha
    return beta*nu/alpha + math.sqrt(s2)*math.sqrt(alpha/(2 - alpha)) + Delta/(2*alpha)
dec_u = (ns**-0.5 + unrolled(nu_g, s2u, alpha, Delta) + unrolled(nu_b, s2u, alpha, Delta))/gmin + Delta/2
out.update(nu_g=nu_g, nu_b=nu_b, limsup_g=eg, limsup_b=eb, decoded_bound=dec, decoded_bound_unrolled=dec_u)

# two optima, exact enumeration
M = 20; n_plus = 12; x = (2*n_plus - M)/M
a, c, r, h, d = 0.65, 0.25, 0.10, 0.8, -0.7
def next_loss(t):
    tot = 0.0
    for e1 in (0.0, 0.2):
        for e2 in (-0.1, 0.1):
            xh1 = x + e1; xh2 = x + e2
            u = -h*(t*xh1 + (1 - t)*xh2)
            v = c*u + r*d
            pp = (1 + a + v)/2      # P(+1) for a variable at +1
            pm = (1 - a + v)/2      # P(+1) for a variable at -1
            el = 0.0
            for k1 in range(n_plus + 1):
                p1 = comb(n_plus, k1)*pp**k1*(1 - pp)**(n_plus - k1)
                for k2 in range(M - n_plus + 1):
                    p2 = comb(M - n_plus, k2)*pm**k2*(1 - pm)**(M - n_plus - k2)
                    xn = (2*(k1 + k2) - M)/M
                    el += p1*p2*xn*xn/4
            tot += 0.25*el
    return tot
def est_err(t):
    return t*t*0.02 + (1 - t)**2*0.01
rows = {}
for t in (1/3, 0.5, 2/3):
    rows[f"{t:.4f}"] = dict(joint_estimation_error=est_err(t), next_step_world_loss=next_loss(t))
out['two_optima_table'] = rows
# formula of Proposition task_risk
q = c*h; z = (a - q)*x + r*d
R = [[0.02, 0.0], [0.0, 0.01]]; bvec = [z*0.1, 0.0]
def formula(t):
    wv = [t, 1 - t]
    wRw = sum(wv[i]*R[i][j]*wv[j] for i in range(2) for j in range(2))
    bw = sum(bvec[i]*wv[i] for i in range(2))
    K = 0.25*((1 - 1/M)*z*z + (1/M)*(1 - a*a + a*a*x*x))
    return K + (1 - 1/M)/4*(q*q*wRw - 2*q*bw)
out['two_optima_formula_residual'] = max(abs(formula(t) - next_loss(t)) for t in (0.0, 1/3, 0.5, 2/3, 1.0))

# TD-I calibrated loss for Model A references
kappa = 1 + g*g/(1 - b*b)
out['TDI_calibrated_loss_per_use_kT'] = kappa/(4*nr)
out['TDI_kappa'] = kappa
json.dump(out, open('check_section9_output.json', 'w'), indent=1)
print(json.dumps(out, indent=1))
