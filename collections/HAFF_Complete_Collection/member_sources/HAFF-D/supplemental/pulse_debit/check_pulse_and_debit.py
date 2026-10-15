"""Own numerical checks; not a reconstruction of an undocumented BRAIN pilot."""
import json
import math
import numpy as np
from scipy.integrate import quad
from scipy.special import roots_legendre


def h(x):
    return 0.0 if x in (0, 1) else -x*math.log(x)-(1-x)*math.log1p(-x)


def debits(epsilon, q=.05):
    pz = (1-math.sqrt(1-16*q*(1-q)*epsilon*(1-epsilon)))/2
    return {'Z': h(pz), 'X': h(q)}


def info_variance(p):
    return p*(1-p)*math.log((1-p)/p)**2


def pulse_check(nodes, omega_cutoff):
    x, w = roots_legendre(nodes)
    shape = np.exp(-1/(1-x*x))
    def fourier(omega):
        return float(np.dot(w*shape, np.cos(omega*x)))
    norm, norm_err = quad(lambda omega: omega*fourier(omega)**2/(4*math.pi),
                          0, omega_cutoff, epsabs=1e-12, epsrel=2e-12, limit=600)
    amp2 = .2/norm
    def energy_integrand(x):
        if abs(x) >= 1:
            return 0.0
        return amp2*x*x*math.exp(-2/(1-x*x))/(1-x*x)**4
    moments = [quad(lambda x: (3+x)**j*energy_integrand(x), -1, 1,
                    epsabs=2e-12, epsrel=2e-12, limit=300)[0] for j in range(3)]
    def invariant(mismatch):
        m2 = mismatch*mismatch
        e0, e1, e2 = moments
        return 1+2*.05*m2*e1+.05**2*m2*m2*(e1*e1-e0*e2)
    return {'nodes': nodes, 'omega_cutoff': omega_cutoff,
            'unit_amplitude_k': norm, 'outer_quad_error': norm_err,
            'amplitude': math.sqrt(amp2), 'moments': moments,
            'C_frozen_a1_alpha1p1': invariant(-.1),
            'C_supplied_mismatch_0p04': invariant(.04)}


eps_drv = .01
theta = math.acos(2*eps_drv-1)
drive = theta*(1-2*eps_drv)/(4*math.sqrt(eps_drv*(1-eps_drv)))-.5
bench = {}
for epsilon in (.01, .1):
    bench[str(epsilon)] = {}
    for p, debit in debits(epsilon).items():
        f = debit+1.5*debit*debit
        heat = f/(2*math.pi)
        bench[str(epsilon)][p] = {'full_debit': debit, 'F': f, 'heat': heat,
                                  'drive_to_heat': drive/heat,
                                  'one_heat_unrecovered_fraction': heat/drive}
q, epsilon = .05, .1
pz = (1-math.sqrt(1-16*q*(1-q)*epsilon*(1-epsilon)))/2
variance = {'Z': info_variance(pz), 'X': info_variance(q)}
matrix_checks = {}
z_a = np.diag([1., 1., -1., -1.])
h_a = np.kron(np.array([[1.,1.],[1.,-1.]])/math.sqrt(2), np.eye(2))
for e in (.01, .1):
    phi_z = np.array([math.sqrt(1-e), 0., 0., math.sqrt(e)])
    for p, phi in [('Z', phi_z), ('X', h_a@phi_z)]:
        pure = np.outer(phi, phi)
        rho = (1-q)*pure + q*z_a@pure@z_a
        eigenvalues = np.linalg.eigvalsh(rho)
        support = eigenvalues[eigenvalues > 1e-14]
        entropy = float(-np.dot(support, np.log(support)))
        var = float(np.dot(support, (-np.log(support)-entropy)**2))
        assert abs(entropy-debits(e)[p]) < 2e-14
        matrix_checks[f'{e}_{p}'] = {'spectrum': eigenvalues.tolist(),
                                     'entropy': entropy, 'information_variance': var}
windows = []
for N in (10**8, 10**12, 10**16):
    L, W, d = N**.25, N**.5, N**.5
    a_x = N**(-.25)
    opportunities = math.floor(N/(W+d))-1
    windows.append({'N': N, 'L_cycles': L, 'W_cycles': W, 'delay_cycles': d,
                    'nonoverlap_opportunities': opportunities,
                    'X_duty': a_x, 'expected_exploration_windows': a_x*opportunities,
                    'sparse_update_gain_bound': a_x,
                    'phase_Jensen_excess_2piQ_per_cycle':
                    1.5*a_x*(1-a_x)*(debits(.1)['X']-debits(.1)['Z'])**2})
pulse = [pulse_check(512, 160), pulse_check(1024, 240)]
assert max(abs(a-b) for a,b in zip(pulse[0]['moments'],pulse[1]['moments'])) < 2e-8
assert abs(pulse[0]['C_supplied_mismatch_0p04']-pulse[1]['C_supplied_mismatch_0p04']) < 1e-10
print(json.dumps({'drive_lower_bound_in_omega_h': drive,
                  'benchmark': bench, 'source_information_variance': variance,
                  'direct_four_by_four_checks': matrix_checks,
                  'window_scaling_diagnostic_not_device': windows,
                  'own_deterministic_pulse': pulse}, indent=2))
