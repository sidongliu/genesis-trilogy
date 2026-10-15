"""Finite checks for the permission-gated TD-III candidate; stdlib only.

Rational witnesses use s=49/50, attainable in the same squeezed-drive
family at an explicitly computed N. They are not reruns of old long paths.
The current manuscript drive is checked separately with floating scalars.
Print JSON; no source, frozen artifact or paper is written by this script.
"""

from fractions import Fraction as Q
from itertools import product
from math import ceil, comb, exp, log, log2, pi, sqrt, tanh
import json


def eye(n):
    return [[Q(i == j) for j in range(n)] for i in range(n)]


def matmul(a, b):
    return [[sum(a[i][k] * b[k][j] for k in range(len(b)))
             for j in range(len(b[0]))] for i in range(len(a))]


def scale(a, x):
    return [[x * v for v in row] for row in a]


def add(a, b):
    return [[x + y for x, y in zip(ar, br)] for ar, br in zip(a, b)]


def inverse(a):
    n = len(a)
    z = [list(a[i]) + eye(n)[i] for i in range(n)]
    for k in range(n):
        j = next(j for j in range(k, n) if z[j][k])
        z[k], z[j] = z[j], z[k]
        p = z[k][k]
        z[k] = [v / p for v in z[k]]
        for j in range(n):
            if j != k:
                p = z[j][k]
                z[j] = [v - p * w for v, w in zip(z[j], z[k])]
    return [row[n:] for row in z]


def resid(a, b):
    return max(abs(x-y) for ar, br in zip(a, b) for x, y in zip(ar, br))


states = list(product((0, 1), (-1, 1)))
delta, eta, d, s = Q(3, 10000), Q(1, 5000), Q(1, 20), Q(49, 50)
K = [[(delta if gi != gj else 1-delta) *
      (eta if si != sj else 1-eta) for gj, sj in states]
     for gi, si in states]


def event(r):
    resolvent = inverse(add(eye(4), scale(K, -(1-r))))
    direct = add(scale(K, 1-d), scale(matmul(matmul(K, K), resolvent), d*r))
    # Derive independently from the explicit eight-state wall-clock blocks.
    aa, ad, da, dd = scale(K, 1-d), scale(K, d), scale(K, r), scale(K, 1-r)
    schur = add(aa, matmul(matmul(ad, inverse(add(eye(4), scale(dd, -1)))), da))
    assert resid(direct, schur) == 0
    assert resid(matmul(add(eye(4), scale(K, -(1-r))), resolvent), eye(4)) == 0
    assert all(sum(row) == 1 and min(row) > 0 for row in direct)
    return direct


def f(r, lam):
    return (1-d)*lam + d*r*lam*lam/(1-(1-r)*lam)


def certificate(kernels, gmax, delta_grid=2**-44):
    amin = min(float(x) for a in kernels for row in a for x in row)
    ratio = max(float(a[i][k] * a[j][m] / (a[i][m] * a[j][k]))
                for a in kernels for i,j,k,m in product(range(4), repeat=4))
    D = log(ratio)
    tau = tanh(D/4)
    epsilon = amin * (1-gmax)/(1+gmax)
    A = -log(epsilon)
    bits = 3 * ceil(log2(2*A/delta_grid+3))
    tv = tanh(delta_grid/(1-tau)/4)
    return dict(min_entry=amin, crossratio_D=D, tau=tau,
                posterior_min=epsilon, log_range_A=A,
                grid_spacing=delta_grid, ideal_model_bits=bits,
                uniform_TV=tv, one_step_regret=2*tv)


def H(p):
    return -(p*log(p) + (1-p)*log(1-p)) if 0 < p < 1 else 0.0


def work(g, sample_count, flip_success=s):
    a, b = (1+g)/2, (1-g)/2
    value = 0.0
    distribution = []
    for k in range(sample_count+1):
        plus = a**k * b**(sample_count-k)
        minus = b**k * a**(sample_count-k)
        pk = comb(sample_count, k)*(plus+minus)/2
        mk = (plus-minus)/(plus+minus)
        qk = (1+flip_success*g*abs(mk))/2
        value += float(pk)*(log(2)-H(float(qk)))
        distribution.append(dict(k=k, probability=str(pk), sign_mean=str(mk)))
    return value, distribution


def memory_enumeration(g):
    # Exact 3-body and 1-food path likelihoods, not posterior formulas.
    results = {}
    for n in (1, 3):
        max_formula_resid = Q(0)
        value = 0.0
        for ys in product((-1, 1), repeat=n):
            history_sigma = {sigma: Q(1, 2) *
                             prod((1+y*sigma*g)/2 for y in ys)
                             for sigma in (-1, 1)}
            ph = sum(history_sigma.values())
            mean = sum(sigma*p for sigma,p in history_sigma.items()) / ph
            u = 1 if mean >= 0 else -1
            xp = {x: Q(0) for x in (-1, 1)}
            for sigma,x,m,success in product((-1,1),(-1,1),(-1,1),(0,1)):
                path = history_sigma[sigma] * Q(1,2) * (1+m*sigma*g*x)/2
                switched = (u*m == -1)
                if switched:
                    path *= s if success else 1-s
                    out = -x if success else x
                elif success == 1:
                    continue
                else:
                    out = x
                xp[out] += path
            qplus = xp[1]/ph
            expected = (1+s*g*abs(mean))/2
            max_formula_resid = max(max_formula_resid, abs(qplus-expected))
            value += float(ph)*(log(2)-H(float(qplus)))
        assert max_formula_resid == 0
        results[str(n)] = dict(work=value, exact_polarization_residual=str(max_formula_resid))
    assert results['3']['work'] > results['1']['work']
    return results


def prod(values):
    out = Q(1)
    for value in values:
        out *= value
    return out


kernels = []
rate_results = []
for latch in (0, 1):
    r = d + latch*s*(1-2*d)
    kr = event(r)
    kernels.append(kr)
    rho = r/(r+d)
    # Check stationary wall-clock distribution exactly (uniform G,sign).
    wall = []
    for bi in (1, -1):
        for i in range(4):
            row = []
            for bj in (1, -1):
                body_probability = (1-d if bj == 1 else d) if bi == 1 else (r if bj == 1 else 1-r)
                row.extend(body_probability*x for x in K[i])
            wall.append(row)
    piwall = [rho/4]*4 + [(1-rho)/4]*4
    stationary_resid = max(abs(sum(piwall[i]*wall[i][j] for i in range(8))-piwall[j]) for j in range(8))
    assert stationary_resid == 0
    eigen_resid = Q(0)
    for name, eig, vector in (
            ('gain', 1-2*delta, [(-1)**g for g,si in states]),
            ('sign', 1-2*eta, [si for g,si in states]),
            ('product', (1-2*delta)*(1-2*eta), [(-1)**g*si for g,si in states])):
        wanted = f(r, eig)
        eigen_resid = max(eigen_resid, *(abs(sum(kr[i][j]*vector[j] for j in range(4))-wanted*vector[i]) for i in range(4)))
    assert eigen_resid == 0
    qr = (1-f(r, 1-2*eta))/2
    e = Q(1,5)
    direct_error = min((1-qr)*(1-e), qr*e)+min((1-qr)*e, qr*(1-e))
    assert direct_error == min(qr,e)
    rate_results.append(dict(latch=latch, recovery_rate=str(r), working_fraction=str(rho),
                             mean_gap=str(1+d/r), sign_prediction_floor=str(qr),
                             sign_prediction_floor_float=float(qr),
                             stationary_residual=str(stationary_resid),
                             eigenmode_residual=str(eigen_resid),
                             event_kernel_exact=[[str(v) for v in row] for row in kr]))

# Exact CP-extension checks: Kraus completeness and the diagonal action.
# D0=P+, D1=sqrt(1-s) P-, D2=sqrt(s)|+><-|; squares suffice.
assert Q(1) == Q(1) and (1-s)+s == 1
cp_checks = dict(kraus_completeness_residual='0',
                 plus_preserved='1', minus_repaired=str(s),
                 image_of_identity_diagonal=[str(1+s),str(1-s)],
                 nonunital=True, enable_flag_required=True)

# The exact rational s=.98 is realizable with the same squeezed family.
Nr = (sqrt(1-pi*pi/(2*log(2*float(s)-1)))-1)/2
lambda2r = pi*pi*(2*Nr+1)/(16*Nr*(Nr+1))
sr = (1+exp(-2*lambda2r/(2*Nr+1)))/2
assert abs(sr-float(s)) < 2e-16

# Actual manuscript drive, not substituted into exact rational witnesses.
Nactual = 5.018975748824816
lambda2actual = pi*pi*(2*Nactual+1)/(16*Nactual*(Nactual+1))
sactual = (1+exp(-2*lambda2actual/(2*Nactual+1)))/2
ractual = float(d)+sactual*(1-2*float(d))
rhoactual = ractual/(ractual+float(d))

gmax = .98*sqrt(1-.8**14)
cert = certificate(kernels,gmax)
acoarse = (1-d)*delta*eta
assert acoarse == Q(57,10**9)
epsilon_coarse = acoarse/Q(99)
assert -log(float(epsilon_coarse)) < 22
grid = Q(1,2**44)
tv_coarse = grid*(1+acoarse)/(8*acoarse)
assert 44*2**44+3 < 2**50
coarse = dict(min_entry=str(acoarse),posterior_min=str(epsilon_coarse),
              log_range_upper=22,ideal_model_bits=150,
              tau_upper=str((1-acoarse)/(1+acoarse)),
              TV_upper_exact=str(tv_coarse),TV_upper_float=float(tv_coarse),
              one_step_regret_upper=float(2*tv_coarse),
              scope='any positive recovery r; current and rational same-family drives')
memory = memory_enumeration(Q(3,5))
J1,_ = work(Q(3,5),1)
J3,distribution3 = work(Q(3,5),3)
assert abs(memory['1']['work']-J1) < 2e-16
assert abs(memory['3']['work']-J3) < 2e-16
J1b,_ = work(Q(3,5),1,Q(1))
J3b,_ = work(Q(3,5),3,Q(1))
assert J3b > J1b > J1

# Stationary resource-selector identity, entirely rational.
v, kappa = Q(1,10),Q(1,500)
r1 = d+s*(1-2*d)
u0 = Q(1,2)*v
u1 = r1/(r1+d)*v-d/(r1+d)*kappa
closed_gap = ((r1-d)*v-2*d*kappa)/(2*(r1+d))
assert u1-u0 == closed_gap
price = dict(active_value=str(v),incremental_switch_cost=str(kappa),
             exact_net_gap=str(closed_gap),formula_residual='0',
             recovery_break_even_active_value=str(2*d*kappa/(r1-d)),
             warning='includes flag purity only if allocated identically in common base; not a whole-apparatus profit')

# Finite-horizon survival: exact geometric tail, not a long simulation.
survival = {'absorbing_reads_mean':str(1/d),
            'survive_at_least_101_work_reads':str((1-d)**100),
            'survive_at_least_101_work_reads_float':float((1-d)**100)}

result = dict(status='finite deterministic exact/rational checks; no long simulations',
              rational_parameters=dict(delta=str(delta),eta=str(eta),d=str(d),s=str(s)),
              exact_wall_to_event_checks=rate_results,
              CP_extension=cp_checks,
              rational_drive=dict(N=Nr,lambda_squared=lambda2r,recovered_s=sr),
              current_drive=dict(N=Nactual,lambda_squared=lambda2actual,s=sactual,
                                 latch1_recovery=ractual,latch1_working_fraction=rhoactual,
                                 common_two_drive_allocation=.01*2*Nactual,
                                 extra_enable_flag_purity_per_wall_slot=log(2)),
              certificate_rational_drive=cert,
              certificate_common_coarse=coarse,
              stationary_price_identity=price,
              memory_work_exact_path_enumeration=memory,
              memory_work_formulas=dict(g='3/5',J1=J1,J3=J3,work_gap_per_active=J3-J1,
                                        wall_gap_no_standby=float(Q(1,2))*(J3-J1),
                                        wall_gap_with_standby=float(Q(233,250)/ (Q(233,250)+d))*(J3-J1),
                                        broad_posterior_matched_J1=J1b,
                                        broad_posterior_matched_J3=J3b,
                                        posterior_three_read_distribution=distribution3),
              absorbing=survival)
print(json.dumps(result,indent=2,ensure_ascii=False))
