#!/usr/bin/env python3
"""One finite training experiment; exact bounds and deterministic angular check."""
from fractions import Fraction as F
import math, random, json
rho=s=1.0
n=4096
L=256
q=F(1,1024)
mu=2*math.sqrt(2)/math.pi
M2=4+2*s*s/rho**2
e=math.sqrt(M2/n)+M2/L+float(q)/math.sqrt(2)
upper=s*s+rho*rho*(1-mu*mu+4*mu*e)
baseline=s*s+rho*rho/2
# A coarse rational certificate avoids relying on the last floating-point digit.
# Elementary 1.4142 < sqrt(2) < 1.4143 and 3.14159 < pi < 3.14160
# imply 0.9 < mu < 0.91; dropping 1/sqrt(2) only enlarges the error.
mu_lo,mu_hi=F(9,10),F(91,100)
assert F(39,1000)**2*n>6
e_rational=F(39,1000)+F(6,L)+q
risk_rational=2-mu_lo**2+4*mu_hi*e_rational
assert risk_rational<F(3,2)
def rnd(t):return math.floor(t+.5)  # ties toward +infinity, independent of U
def encode_observation(x1,x2,y):
    z1,z2=y*x1/rho,y*x2/rho
    length=math.hypot(z1,z2)
    scale=min(1.0,L/length) if length else 1.0
    return rnd(z1*scale/float(q)),rnd(z2*scale/float(q))
def select(A1,A2):
    # deterministic smallest-index tie; nonnegative coordinate gets positive sign
    j=0 if abs(A1)>=abs(A2) else 1
    return j,1 if (A1,A2)[j]>=0 else -1
integer_per_sample=math.floor(F(L)/q+F(1,2))
sum_bound=n*integer_per_sample
assert integer_per_sample==2**18 and sum_bound==2**30
assert -(2**31)<=-sum_bound and sum_bound<=2**31-1
assert n<2**13
# Endpoint clipping/rounding boundary checks; readouts are not generated from U here.
for z in ((L,0),(-L,0),(0,L),(0,-L),(10**10,0),(0,-10**10)):
    length=math.hypot(*z);a=min(1,L/length) if length else 1
    assert max(abs(rnd(v*a/float(q))) for v in z)<=integer_per_sample
grid=2**16
angular_mu=sum(max(abs(math.cos((i+.5)*2*math.pi/grid)),
                   abs(math.sin((i+.5)*2*math.pi/grid))) for i in range(grid))/grid
assert abs(angular_mu-mu)<1e-8
ideal=s*s+rho*rho*(1-mu*mu)
replay=s*s+rho*rho*(1+mu*mu)
# A deterministic independent four-code replay averages its alignment to zero.
replay_alignment=sum((math.cos((i+.5)*2*math.pi/grid)+
                      -math.cos((i+.5)*2*math.pi/grid)+
                      math.sin((i+.5)*2*math.pi/grid)+
                      -math.sin((i+.5)*2*math.pi/grid))/4
                     for i in range(grid))/grid
assert abs(replay_alignment)<1e-12
# Small fixed-seed finite-trainer diagnostic. Exact test risk conditional on U,w.
rng=random.Random(1729)
histories=128
rows=[]
maximum_partial_sum=0
for k in range(histories):
    angle=rng.random()*2*math.pi
    u=(math.cos(angle),math.sin(angle))
    acc=[0,0]
    for _ in range(n):
        x1,x2=rng.gauss(0,1),rng.gauss(0,1)
        y=rho*(u[0]*x1+u[1]*x2)+s*rng.gauss(0,1)
        z=encode_observation(x1,x2,y)
        acc[0]+=z[0];acc[1]+=z[1]
        maximum_partial_sum=max(maximum_partial_sum,abs(acc[0]),abs(acc[1]))
        assert max(map(abs,acc))<=sum_bound
    j,sign=select(*acc)
    alignment=sign*u[j]
    risk=s*s+rho*rho*(1+mu*mu-2*mu*alignment)
    # Freeze preserves the same fixed U,w coefficient risk exactly.
    frozen=s*s+rho*rho*sum((u[i]-mu*(sign if i==j else 0))**2 for i in range(2))
    assert abs(risk-frozen)<1e-12
    rows.append((risk,alignment))
average=sum(r[0] for r in rows)/histories
se=(sum((r[0]-average)**2 for r in rows)/(histories*(histories-1)))**.5
report={
 "parameters":{"rho":rho,"s":s,"n":n,"L":L,"q":str(q)},
 "analytic":{"mu":mu,"mu_squared":mu*mu,"M2":M2,
 "e_bound":e,"risk_upper_bound":upper,"history_independent_oracle_lower_bound":baseline,
 "strict_certified_margin":baseline-upper,"ideal_four_code_risk":ideal,
 "coarse_exact_rational_risk_upper":str(risk_rational),
 "independent_replay_original_coefficient_risk":replay,
 "independent_replay_oracle_coefficient_risk":baseline},
 "deterministic_angle_check":{"grid_points":grid,"mu_midpoint_sum":angular_mu,
 "absolute_mu_error":abs(angular_mu-mu),"independent_code_replay_alignment":replay_alignment},
 "registers":{"quantized_coordinate_abs_bound":integer_per_sample,
 "integer_sum_abs_bound":sum_bound,"signed_accumulator_bits_each":32,
 "counter_bits":13,"training_accumulator_counter_bits":77,
 "separately_allocated_output_bits":2,"test_retained_code_bits":2,
 "observed_largest_partial_sum":maximum_partial_sum,
 "analog_front_end_arithmetic_readonly_constants_and_control_not_in_register_count":True},
 "finite_training_diagnostic":{"seed":1729,"independent_histories":histories,
 "risk_mean_exactly_integrated_over_test_input":average,"standard_error_over_histories":se,
 "freeze_preserves_same_static_risk":True,
 "diagnostic_not_the_finite_n_proof":True},
 "all_checks_pass":upper<baseline and abs(angular_mu-mu)<1e-8
}
assert report["all_checks_pass"]
print(json.dumps(report,indent=2))

