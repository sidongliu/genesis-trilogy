#!/usr/bin/env python3
"""Exact checks of the specified reversible gates; no timestamps in readout data."""
from fractions import Fraction as Q
from itertools import product, permutations
from collections import defaultdict
import json, math

N_STATIONS = 3
ORDERS = list(permutations(range(1,N_STATIONS+1)))
EPS = (Q(1,10),Q(1,5),Q(3,10))
def tpos(i): return 2+3*(i-1)
def kpos(i): return 3+3*(i-1)
def rpos(i): return 4+3*(i-1)
def station(bits,i):
    bits[rpos(i)] ^= bits[0]            # actual CNOT(S -> R_i)
    bits[0],bits[tpos(i)] = bits[tpos(i)],bits[0]  # actual SWAP
def inverse_station(bits,i):
    bits[0],bits[tpos(i)] = bits[tpos(i)],bits[0]
    bits[rpos(i)] ^= bits[0]
def preparation(z,noise):
    out=[0]*(3*N_STATIONS+2)
    out[0]=out[1]=z[0]
    for i in range(1,N_STATIONS+1):
        out[tpos(i)]=out[kpos(i)]=z[i]
        out[rpos(i)]=noise[i-1]
    return out
def experiment(order,eps=EPS,common_noise=False):
    law, full = defaultdict(Q),defaultdict(Q)
    corr=[[Q(0) for _ in range(N_STATIONS)] for _ in range(N_STATIONS+1)]
    spent=[[Q(0) for _ in range(N_STATIONS)] for _ in range(N_STATIONS+1)]
    noises=product((0,1),repeat=1 if common_noise else N_STATIONS)
    noises=list(noises)
    for z in product((0,1),repeat=N_STATIONS+1):
        for ns in noises:
            noise=ns*N_STATIONS if common_noise else ns
            weight=Q(1,2**(N_STATIONS+1))
            for j,x in enumerate(ns):
                e=eps[0] if common_noise else eps[j]
                weight *= e if x else 1-e
            bits=preparation(z,noise)
            for i in order: station(bits,i)
            keys=(bits[1],)+tuple(bits[kpos(i)] for i in range(1,N_STATIONS+1))
            read=tuple(bits[rpos(i)] for i in range(1,N_STATIONS+1))
            law[(keys,read)] += weight
            full[tuple(bits)] += weight
            for i in range(N_STATIONS+1):
                for j in range(1,N_STATIONS+1):
                    corr[i][j-1]+=weight*(-1)**(keys[i]+read[j-1])
                    spent[i][j-1]+=weight*(-1)**(keys[i]+bits[tpos(j)])
    assert sum(law.values())==1
    pred=dict(zip(order,(0,)+order[:-1]))
    for j in range(1,N_STATIONS+1):
        for i in range(N_STATIONS+1):
            e=eps[0] if common_noise else eps[j-1]
            assert corr[i][j-1] == (1-2*e if i==pred[j] else 0)
            assert spent[i][j-1] == (1 if i==pred[j] else 0)
    return dict(law),corr,spent

def recover(corr,threshold):
    pred={}
    for j in range(1,N_STATIONS+1):
        found=[i for i in range(N_STATIONS+1) if corr[i][j-1]>threshold]
        if len(found)!=1: return None
        pred[j]=found[0]
    nxt={}
    for j,i in pred.items():
        if i in nxt:return None
        nxt[i]=j
    chain=[]; i=0
    while i in nxt:
        i=nxt[i]
        if i in chain:return None
        chain.append(i)
    return tuple(chain) if len(chain)==N_STATIONS else None
def marginal_y(law):
    out=defaultdict(Q)
    for (_,ys),w in law.items():out[ys]+=w
    return dict(out)
def signature(law): return tuple(sorted((k,v) for k,v in law.items() if v))
# Full basis permutations check actual CNOT+SWAP and its inverse.
basis_size=2**(3*N_STATIONS+2)
for order in ORDERS:
    outputs=set()
    for bits0 in product((0,1),repeat=3*N_STATIONS+2):
        bits=list(bits0)
        for i in order:station(bits,i)
        outputs.add(tuple(bits))
        for i in reversed(order):inverse_station(bits,i)
        assert tuple(bits)==bits0
    assert len(outputs)==basis_size

# General static pointer-controlled rotations: exact full 16x16 operator.
rot=((Q(3,5),Q(4,5)),(Q(5,13),Q(12,13)),(Q(8,17),Q(15,17)))
def static_operator(order):
    out=[]
    for col in range(16):
        amp={col:Q(1)}
        for i in order:
            c,s=rot[i-1]; nxt=defaultdict(Q); mask=1<<(i-1)
            for b,a in amp.items():
                if not b&8:nxt[b]+=a;continue
                bit=bool(b&mask)
                nxt[b]+=c*a
                nxt[b^mask]+=(-s if bit else s)*a
            amp=nxt
        out.append(tuple(amp.get(row,Q(0)) for row in range(16)))
    return tuple(out)
static=[static_operator(p) for p in ORDERS]
assert all(a==static[0] for a in static)

all_laws=[]; cases=[]
gmin=min(1-2*e for e in EPS)
for order in ORDERS:
    law,corr,spent=experiment(order)
    assert recover(corr,gmin/2)==order
    all_laws.append(law)
    cases.append({"private_check_order":order,
                  "public_row_labels":list(range(4)),
                  "public_column_labels":[1,2,3],
                  "endpoint_C":[[str(x) for x in row] for row in corr],
                  "recovered_chain":recover(corr,gmin/2)})
assert len({signature(x) for x in all_laws})==6
assert all(marginal_y(x)==marginal_y(all_laws[0]) for x in all_laws)
fair=[experiment(p,(Q(1,2),)*3) for p in ORDERS]
assert len({signature(x[0]) for x in fair})==1
assert all(recover(x[2],Q(1,2))==p for x,p in zip(fair,ORDERS))
one_fair=[experiment(p,(Q(1,2),EPS[1],EPS[2]))[0] for p in ORDERS]
assert len({signature(x) for x in one_fair})==6
two_fair=[experiment(p,(Q(1,2),Q(1,2),EPS[2]))[0] for p in ORDERS]
assert len({signature(x) for x in two_fair})==3
assert signature(experiment((1,3,2),(Q(1,2),Q(1,2),EPS[2]))[0]) == signature(experiment((2,1,3),(Q(1,2),Q(1,2),EPS[2]))[0])
for p in ORDERS:
    assert recover(experiment(p,(Q(1,5),)*3,common_noise=True)[1],Q(3,10))==p
zeta=.01
safe_trials=1+math.floor(8/float(gmin*gmin)*math.log(2*N_STATIONS*(N_STATIONS+1)/zeta))
report={
 "n":3,"epsilon":[str(e) for e in EPS],"gate_basis_states_per_order":basis_size,
 "all_six_full_gate_permutations_bijective_and_exactly_inverted":True,
 "static_full_controlled_rotation_operators_identical_for_all_orders":True,
 "cases":cases,
 "positive_bias_endpoint_KY_laws_distinct":6,
 "Y_only_law_independent_of_order":True,
 "all_fair_KY_laws_distinct":1,
 "all_fair_spent_T_reading_recovers_all_orders":True,
 "one_fair_KY_laws_distinct":6,
 "two_fair_KY_laws_distinct":3,
 "two_fair_ambiguous_orders":[[1,3,2],[2,1,3]],
 "common_label_independent_noise_does_not_break_identification":True,
 "sample_certificate":{"g_min":str(gmin),"failure_target":zeta,"safe_integer_N":safe_trials,
   "union_Hoeffding_bound":2*N_STATIONS*(N_STATIONS+1)*math.exp(-safe_trials*float(gmin*gmin)/8),
   "threshold":str(gmin/2),"strict_error_event":"max abs(C_hat-C) < g_min/2"},
 "resources_per_trial":{"modeled_qubits":3*N_STATIONS+2,"independent_fair_choices":4,"biased_R":3},
 "all_checks_pass":True
}
print(json.dumps(report,indent=2))

