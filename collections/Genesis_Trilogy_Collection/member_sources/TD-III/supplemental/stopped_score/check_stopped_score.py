"""Independent finite-tree check of the conditional-average score budget.

No existing research code is imported. A two-point mixing law tests the
reading-score identity only; it is not a van-Trees test for a discrete prior.
"""
from itertools import product
from pathlib import Path
from fractions import Fraction as F
import json,math

J=.30
theta=(0.,3.)
prior=(.1,.9)
def logistic(x):return 1/(1+math.exp(-x))
def design(weights):
    # The instrument is chosen from the acquired posterior, never theta.
    average=sum(w*4*logistic(2*t)*(1-logistic(2*t)) for w,t in zip(weights,theta))
    return 2. if average<=J else .5

leaves=[]; nodes=[]
def visit(history,weights,probabilities,scores,information):
    a=design(weights)
    ps=[logistic(a*t) for t in theta]
    fi=[a*a*p*(1-p) for p in ps]
    average=sum(w*i for w,i in zip(weights,fi))
    assert average<=J+1e-15
    nodes.append(dict(history=history,design=a,bayesian_conditional_fisher=average,pointwise_fisher=fi))
    for y in (0,1):
        likelihood=[p if y else 1-p for p in ps]
        joint=[w*l for w,l in zip(probabilities,likelihood)]
        norm=sum(joint)
        newscore=[v+a*(y-p) for v,p in zip(scores,ps)]
        newinfo=[v+i for v,i in zip(information,fi)]
        h=history+[y]
        if y or len(h)==4:
            leaves.append((joint,newscore,newinfo,len(h)))
        else:
            visit(h,[w/norm for w in joint],joint,newscore,newinfo)
visit([],list(prior),list(prior),[0.,0.],[0.,0.])
mass=sum(sum(w) for w,s,i,n in leaves)
score_mean=sum(sum(w*x for w,x in zip(ws,s)) for ws,s,i,n in leaves)
joint_information=sum(sum(w*x*x for w,x in zip(ws,s)) for ws,s,i,n in leaves)
sum_conditional_information=sum(sum(w*x for w,x in zip(ws,i)) for ws,s,i,n in leaves)
expected_count=sum(sum(ws)*n for ws,s,i,n in leaves)
assert abs(mass-1)<1e-14 and abs(score_mean)<1e-14
assert abs(joint_information-sum_conditional_information)<1e-14
assert joint_information<=J*expected_count
assert any(max(n['pointwise_fisher'])>J for n in nodes)

# A random instrument label is known before acquisition, independent of theta.
# At theta=0 the Bernoulli channel p=(1+a theta)/2 has FI=a^2.
# Only the strong labelled channel is actually read. Its unconditional
# potential-read average cannot be substituted for its selected budget.
unconditional=F(1,10)*1+F(9,10)*F(1,100)
selected_information=F(1,10)
count=F(1,10)
assert unconditional<F(3,10) and selected_information>F(3,10)*count
out=dict(nodes=nodes,total_probability=mass,score_mean=score_mean,joint_information=joint_information,expected_sum_fisher=sum_conditional_information,isometry_residual=joint_information-sum_conditional_information,expected_count=expected_count,bayesian_budget=J*expected_count,unconditional_budget_counterexample=dict(potential_average=str(unconditional),selected_information=str(selected_information),incorrect_J_times_count=str(F(3,10)*count)))
path=Path(__file__).with_name('stopped_average.json')
path.write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out,indent=2))
