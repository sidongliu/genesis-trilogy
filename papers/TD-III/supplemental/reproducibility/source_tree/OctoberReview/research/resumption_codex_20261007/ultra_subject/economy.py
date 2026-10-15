"""Exact bounded-block economy; one drive state prices both error and energy."""
import json, math
import numpy as np
from scipy.stats import binom
from scipy.optimize import minimize_scalar
from pathlib import Path

ROOT=Path(__file__).resolve().parent
def H(p):
    p=np.clip(p,1e-15,1-1e-15)
    return -p*np.log(p)-(1-p)*np.log1p(-p)
def S(r,x): return H((1+np.sqrt(r*r+(1-r*r)*x*x))/2)
def drive(N,family):
    if family=='coherent': V=1.; lam=math.pi**2/(16*N)
    else: V=1/(2*N+1); lam=math.pi**2*(2*N+1)/(16*N*(N+1))
    alpha2=math.pi**2/(16*lam)
    occupation=alpha2+(V+1/V-2)/4
    return (1+math.exp(-2*lam*V))/2,lam,occupation

def economy(n,theta,N,family='gaussian',m=32,L=4096,body=False):
    r=.8**n; h=math.sqrt(1-r*r); gs=np.array([.90*h,.98*h])
    ks=np.arange(m+1); lk=np.array([binom.pmf(ks,m,(1+g)/2) for g in gs])
    weights=lk.sum(axis=0)/2; posterior=lk/lk.sum(axis=0)
    gb=posterior.T@gs
    s,lam,occupation=drive(N,family)
    income=math.log(2)-H((1+s*gb)/2)
    reset=math.log(2)-S(r,0)+S(r,gb)
    food=income-reset-theta*(N+.5*lam)
    # References: thermal preparation, separate outcome erasure, counter erasure.
    calibration=(2*m*math.log(2)+math.log(m+1))/L
    bcost=0.; bloss=0.
    if body:
        x=np.zeros(2); c=.9
        for _ in range(L):
            xb=posterior.T@x; yb=posterior.T@(gs*x)
            probs=np.stack(((1-yb)/2,(1+yb)/2))
            bpost=np.stack(((xb-gb)/(1-yb),(xb+gb)/(1+yb)))
            chi=S(r,xb)-np.sum(probs*S(r,bpost),axis=0)
            erasure=H((1+yb)/2)-chi
            bcost+=float(weights@(erasure+theta*(N+(1-yb)*lam/2)))/L
            x=c*((1-s)*x+s*gs)
            bloss+=float(np.mean((1-x)/2))/L
    return {'net':float(weights@food)-calibration-bcost,'harvest':float(weights@income),
            'food_reset':float(weights@reset),'calibration':calibration,
            'food_drive':theta*(N+.5*lam),'body_cost':bcost,'body_loss':bloss,
            's':s,'lambda2':lam,'N':N,'occupation_residual':abs(occupation-N),
            'record_purity_input':(4 if body else 2)*n*math.log(2),
            'reference_purity_input':m*n*math.log(2)/L,
            'total_record_purity_input':((4 if body else 2)*n+m*n/L)*math.log(2),
            'posterior_normalization':float(np.max(abs(posterior.sum(axis=0)-1))),
            'probability_normalization':float(abs(weights.sum()-1))}

def main():
    rows=[]
    for n,theta in [(3,.01),(5,.01),(7,.01),(5,.05),(7,.05),(10,.1)]:
        for family in ['coherent','gaussian']:
            # Food-only optimum; body is evaluated with this SAME state.
            opt=minimize_scalar(lambda x:-economy(n,theta,x,family)['net'],bounds=(.15,50),method='bounded')
            a=economy(n,theta,opt.x,family); b=economy(n,theta,opt.x,family,body=True)
            rows.append({'n':n,'theta':theta,'family':family,'food':a,'food_body':b})
    encoded=json.dumps(rows,indent=2)
    (ROOT/'economy_output.json').write_text(encoded+'\n')
    print(encoded)
if __name__=='__main__': main()
