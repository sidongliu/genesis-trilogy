"""Independent finite-sum/Markov check of the supplied IMP-RG protocol."""
import json
from pathlib import Path
import numpy as np
from scipy.stats import binom, norm

n, cap, epsilon = 4000, 16, 1e-4
angles = 2*np.pi*np.arange(64)/64
lap = 4*np.sin(angles[:,None]/2)**2+4*np.sin(angles[None,:]/2)**2
aniso = 16*np.sin(angles[:,None]/2)**4+16*np.sin(angles[None,:]/2)**4
v0 = np.mean(1/(.1+lap))
v1 = np.mean(1/(.1+lap+aniso/12))
thresholds = np.array([1.5222951844903065, 1.445716072545341])
p0 = 2*norm.sf(thresholds)
p1 = 2*norm.sf(thresholds*np.array([np.sqrt(.9),np.sqrt(v0/v1)]))

def bayes(m,p,r):
    if m==0:
        return .5
    u = np.log(r/p)
    v = np.log((1-r)/(1-p))
    crossing = -m*v/(u-v)
    if r>p:
        k = int(np.ceil(crossing))
        return .5*(binom.sf(k-1,m,p)+binom.cdf(k-1,m,r))
    k = int(np.floor(crossing))
    return .5*(binom.cdf(k,m,p)+binom.sf(k,m,r))

errors = [np.array([bayes(m,p0[b],p1[b]) for m in range(n+1)]) for b in range(2)]
values = (errors[0]+errors[1][::-1])/2
split = int(np.argmin(values))
fixed = float(values[split])
det0 = np.array([binom.sf(552,n,p0[0]), binom.cdf(553,n,p0[1])])
det1 = np.array([binom.sf(552,n,p1[0]), binom.cdf(553,n,p1[1])])
states = [(j,b,c) for j in range(2) for b in range(2) for c in range(1,cap+1)]
ix = {s:i for i,s in enumerate(states)}
matrix = np.zeros((len(states),len(states)))
loss = np.empty(len(states))
for i,(j,b,c) in enumerate(states):
    alt = det1[b] if b==j else det0[b]
    detect = (det0[b]+alt)/2
    loss[i] = (det0[b]+1-alt)/2
    routes = [(b,min(cap,c+1),detect),(b,c,(1-detect)/2)]
    routes.append((b,c-1,(1-detect)/2) if c>1 else (1-b,1,(1-detect)/2))
    for bn,cn,weight in routes:
        matrix[i,ix[(j,bn,cn)]] += weight*(1-epsilon)
        matrix[i,ix[(1-j,bn,cn)]] += weight*epsilon
linear = matrix.T-np.eye(len(states))
linear[-1,:] = 1
rhs = np.zeros(len(states)); rhs[-1] = 1
stationary = np.linalg.solve(linear,rhs)
lh = float(stationary@loss)
out = dict(p0=p0.tolist(),p1=p1.tolist(),best_fixed_split=split,
           exact_expression_numerical_minimum=fixed,history_error=lh,
           price_difference=fixed-lh,
           stationarity_residual=float(np.max(abs(stationary@matrix-stationary))),
           qualification='binary64 evaluation of exact finite formulas; no interval certification')
Path(__file__).with_name('HISTORY_PRICE.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out,indent=2))
