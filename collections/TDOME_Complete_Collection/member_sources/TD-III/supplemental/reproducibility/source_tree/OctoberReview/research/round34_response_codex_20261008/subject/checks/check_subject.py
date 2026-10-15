"""Independent round34 checks: short repair witness and exact timing certificate.

No import of Claude's scripts. The repair continuation has two wall slots,
so the Bellman difference is computed both by its formula and explicit paths.
All transcendental work uses 80-digit mpmath; exact timing uses Fractions.
"""
from fractions import Fraction as F
import json
import mpmath as mp

mp.mp.dps = 80
gm = mp.mpf(3) / 20
s = mp.mpf(49) / 50
d = mp.mpf(1) / 20

def posterior(gp, n, k):
    vals = [(sign, gain, ((1+sign*gain)/2)**k * ((1-sign*gain)/2)**(n-k))
            for sign in (1, -1) for gain in (gm, gp)]
    norm = sum(v for _, _, v in vals)
    return [(sign, gain, v/norm) for sign, gain, v in vals]

def confidence(gp, n, k):
    return sum(v for sign, _, v in posterior(gp, n, k) if sign == 1)

def equal_confidence(gp):
    return confidence(gp, 2, 2) - confidence(gp, 10, 8)

lo = mp.mpf('0.959375749239844')
hi = mp.mpf('0.959375749239846')
assert equal_confidence(lo) * equal_confidence(hi) < 0
gp = mp.findroot(equal_confidence, (lo, hi))

def confidence_exact(x, n, k):
    raw = [(sg, ((1+sg*g)/2)**k*((1-sg*g)/2)**(n-k))
           for sg in (1,-1) for g in (F(3,20),x)]
    return sum(v for sg,v in raw if sg == 1)/sum(v for _,v in raw)

bracket_exact = (F(str(lo)), F(str(hi)))
bracket_signs = [confidence_exact(x,2,2)-confidence_exact(x,10,8)
                 for x in bracket_exact]
assert bracket_signs[0] < 0 < bracket_signs[1]

class Interval:
    """Rational outward enclosure; no floating-point bound arithmetic."""
    def __init__(self,a,b=None):
        self.a=F(a); self.b=F(a if b is None else b)
    def __add__(self,other):
        o=other if isinstance(other,Interval) else Interval(other)
        return Interval(self.a+o.a,self.b+o.b)
    __radd__=__add__
    def __neg__(self): return Interval(-self.b,-self.a)
    def __sub__(self,o): return self+-o if isinstance(o,Interval) else self+(-F(o))
    def __rsub__(self,o): return -self+o
    def __mul__(self,other):
        o=other if isinstance(other,Interval) else Interval(other)
        vals=[self.a*o.a,self.a*o.b,self.b*o.a,self.b*o.b]
        return Interval(min(vals),max(vals))
    __rmul__=__mul__
    def __truediv__(self,other):
        o=other if isinstance(other,Interval) else Interval(other)
        assert o.a>0 or o.b<0
        return self*Interval(1/o.b,1/o.a)
    def __pow__(self,n):
        v=Interval(1)
        for _ in range(n):v=v*self
        return v
    def abs(self):
        return Interval(0 if self.a<=0<=self.b else min(abs(self.a),abs(self.b)),
                        max(abs(self.a),abs(self.b)))
    def positive(self):return Interval(max(F(0),self.a),max(F(0),self.b))
    def bounds(self):return [float(self.a),float(self.b)]

def rational_next_task(n,k,kappa,proc):
    g=Interval(*bracket_exact)
    def stats(nn,kk):
        raw=[(sg,gg,((1+sg*gg)/2)**kk*((1-sg*gg)/2)**(nn-kk))
             for sg in (1,-1) for gg in (Interval(F(3,20)),g)]
        den=sum(v for _,_,v in raw)
        mean=sum(sg*gg*v for sg,gg,v in raw)/den
        return mean
    mean=stats(n,k)
    py=(1+mean)/2
    val=Interval(0)
    for yy,prob in ((1,py),(0,1-py)):
        x=(F(49,50)*stats(n+1,k+yy)).abs()
        # F(x)=ln2-h((1+x)/2), a positive power series.
        lower=sum(x**(2*j)/F(2*j*(2*j-1)) for j in range(1,25))
        rem=x.b**50/(F(50*49)*(1-x.b*x.b))
        food=Interval(lower.a,lower.b+rem)
        val+=prob*((food-kappa/2).positive()-proc)
    return val

rational_certificates=[]
for kappa,proc in ((F(2,25),F(0)),(F(9,4000),F(3,25))):
    entries=[]
    for n,k in ((2,2),(10,8)):
        a=rational_next_task(n,k,kappa,proc)
        r1=F(1,20)+F(49,50)*F(9,10)
        diff=F(1,20)*((r1-F(1,20))*a-kappa*(2-r1))
        entries.append(dict(n=n,k=k,next_task=a.bounds(),q_difference=diff.bounds()))
        assert (diff.a>0 if n==2 else diff.b<0)
    rational_certificates.append(dict(kappa=str(kappa),processing=str(proc),entries=entries))

def entropy(p):
    return -p*mp.log(p) - (1-p)*mp.log(1-p)

def expected_next_task(gp, n, k, kappa, proc):
    p = posterior(gp, n, k)
    py = sum(v*(1+sign*gain)/2 for sign, gain, v in p)
    val = mp.mpf(0)
    for y, prob in ((1, py), (0, 1-py)):
        pp = posterior(gp, n+1, k+y)
        mean = sum(v*sign*gain for sign, gain, v in pp)
        food = mp.log(2)-entropy((1+s*abs(mean))/2)
        val += prob*(max(mp.mpf(0), food-kappa/2)-proc)
    return val

def short_q(gp, n, k, kappa, proc, latch):
    """Enumerate working t+1, failure, recovery t+2 and disabled terminal."""
    r = d+latch*s*(1-2*d)
    a = expected_next_task(gp, n, k, kappa, proc)
    # Common t+1-working continuations cancel. At t+1 no latch is useful:
    # failure at t+2 gives no later opportunity within H=t+2.
    return d*(r*a-latch*kappa*(1+(1-r)))

repair = []
for kappa, proc in ((mp.mpf('0.08'), mp.mpf(0)),
                    (mp.mpf('0.00225'), mp.mpf('0.12'))):
    pair = []
    for n, k in ((2, 2), (10, 8)):
        a = expected_next_task(gp, n, k, kappa, proc)
        r1 = d+s*(1-2*d)
        diff = d*((r1-d)*a-kappa*(2-r1))
        enum = short_q(gp, n, k, kappa, proc, 1)-short_q(gp, n, k, kappa, proc, 0)
        assert abs(enum-diff) < mp.mpf('1e-70')
        pair.append(dict(n=n, k=k, next_task=str(a), q_difference=str(diff)))
    assert mp.mpf(pair[0]['q_difference']) > 0 > mp.mpf(pair[1]['q_difference'])
    repair.append(dict(kappa=str(kappa), processing=str(proc), histories=pair))

# Renewal stopping on a=1 or five quiet innovations, exactly at eta=d=1/10.
p = F(9, 50)
z = 1-p
u = F(40, 41)
w = F(1, 10)
K = 5
duration = sum(z**j for j in range(K))
cost = w+sum(z**j*(1-u**j)/2 for j in range(1,K))
rho = cost/duration
v = {j:w for j in range(K, K+2)}
for j in range(K-1, 0, -1):
    v[j] = (1-u**j)/2-rho+p*w+z*v[j+1]
assert -rho+p*w+z*v[1] == 0
assert all(v[j] < w for j in range(1, K))
assert (1-u**K)/2 > rho
assert F(1,2)-rho > 0

out = {
    'gain_root':str(gp),
    'root_bracket':[str(lo),str(hi)],
    'root_equation_at_bracket':[str(equal_confidence(lo)),str(equal_confidence(hi))],
    'confidence':str(confidence(gp,2,2)),
    'gain_means':[str(sum(v*g for _,g,v in posterior(gp,n,k))) for n,k in ((2,2),(10,8))],
    'legal_history_gate_probabilities':{
        'slots_1_and_10':str(d*d*(1-d)**7),
        'all_working':str((1-d)**9)},
    'repair':repair,
    'rational_interval_certificates':rational_certificates,
    'root_bracket_signs_exact':[('-' if x<0 else '+') for x in bracket_signs],
    'sqrtlaw':{
        'rho_fraction':str(rho), 'rho_float':float(rho),
        'duration':str(duration), 'cycle_cost':str(cost),
        'bias_function':{str(k):str(x) for k,x in v.items()},
        'continuation_slacks':{str(k):str(w-v[k]) for k in range(1,K)},
        'stopping_slack':str((1-u**K)/2-rho)}}

# A neutral adaptive gate does not give a finite-H fixed-count mixture.
# State family: Ber(1/2), Ber(9/10), equal prior. H=4, failure h=1/3.
# On the first acquired outcome choose r=1 if it is 1, otherwise r=0.
def adaptive_gate_risk():
    risks=[]
    count_prob=[mp.mpf(0)]*5
    ps=(mp.mpf('0.5'),mp.mpf('0.9'))
    def visit(t,working,latch,ys,weights):
        if t==5:
            tot=sum(weights)
            mean=sum(a*p for a,p in zip(weights,ps))/tot
            risks.append(tot*entropy(mean)-sum(a*entropy(p) for a,p in zip(weights,ps)))
            count_prob[len(ys)]+=tot
            return
        branches=[]
        if working:
            for y in (0,1):
                ww=[a*(p if y else 1-p) for a,p in zip(weights,ps)]
                rr=y if not ys else latch
                branches.append((ys+(y,),ww,rr))
        else:
            branches.append((ys,weights,latch))
        for newys,ww,rr in branches:
            next_work=mp.mpf(2)/3 if working else mp.mpf(rr)
            for nw,prob in ((True,next_work),(False,1-next_work)):
                if prob:visit(t+1,nw,rr,newys,[a*prob for a in ww])
    visit(1,True,0,(),[mp.mpf('0.5')]*2)
    rn=[]
    for n in range(5):
        value=mp.mpf(0)
        for k in range(n+1):
            ww=[mp.mpf('0.5')*mp.binomial(n,k)*p**k*(1-p)**(n-k) for p in ps]
            tot=sum(ww)
            mean=sum(a*p for a,p in zip(ww,ps))/tot
            value+=tot*entropy(mean)-sum(a*entropy(p) for a,p in zip(ww,ps))
        rn.append(value)
    true=sum(risks)
    mixture=sum(a*b for a,b in zip(count_prob,rn))
    assert abs(true-mixture)>mp.mpf('0.001')
    return dict(actual_risk=str(true),selected_count_mixture=str(mixture),
                count_probabilities=[str(x) for x in count_prob])
out['neutral_gate_counterexample']=adaptive_gate_risk()
print(json.dumps(out, indent=2))
