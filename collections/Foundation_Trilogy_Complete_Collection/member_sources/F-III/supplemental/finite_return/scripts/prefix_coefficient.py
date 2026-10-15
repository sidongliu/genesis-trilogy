"""All seven gates, all three old purifiers: shortest-path insertion audit.

No locally prepared post-acquisition probe is substituted. Gate prefixes
conjugate both the actual retained input and the output observable.
"""
import json, math
from fractions import Fraction
from pathlib import Path
import numpy as np
from full_return import assemble,local,Z,N

out={}
for s in (-1,1):
    for policy in ('learned','odd','even'):
        sys=assemble(s,policy)
        prefix=np.eye(64,dtype=complex); prefixes=[]
        for j in range(-3,14):
            if j>-3: prefix=sys['gates'].get(j-1,np.eye(64))@prefix
            prefixes.append(prefix.copy())
        psi=sys['data']; obs=prefixes[-1].conj().T@local(N,4)@prefixes[-1]
        values=[]
        for j,w in enumerate(prefixes):
            z=w.conj().T@local(Z,2)@w
            values.append(float(2*np.vdot(psi,obs@z@psi).imag))
        expected=[0.]*8+[-.64*s]*8+[0.]
        assert np.max(abs(np.array(values)-expected))<2e-15,(s,policy,values)
        factor=Fraction(-16*s,25)*sum(Fraction(math.factorial(j+23),math.factorial(j)) for j in range(8,16))/math.factorial(16)/math.factorial(40)
        out[f'{s}_{policy}']=dict(prefix_derivatives=values,exact_coefficient_over_lambda_F_w_J32=str(factor),numeric_factor=float(factor))
        print(s,policy,values,float(factor))

# Initial packet endpoint probability and the complete t^56 radiation coefficient.
sys=assemble(1,'learned'); wf=float(abs(sys['packet'][27])**2)
v0=math.sqrt((.95-.1375)/10)
for s in (-1,1):
    ps=.9 if s==1 else .1
    aR=wf*.8**16*ps/math.factorial(8)**2
    DP=-2/(2*math.pi)*.5*.01*v0*aR/(10**2*20*18*19*20)
    F=-.01*DP/(2*22*23)
    local_factor=out[f'{s}_learned']['numeric_factor']
    out[f'physical_{s}']=dict(wf=wf,aR=aR,DP=DP,F=F,KX=local_factor*.05*F*wf*.8**32)
(Path(__file__).resolve().parents[1]/'results'/'prefix.json').write_text(json.dumps(out,indent=2)+'\n')
