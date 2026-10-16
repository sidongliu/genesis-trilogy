"""Finite-oscillator check of the conditional G3 work identity, not a field proof."""
import json
import numpy as np
from scipy.linalg import expm

X=np.array([[0,1],[1,0]],complex)
Z=np.diag([1.,-1.]).astype(complex)
Y=np.array([[0,-1j],[1j,0]],complex)
C=np.diag([1.,2.]); lam=.21; rows=[]
for n in (24,40):
    b=np.diag(np.sqrt(np.arange(1,n)),1); Q=b+b.T
    N=np.diag(np.arange(n)); vacuum=np.zeros((n,n));vacuum[0,0]=1
    for a in (np.array([.3,.4]),np.array([1.,0]),np.array([0.,1.])):
        v=C@a; A=v[0]*X+v[1]*Z; q=float(v@v)
        U=expm(-1j*lam*np.kron(A,Q))
        rho=np.kron(np.eye(2)/2,vacuum); out=U@rho@U.conj().T
        work=float(np.trace(out@np.kron(np.eye(2),N)).real)
        shell=float(np.max(np.abs(A@A-q*np.eye(2))))
        J=.7*Y-Y@A
        lorentz=float(np.max(np.abs(J@J-(.7**2-q)*np.eye(2))))
        rows.append(dict(cutoff=n,a=a.tolist(),work=work,predicted=lam**2*q,
                         work_error=abs(work-lam**2*q),shell_error=shell,lorentz_error=lorentz))
        assert abs(work-lam**2*q)<1e-12 and shell<1e-12 and lorentz<1e-12
# Average pricing alone hides a non-Clifford coupling: A=Z+I.
fail=[]
n=24
b=np.diag(np.sqrt(np.arange(1,n)),1); Q=b+b.T
N=np.diag(np.arange(n)); vacuum=np.zeros((n,n)); vacuum[0,0]=1
A=Z+np.eye(2); U=expm(-1j*lam*np.kron(A,Q))
for name,rhoB in [('up',np.diag([1.,0.])),('down',np.diag([0.,1.])),('mixed',np.eye(2)/2)]:
    rho=np.kron(rhoB,vacuum); out=U@rho@U.conj().T
    work=float(np.trace(out@np.kron(np.eye(2),N)).real)
    predicted=lam**2*float(np.trace(rhoB@A@A).real)
    fail.append(dict(state=name,work=work,predicted=predicted))
    assert abs(work-predicted)<1e-12
assert abs(fail[0]['work']-fail[1]['work'])>.1
print(json.dumps({'status':'pass','scope':'finite oscillator, independent vacuum; no spatial field or autonomous clock simulation','C':C.tolist(),'K_over_kappa':(C.T@C).tolist(),'rows':rows,'state_dependence_counterexample':fail},indent=2))
