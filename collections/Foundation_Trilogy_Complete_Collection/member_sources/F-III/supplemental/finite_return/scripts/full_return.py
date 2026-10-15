"""Independent sparse reassembly of the six-qubit retained scalar/JT action.

The new final gate is C_X(P,E0). Every purification column is kept.
Actual runs evolve the mass equation as well as independent work integrals.
Replays hold y fixed and selectively replace Phi; they are causal diagnostics.
"""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import argparse, json, math
from pathlib import Path
import numpy as np
from scipy.sparse import csr_matrix, diags, eye, kron
from scipy.integrate import solve_ivp

ROOT=Path(__file__).resolve().parents[1]
I2=np.eye(2,dtype=complex)
X=np.array([[0,1],[1,0]],complex)
Y=np.array([[0,-1j],[1j,0]],complex)
Z=np.diag([1.,-1.])
N=np.diag([0.,1.])
J=.8; GAP=.05; CHI=.01; LAM=.05; OFFSET=.04
KAP=.01; GAM=1/(2*math.pi)
PHISTAR=3+30/(1.95*2.05)/100
CLOCK=np.arange(-30,71)

def local(a,j):
    r=np.ones((1,1),complex)
    for k in range(6): r=np.kron(r,a if j==k else I2)
    return r

def assemble(sign,policy,read='X'):
    n=[np.diag(local(N,j)).real for j in range(6)]
    zp=np.diag(local(Z,2)).real
    charge=n[0]+sign*n[1]
    py=local((I2+Y)/2,2)
    pq=local((I2+(X if read=='X' else Y))/2,2)
    acquisition=np.eye(64)-py+py@local(X,3)
    frame=np.diag(n[3])+(np.eye(64)-np.diag(n[3]))@local(X,1)
    frame={'learned':frame,'odd':np.eye(64),'even':local(X,1)}[policy]
    gates={0:np.diag(np.exp(-.05j*charge*(1-2*n[4]))),
           2:np.diag(np.exp(-.5j*sign*math.asin(.8)*zp)),
           4:acquisition,6:frame,
           8:np.diag(np.exp(-.25j*math.pi*charge*(1-2*n[5]))),
           10:frame,12:np.eye(64)-pq+pq@local(X,4)}
    adj=diags([np.ones(100),np.ones(100)],[-1,1],format='csr')
    base=kron(eye(101),diags(2*J+OFFSET+GAP*(n[4]+n[5])),format='csr')-J*kron(adj,eye(64),format='csr')
    for j,g in gates.items():
        edge=csr_matrix(([1.],([j+31],[j+30])),shape=(101,101))
        base-=J*kron(edge,csr_matrix(g-np.eye(64)),format='csr')
        base-=J*kron(edge.T,csr_matrix(g.conj().T-np.eye(64)),format='csr')
    data=np.zeros((64,8),complex)
    for ref in range(8):
        old,e0,e1=(ref>>2)&1,(ref>>1)&1,ref&1
        b1,b2=(0,1) if old==0 else (1,0)
        for p in (0,1):
            data[(b1<<5)|(b2<<4)|(p<<3)|(e0<<1)|e1,ref]=math.sqrt((.1 if e0 else .9)*(.1 if e1 else .9))/2
    a=math.pi/1.2; xx=CLOCK+10
    packet=np.exp(.5j*math.pi*CLOCK)*(np.sinc(.6*(xx+a)/math.pi)+np.sinc(.6*(xx-a)/math.pi))
    packet[abs(xx)>7]=0; packet/=np.linalg.norm(packet)
    psi=(packet[:,None,None]*data).reshape(6464,8)
    return dict(base=base,psi=psi,z=np.tile(zp,101),a=np.tile(charge**2,101),r=np.tile(n[3],101),e=np.tile(n[4],101),e1=np.tile(n[5],101),packet=packet,gates=gates,data=data)

def score(psi):
    v=psi.reshape(101,4,16,2,4)
    return float(1-np.sum(abs(v[:,1,:,0,:]+v[:,2,:,1,:])**2)/2)

def run(sign,policy,alpha=1.,lam=LAM,chi=CHI,method='DOP853',rtol=2e-10,read='X',cm0=4.7975):
    sys=assemble(sign,policy,read)
    def core(psi,y,phi):
        spring=chi*(sys['a']*(y+.5)**2+(alpha*sys['r']+sys['e'])*(y-.5)**2)/2
        hp=sys['base']@psi+(spring+lam*(phi-3)*sys['z'])[:,None]*psi
        pr=np.sum(abs(psi)**2,axis=1)
        fy=chi*np.dot(pr,sys['a']*(y+.5)+(alpha*sys['r']+sys['e'])*(y-.5))
        fp=lam*np.dot(pr,sys['z'])
        return hp,float(np.vdot(psi,hp).real),fy,fp,pr
    ec0=core(sys['psi'],.5,3)[1]
    rest=10-.95-50*(3-PHISTAR)**2-ec0
    bm=math.sqrt(9-cm0); bp=bm-.1
    c0=[.5,math.sqrt((.95-.1375)/10),0,0,3,0,9-bp*bp,cm0,10,0,0,0,0]
    init=np.r_[np.array(c0,complex),sys['psi'].ravel()]
    def rhs(t,v):
        y,vy,th,vt,phi,w,cp,cm,m=v[:9].real
        ps=v[13:].reshape(6464,8)
        hp,ec,fy,fp,_=core(ps,y,phi)
        bp=math.sqrt(w*w+phi*phi-cp); bm=math.sqrt(w*w+phi*phi-cm)
        power=GAM*vt*vt; up=100*(phi-PHISTAR)
        dc=[vy,(-(y-th)-.1*y-fy)/20,vt,(-(th-y)-GAM*vt)/10,
            w,((bp-bm)*power/2-(up+fp)*bp*bm)/m-phi,
            -KAP*power*(bp+w),KAP*power*(bm-w),(up+fp)*w-power,
            power,fy*vy,fp*w,up*w]
        return np.r_[np.array(dc,complex),(-1j*hp).ravel()]
    times=np.linspace(0,30,301)
    sol=solve_ivp(rhs,(0,30),init,method=method,rtol=rtol,atol=rtol/50,max_step=.15,dense_output=True,t_eval=times)
    if not sol.success: raise RuntimeError(sol.message)
    errors=[]; rows=[]
    for j,t in enumerate(times):
        y,vy,th,vt,phi,w,cp,cm,m,q,wy,wp,wu=sol.y[:13,j].real
        ps=sol.y[13:,j].reshape(6464,8)
        hp,ec,fy,fp,pr=core(ps,y,phi)
        em=5*vt*vt+10*vy*vy+.5*(th-y)**2+.05*y*y
        u=50*(phi-PHISTAR)**2
        errors.append([sum(pr)-1,em+q+wy-.95,ec-ec0-wy-wp,u-50*(3-PHISTAR)**2-wu,
            m-rest-u-em-ec,math.sqrt(w*w+phi*phi-cm)-math.sqrt(w*w+phi*phi-cp)-KAP*m])
        rows.append([t,y,phi,GAM*vt*vt,np.dot(pr,sys['r']),np.dot(pr,sys['e']),score(ps),np.sum(abs(ps.reshape(101,64,8)[CLOCK>=13])**2),m,ec])
    rows=np.array(rows)
    proto=sys['data'].copy()
    for _,g in sorted(sys['gates'].items()): proto=g@proto
    protoloss=score((sys['packet'][:,None,None]*proto).reshape(6464,8))
    report=dict(sign=sign,policy=policy,alpha=alpha,lambda_sensor=lam,chi=chi,read=read,method=method,rtol=rtol,cm0=cm0,
        initial_core=ec0,rest_mass=rest,final_loss=rows[-1,6],prototype_loss=protoloss,completion=rows[-1,7],nE0=rows[-1,5],
        initial_mass=10.,final_mass=rows[-1,8],initial_cell_energy=.01,cell0_gain=GAP*(rows[-1,5]-.1),
        max_ledger_errors=np.max(abs(np.array(errors)),axis=0).tolist(),min_mass=min(rows[:,8]),
        min_offset=float(min(OFFSET-abs(lam*(rows[:,2]-3)))),final_core=rows[-1,9],
        initial_support=50*(3-PHISTAR)**2,exported_scalar_energy=float(sol.y[9,-1].real))
    return report,rows,sol,sys

def replay(sys,trajectory,alpha,baseline=None,sites='all',method='DOP853',rtol=2e-10,lam=LAM,chi=CHI):
    mask=np.repeat(CLOCK<=4 if sites=='pre' else CLOCK>=5 if sites=='post' else np.zeros(101,bool) if sites=='none' else np.ones(101,bool),64)
    def rhs(t,v):
        y,phi=trajectory.sol(t)[[0,4]].real
        if baseline is not None:
            old=baseline.sol(t)[4].real
            ph=np.where(mask,phi,old)
        else: ph=phi
        ps=v.reshape(6464,8)
        diag=chi*(sys['a']*(y+.5)**2+(alpha*sys['r']+sys['e'])*(y-.5)**2)/2+lam*(ph-3)*sys['z']
        return (-1j*(sys['base']@ps+diag[:,None]*ps)).ravel()
    sol=solve_ivp(rhs,(0,30),sys['psi'].ravel(),method=method,rtol=rtol,atol=rtol/50,max_step=.15)
    ps=sol.y[:,-1].reshape(6464,8); pr=np.sum(abs(ps)**2,axis=1)
    return float(np.dot(pr,sys['e']))

def main():
    p=argparse.ArgumentParser(); p.add_argument('mode',choices=['base','edge','audit','tight','null','background']); a=p.parse_args()
    out={}
    if a.mode=='base':
        for pol in ('learned','odd','even'):
            for sign in (-1,1):
                rep,rows,_,_=run(sign,pol); out[f'{pol}_{sign}']=rep
                np.save(ROOT/'results'/f'{pol}_{sign}.npy',rows)
                print(json.dumps(rep),flush=True)
        out['mean_losses']={pol:sum(out[f'{pol}_{s}']['final_loss'] for s in (-1,1))/2 for pol in ('learned','odd','even')}
    elif a.mode=='background':
        for cm in (4.6975,4.8975):
            rep,rows,_,_=run(-1,'learned',cm0=cm)
            out[str(cm)]=rep; np.save(ROOT/'results'/f'background_{cm}.npy',rows)
            print(json.dumps(rep),flush=True)
    else:
        for sign in (-1,1):
            opts=dict(method='RK45',rtol=3e-12) if a.mode=='audit' else {}
            if a.mode=='tight': opts=dict(method='DOP853',rtol=2e-12)
            if a.mode=='null': opts['lam']=0.
            r0,_,s0,sys=run(sign,'learned',alpha=0,**opts)
            r1,rows,s1,sys=run(sign,'learned',alpha=1,**opts)
            lam=opts.get('lam',LAM)
            common=dict(method=opts.get('method','DOP853'),rtol=opts.get('rtol',2e-10),lam=lam)
            all1=replay(sys,s1,1,**common)
            # all mask returns actual phi; use zero mask for Phi^(0).
            old=replay(sys,s1,1,s0,sites='none',**common)
            post=replay(sys,s1,1,s0,sites='post',**common)
            pre=replay(sys,s1,1,s0,sites='pre',**common)
            out[str(sign)]=dict(alpha0=r0,alpha1=r1,actual_replay=all1,baseline_replay=old,
                geometry_edge=all1-old,post_edge=post-old,pre_edge=pre-old,nonlinear_cross=all1-post-pre+old,
                replay_vs_actual=all1-r1['nE0'],total_port_contrast=r1['nE0']-r0['nE0'])
            print(json.dumps(out[str(sign)]),flush=True)
    (ROOT/'results'/f'{a.mode}.json').write_text(json.dumps(out,indent=2)+'\n')

if __name__=='__main__': main()
