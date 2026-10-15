"""Independent periodic relative-coordinate two-walker check (root, informed audit).

The two distinguishable coined walkers first coin, then shift, then receive
exp(+i phi) at coincident sites. r=(xA-xB)/2 and R=(xA+xB)/2 on the even
relative lattice; K is conjugate to R. Periodic truncations preserve unitarity.
No audit implementation is imported.
"""
from pathlib import Path
import json
import numpy as np
from scipy.linalg import eig

OUT=Path(__file__).resolve().parent

def unitary(K,half,phi=.4):
    n=2*half+1
    coin=lambda a: np.array([[np.cos(a),-np.sin(a)],[np.sin(a),np.cos(a)]])
    C=np.kron(coin(.3),coin(.5))
    U=np.zeros((4*n,4*n),complex)
    for m in range(n):
        for out,(ha,hb) in enumerate(((1,1),(1,-1),(-1,1),(-1,-1))):
            dest=(m+(ha-hb)//2)%n
            U[4*dest+out,4*m:4*m+4]=np.exp(-1j*K*(ha+hb)/2)*C[out]
    U[4*half:4*half+4]*=np.exp(1j*phi)
    return U

def localized(K,half=40,near=None):
    U=unitary(K,half)
    vals,vec=eig(U,check_finite=False)
    ph=-np.angle(vals)
    prob=np.abs(vec.reshape(2*half+1,4,-1))**2
    contact=prob[half].sum(axis=0)
    radius=(prob.sum(axis=1)*np.abs(np.arange(-half,half+1))[:,None]).sum(axis=0)
    # At K=0 the selected positive isolated band is below the ++ threshold .8.
    # Away from zero continue by overlap, not by a momentum cutoff.
    if near is None:
        candidates=np.flatnonzero((ph>.5)&(ph<.8)&(contact>.05))
        i=candidates[np.argmax(contact[candidates])]
    else:
        i=np.argmax(abs(near.conj()@vec)**2)
    row=dict(K=float(K),omega=float(ph[i]),contact=float(contact[i]),
             mean_abs_r=float(radius[i]),
             eigen_residual=float(np.linalg.norm(U@vec[:,i]-vals[i]*vec[:,i])),
             modulus_error=float(abs(abs(vals[i])-1)))
    return row,vec[:,i]

def run():
    results={'convention':__doc__,'finite_size':[],'band':[]}
    for half in (30,45,65):
        row,_=localized(0,half)
        h=.005
        pos,_=localized(h,half)
        neg,_=localized(-h,half)
        curv=(pos['omega']+neg['omega']-2*row['omega'])/h**2
        row.update(half=half,curvature=curv,inertial_mass=1/curv,readout_c2=row['omega']*curv)
        results['finite_size'].append(row)
    old=None
    for K in np.linspace(0,2,41):
        row,old=localized(K,45,old)
        results['band'].append(row)
    results['max_grid_speed_to_K2']=float(max(abs(np.gradient(
        [x['omega'] for x in results['band']],.05))))
    (OUT/'ROOT_QR_NUMERICS.json').write_text(json.dumps(results,indent=2)+'\n')
    print(json.dumps({'finite_size':results['finite_size'],
        'selected_band_rows':[results['band'][i] for i in (0,14,20,35,40)],
        'max_grid_speed_to_K2':results['max_grid_speed_to_K2']},indent=2))

if __name__=='__main__': run()
