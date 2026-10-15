"""Actual controlled-X instruments on the same source, including old reference.
No diamond-norm optimizer: the norm identity is proved in the manuscript.
These dense-state checks verify its physical map and attainability.
"""
import itertools
import json
import math
import numpy as np
from scipy.linalg import expm

I = np.eye(2)
X = np.array([[0, 1], [1, 0]], complex)
Y = np.array([[0, -1j], [1j, 0]], complex)
Z = np.diag([1., -1.])
P = [np.diag([1., 0.]), np.diag([0., 1.])]
V = np.kron(np.kron(P[0], I), I) + np.kron(np.kron(P[1], I), X)
PR = [np.kron(p, I) for p in P]
rng = np.random.default_rng(20261012)

def trace_norm(a):
    return np.linalg.svd(a, compute_uv=False).sum()

def collision(rho, alpha):
    eta = (I + alpha * Z) / 2
    joint = V @ np.kron(rho, eta) @ V.conj().T
    tensor = joint.reshape(4, 2, 4, 2)
    return [tensor[:, y, :, y] for y in (0, 1)]

def majority_error(alpha, m):
    eps = (1-alpha)/2
    return sum(math.comb(m, k)*eps**k*(1-eps)**(m-k) *
               (1 if 2*k > m else .5 if 2*k == m else 0)
               for k in range(m+1))

def actual(rho, alpha, m):
    branches = [((), rho)]
    for _ in range(m):
        branches = [(history+(y,), block)
                    for history, state in branches
                    for y, block in enumerate(collision(state, alpha))]
    out = [np.zeros_like(rho), np.zeros_like(rho)]
    for bits, block in branches:
        if 2*sum(bits) == m:
            for y in (0, 1):
                out[y] += .5*block
        else:
            out[int(2*sum(bits) > m)] += block
    return out

rows = []
for alpha in (0., .1, .6, 1.):
    for m in (1, 2, 3, 5):
        error = majority_error(alpha, m)
        residual = distance_residual = 0.
        for trial in range(12):
            if trial == 0:
                psi = np.array([1, 0, 0, 1], complex)/np.sqrt(2)
                rho = np.outer(psi, psi.conj())
            else:
                a = rng.normal(size=(4, 4)) + 1j*rng.normal(size=(4, 4))
                rho = a@a.conj().T
                rho /= np.trace(rho)
            blocks = [p@rho@p for p in PR]
            coarse = actual(rho, alpha, m)
            predicted = [(1-error)*blocks[y]+error*blocks[1-y] for y in (0, 1)]
            residual = max(residual, max(np.linalg.norm(a-b)
                            for a, b in zip(coarse, predicted)))
            halfnorm = sum(trace_norm(coarse[y]-blocks[y]) for y in (0, 1))/2
            distance_residual = max(distance_residual, abs(halfnorm-error))
        bound = math.exp(-m*alpha*alpha/2)
        assert residual < 1e-12 and distance_residual < 1e-12 and error <= bound + 1e-14
        rows.append(dict(alpha=alpha, m=m, error=error,
                         map_residual=residual, distance_residual=distance_residual,
                         hoeffding=bound))

# Actual bounded-function probe; positive identity shift is retained.
raw = rng.normal(size=(3, 3)) + 1j*rng.normal(size=(3, 3))
O = raw + raw.conj().T
O /= np.linalg.norm(O, 2)
vals, vecs = np.linalg.eigh(O)
couplings = []
for alpha in (0., .1, 1.):
    theta = (vecs*(.5*np.arccos(alpha*vals)))@vecs.conj().T
    H = np.kron(theta, Y)
    shifted = H + np.pi/2*np.eye(6)
    U = expm(-1j*shifted)
    K0 = U.reshape(3, 2, 3, 2)[:, 0, :, 0]
    effect = K0.conj().T@K0
    residual = np.linalg.norm(effect-(np.eye(3)+alpha*O)/2)
    assert residual < 1e-12 and np.linalg.eigvalsh(shifted)[0] > -1e-12
    couplings.append(dict(alpha=alpha, residual=residual,
                         min_positive_H=float(np.linalg.eigvalsh(shifted)[0])))

# Identical sharp effects, different post-read physical source.
flipped_distance = .5*trace_norm(X@P[0]@X-P[0])
assert abs(flipped_distance-1) < 1e-12

# Exact mean of a nonrelaxing source orbit.
a = .6
rho = (I+a*X)/2
cesaro = []
for T in (.1, 1., math.pi, 10., 100.):
    unitary = expm(-1j*T*Z)
    instantaneous = unitary@rho@unitary.conj().T
    mean = np.eye(2, dtype=complex)/2
    mean[0, 1] = (a/2)*(1-np.exp(-2j*T))/(2j*T)
    mean[1, 0] = mean[0, 1].conjugate()
    orbit_distance = .5*trace_norm(instantaneous-I/2)
    mean_distance = .5*trace_norm(mean-I/2)
    assert abs(orbit_distance-a/2) < 1e-12
    assert abs(mean_distance-a*abs(math.sin(T))/(2*T)) < 1e-12
    cesaro.append(dict(T=T, orbit_distance=orbit_distance,
                      mean_distance=mean_distance))

print(json.dumps(dict(instrument=rows, bounded_probe=couplings,
                      same_effect_output_distance=flipped_distance,
                      cesaro=cesaro), indent=2))
