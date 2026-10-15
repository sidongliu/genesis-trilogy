"""Finite-dimensional bookkeeping check of the compatibility witness for Foundation II (round 21).
The half-line Virasoro algebra B (type III_1) is replaced by a finite stand-in M_2 with a faithful NON-tracial state, so that modular flows are not trivial; K, P1, P2 are M_2 as in the witness.
A_U = B (x) K (x) P1 (x) P2, A_1 = B (x) K (x) P1 (x) 1, A_2 = B (x) K (x) 1 (x) P2, Phi = B (x) 1 (x) 1 (x) 1.
Reference state w0 = rho_B (x) tau (x) tau (x) tau; sources  varpi_1 = rho_B (x) rho_+ (x) eta_1 (x) tau,  varpi_2 = rho_B (x) rho_- (x) tau (x) eta_2,  rho_pm = (1 pm a Z)/2, eta_i faithful.
Checks: (1) restrictions of varpi_1, varpi_2, w0 to Phi coincide; (2) they differ on the overlap B (x) K by delta(Z_K) = 2a; (3) E_i := id (x) (varpi_i-restricted state on the removed factors) is a unital completely positive
conditional expectation of A_i onto Phi with varpi_i o E_i = varpi_i; (4) the modular flow of varpi_i on A_i restricts on Phi to the modular flow of the shared state psi, and psi = w0|Phi so it is also the flow of the reference state.
It checks the bookkeeping of the witness, not the type III_1 statement (that needs B = Vir_1(0,inf) and the Bisognano-Wichmann/GLW facts)."""
import numpy as np
from scipy.linalg import fractional_matrix_power as fmp, expm
rng = np.random.default_rng(7)
def kron(*a):
    out = np.array([[1.0 + 0j]])
    for x in a: out = np.kron(out, x)
    return out
I2 = np.eye(2); Z = np.diag([1., -1.]); tau = I2/2
a = 0.6
rho_p = (I2 + a*Z)/2; rho_m = (I2 - a*Z)/2
rho_B = np.diag([.7, .3]).astype(complex)
th = 0.4; R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
eta1 = R @ np.diag([.8, .2]) @ R.T; eta2 = R.T @ np.diag([.6, .4]) @ R
# density matrices on the full space B K P1 P2 (ordering B,K,P1,P2)
D0 = kron(rho_B, tau, tau, tau); D1 = kron(rho_B, rho_p, eta1, tau); D2 = kron(rho_B, rho_m, tau, eta2)
def rand_obs(dim):
    X = rng.normal(size=(dim, dim)) + 1j*rng.normal(size=(dim, dim)); return X
# (1) restrictions to Phi = B x 1 x 1 x 1
xB = rand_obs(2); xPhi = kron(xB, I2, I2, I2)
v0, v1, v2 = [np.trace(D @ xPhi) for D in (D0, D1, D2)]
assert abs(v0 - v1) < 1e-13 and abs(v0 - v2) < 1e-13
# (2) overlap B x K
dZ = np.trace(D1 @ kron(I2, Z, I2, I2)) - np.trace(D2 @ kron(I2, Z, I2, I2)); assert abs(dZ - 2*a) < 1e-13
# (3) conditional expectation E1 : A1 -> Phi, E1(x) = (id_B (x) varpi_1|_{K,P1})(x) (x) 1 ; check on a random element of A1 = B K P1 (x) 1
def E1(Aop):   # Aop acts on B K P1 (8x8); returns element of B (2x2)
    out = np.zeros((2, 2), complex); rK = rho_p; rP = eta1
    T = Aop.reshape(2, 4, 2, 4)                      # (B,KP1,B',KP1')
    state = np.kron(rK, rP)                            # density on K P1
    for i in range(2):
        for j in range(2): out[i, j] = np.trace(T[i, :, j, :] @ state)
    return out
A1el = rand_obs(8)                                      # element of B K P1
full = kron(A1el, I2)                                   # embedded in A_U (P2 factor identity)
v1_full = np.trace(D1 @ full)
v1_E = np.trace(D1 @ kron(E1(A1el), I2, I2, I2))
assert abs(v1_full - v1_E) < 1e-12                      # varpi_1 o E_1 = varpi_1
assert np.allclose(E1(kron(xB, I2, I2)), xB*1.0)        # E_1 is the identity on Phi
# complete positivity: E1 is a partial 'state-slice', which is CP; check positivity on a random positive element
P = rand_obs(8); P = P @ P.conj().T
assert np.linalg.eigvalsh((E1(P) + E1(P).conj().T)/2).min() > -1e-12
# (4) modular flows: sigma_t^{phi}(x) = D^{it} x D^{-it}, restricted to Phi
def flow(D, t): 
    w, V = np.linalg.eigh((D + D.conj().T)/2); return V @ np.diag(w.astype(complex)**(1j*t)) @ V.conj().T
t = 0.83
psi_flow = flow(rho_B, t) @ xB @ flow(rho_B, t).conj().T
for D, name in ((D1, 'varpi_1'), (D2, 'varpi_2'), (D0, 'w0')):
    U = flow(D, t); img = U @ xPhi @ U.conj().T
    assert np.allclose(img, kron(psi_flow, I2, I2, I2), atol=1e-12), name
print("witness bookkeeping OK: restrictions agree on Phi; delta12(Z_K) =", float(dZ.real), "; E_1 preserves varpi_1; modular flows of varpi_1, varpi_2, w0 restrict to the same flow on Phi (t=%.2f)" % t)
