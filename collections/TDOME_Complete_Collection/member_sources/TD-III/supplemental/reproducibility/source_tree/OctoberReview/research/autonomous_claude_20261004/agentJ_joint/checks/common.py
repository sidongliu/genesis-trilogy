"""Shared formulas for the joint model checks (agent J). Entropies in nats."""
import numpy as np

LN2 = np.log(2.0)

def h(x):
    x = np.clip(np.asarray(x, dtype=float), 1e-300, 1 - 1e-16)
    return -(x * np.log(x) + (1 - x) * np.log(1 - x))

def Hp(p, r):
    """Entropy of p|e0><e0|+(1-p)|e1><e1| with overlap r (HAFF-E)."""
    return h((1 + np.sqrt(1 - 4 * p * (1 - p) * (1 - r ** 2))) / 2)

def helstrom_gain(r):
    """Gain 1-2*eps of the Helstrom measurement at p=1/2: sqrt(1-r^2)."""
    return np.sqrt(1 - r ** 2)

def info_XM(g):
    """I(X:M) for a symmetric binary channel of gain g, uniform X."""
    return LN2 - h((1 + g) / 2)

def chi_M_G(g, rG):
    """chi(M:E_G): M read from F with gain g, G pure-state fragment of overlap rG."""
    return Hp(0.5, rG) - Hp((1 + g) / 2, rG)

def eps_coherent(N):
    """QR-B coherent half-turn drive failure probability."""
    return (1 - np.exp(-np.pi ** 2 / (8 * N))) / 2

def eps_optimal_bound(N):
    """Smallest eps compatible with QR-B Thm energy (N >= theta(1-2e)/(4 sqrt(e(1-e))) - 1/2), by bisection."""
    def rhs(e):
        th = np.arccos(2 * e - 1)
        return th * (1 - 2 * e) / (4 * np.sqrt(e * (1 - e))) - 0.5
    lo, hi = 1e-14, 0.4999999
    if rhs(hi) >= N:
        return hi
    for _ in range(200):
        mid = np.sqrt(lo * hi)
        if rhs(mid) > N:
            lo = mid
        else:
            hi = mid
    return hi

def eps_gaussian(N):
    """Failure of the best displaced squeezed Gaussian drive with occupation N (QR-B Prop gauss, invert min N formula)."""
    # min N = (sqrt(1+pi^2/(4 l)) - 1)/2 , l = -1/2 log(1-2 eps)
    a = (2 * N + 1) ** 2 - 1  # = pi^2/(4 l)
    l = np.pi ** 2 / (4 * a)
    return (1 - np.exp(-2 * l)) / 2
