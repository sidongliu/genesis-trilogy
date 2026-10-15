"""Hull characters in the SU(2)_1 model and their separation of stabilizers.

Run: python3 characters.py > characters.out   (deterministic, exact integer arithmetic where possible)

The SU(2)_1 vacuum module decomposes under SO(3) x Vir(c=1) as  sum_j V_j (x) L(1, j^2),
with ch L(1, j^2) = q^{j^2} (1 - q^{2j+1}) / phi(q).  For a closed subgroup K of SO(3) the
K-invariant subspace (the vacuum space of the hull M^K) has character
     chi_K(q) = sum_j m_j(K) (q^{j^2} - q^{(j+1)^2}) / phi(q),   m_j(K) = dim V_j^K.
"""
from fractions import Fraction
import math
import itertools

N = 40       # q-expansion order
JMAX = 60    # spins used for the injectivity test


def partitions(nmax):
    p = [0] * (nmax + 1)
    p[0] = 1
    for k in range(1, nmax + 1):
        for n in range(k, nmax + 1):
            p[n] += p[n - k]
    return p


P = partitions(N)


def char_from_m(m):
    """Coefficients of chi_K up to q^N from multiplicities m[0..]."""
    num = [0] * (N + 1)
    for j, mj in enumerate(m):
        if j * j <= N:
            num[j * j] += mj
        if (j + 1) ** 2 <= N:
            num[(j + 1) ** 2] -= mj
    # divide by phi(q): multiply by sum p(n) q^n
    return [sum(num[k] * P[n - k] for k in range(n + 1)) for n in range(N + 1)]


def chi_spin(j, theta):
    """Character of spin j at rotation angle theta (real, exact sum of cosines)."""
    return sum(math.cos(m * theta) for m in range(-j, j + 1))


def m_finite(angle_counts, j):
    order = sum(c for _, c in angle_counts)
    val = sum(c * chi_spin(j, th) for th, c in angle_counts) / order
    r = round(val)
    assert abs(val - r) < 1e-8, (angle_counts, j, val)
    return r


def cyclic(n):
    return [(2 * math.pi * k / n, 1) for k in range(n)]


def dihedral(n):
    return cyclic(n) + [(math.pi, n)]


T = [(0.0, 1), (2 * math.pi / 3, 8), (math.pi, 3)]
O = [(0.0, 1), (2 * math.pi / 3, 8), (math.pi / 2, 6), (math.pi, 3), (math.pi, 6)]
I = [(0.0, 1), (2 * math.pi / 3, 20), (2 * math.pi / 5, 12), (4 * math.pi / 5, 12), (math.pi, 15)]

groups = {}
for n in range(1, 41):
    groups[f"C{n}"] = [m_finite(cyclic(n), j) for j in range(JMAX + 1)]
for n in range(2, 41):
    groups[f"D{n}"] = [m_finite(dihedral(n), j) for j in range(JMAX + 1)]
groups["T"] = [m_finite(T, j) for j in range(JMAX + 1)]
groups["O"] = [m_finite(O, j) for j in range(JMAX + 1)]
groups["I"] = [m_finite(I, j) for j in range(JMAX + 1)]
groups["SO(2)"] = [1] * (JMAX + 1)
groups["O(2)"] = [1 if j % 2 == 0 else 0 for j in range(JMAX + 1)]
groups["SO(3)"] = [1] + [0] * JMAX

ok = True


def check(name, cond):
    global ok
    print(f"[{'PASS' if cond else 'FAIL'}] {name}")
    ok &= bool(cond)


# sanity: orders of finite groups from the angle data
check("orders T=12, O=24, I=60", sum(c for _, c in T) == 12 and sum(c for _, c in O) == 24 and sum(c for _, c in I) == 60)
# C1 = trivial group: m_j = 2j+1
check("m_j(C1) = 2j+1", all(groups["C1"][j] == 2 * j + 1 for j in range(JMAX + 1)))
# known Molien data: first invariant of T at j=3, O at j=4, I at j=6
first = lambda m: next(j for j in range(1, JMAX + 1) if m[j] > 0)
check("first nontrivial invariant: T at j=3, O at j=4, I at j=6",
      first(groups["T"]) == 3 and first(groups["O"]) == 4 and first(groups["I"]) == 6)

# characters
ch = {k: char_from_m(v) for k, v in groups.items()}
check("chi_SO(2) = 1/phi (U(1) current Fock space): p(n)", ch["SO(2)"] == P)
theta3_over_phi = [sum((2 if m else 1) * P[n - m * m] for m in range(0, int(math.isqrt(n)) + 1)) for n in range(N + 1)]
check("chi_{trivial} = theta_3/phi (SU(2)_1 vacuum character)", ch["C1"] == theta3_over_phi)
vir = [P[n] - (P[n - 1] if n >= 1 else 0) for n in range(N + 1)]
check("chi_SO(3) = (1-q)/phi (Vir_1 vacuum); levels 0..9 = 1,0,1,1,2,2,4,4,7,8 as in C-I",
      ch["SO(3)"] == vir and vir[:10] == [1, 0, 1, 1, 2, 2, 4, 4, 7, 8])
check("chi_O(2) at level 4 = 3 (even Fock space, C-I Prop. U1)", ch["O(2)"][4] == 3)
# even Fock space = (1/phi + phi(q)/phi(q^2)... ) / 2 : count partitions with an even number of parts
def even_parts(nmax):
    # generating function prod 1/(1 - t q^k), coefficient of even powers of t
    dp = [[0] * (nmax + 1) for _ in range(2)]
    dp[0][0] = 1
    for k in range(1, nmax + 1):
        for n in range(k, nmax + 1):
            for par in (0, 1):
                dp[par][n] += dp[1 - par][n - k]
    return dp[0]
check("chi_O(2) = partitions into an even number of parts", ch["O(2)"] == even_parts(N))
# C_n: charge lattice n*Z: chi = sum_{m in nZ} q^{m^2} / phi
for n in (2, 3, 5):
    lat = [sum(P[n0 - m * m] * (2 if m else 1) for m in range(0, int(math.isqrt(n0)) + 1) if m % n == 0) for n0 in range(N + 1)]
    check(f"chi_C{n} = theta_(nZ)/phi (rank-one lattice extension)", ch[f"C{n}"] == lat)

# injectivity: m-sequences pairwise distinct (equivalently characters distinct, since
# phi * chi = sum_j (m_j - m_{j-1}) q^{j^2} recovers every m_j)
keys = list(groups)
dups = [(a, b) for a, b in itertools.combinations(keys, 2) if groups[a] == groups[b]]
check(f"multiplicity sequences (j <= {JMAX}) pairwise distinct over {len(keys)} conjugacy classes", not dups)
if dups:
    print("  duplicates:", dups)
# the first level at which characters of two classes differ (shows how early the geometry separates them)
def first_diff(a, b):
    for n in range(N + 1):
        if ch[a][n] != ch[b][n]:
            return n
    return None
for a, b in [("SO(2)", "O(2)"), ("O(2)", "SO(3)"), ("C2", "D2"), ("D6", "T"), ("T", "O"), ("O", "I"), ("I", "SO(3)"), ("C12", "D6")]:
    print(f"  first level where chi_{a} and chi_{b} differ: {first_diff(a, b)}")

# angle-distribution invariants used in the proof
def pi_mass(angle_counts):
    order = sum(c for _, c in angle_counts)
    return Fraction(sum(c for th, c in angle_counts if abs(math.cos(th) + 1) < 1e-12), order)
print("  mass at angle pi:  T", pi_mass(T), " O", pi_mass(O), " I", pi_mass(I),
      " D6", pi_mass(dihedral(6)), " C12", pi_mass(cyclic(12)))

print()
print("hull characters, levels 0..16")
for k in ["SO(3)", "O(2)", "SO(2)", "C2", "C3", "D2", "T", "O", "I", "C1"]:
    print(f"  {k:6s}", ch[k][:17])
print()
print("z-axis dephasing noise: closed subgroups containing SO(2)_z are SO(2), O(2), SO(3);")
print("  their hulls have characters", ch["SO(2)"][:9], ch["O(2)"][:9], ch["SO(3)"][:9])
print()
print("ALL PASS" if ok else "SOME CHECK FAILED")
