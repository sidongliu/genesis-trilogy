"""c4: Proposition R4 (price of a standard against sign inversions).

(1) Exact oracle lower bound min_G (w + V F(G))/G, F(G)=sum_{j<G} 1/2(1-(1-2eta)^j), against sqrt(2 w V eta).
(2) Explicit policy in the body loop (g=0.9, d=0.05, s=0.95): every G uses, n_r reference reads of a known +1 input,
    majority vote sets the sign estimate; corrective flips use the interpreted reading. Cost per use = V*loss + w*n_r/G.
    Excess over the eta=0 loop compared with sqrt(2 w V (1-2 L0) eta).
"""
import json
import numpy as np

def oracle(w, V, eta, Gmax=10 ** 6):
    G = np.arange(1, Gmax + 1, dtype=float)
    f = 0.5 * (1 - (1 - 2 * eta) ** (G - 1))
    F = np.cumsum(f)
    c = (w + V * F) / G
    i = int(np.argmin(c))
    return float(c[i]), int(G[i])

res = {"oracle": []}
for w in (0.01, 0.1, 1.0):
    for eta in (1e-5, 1e-4, 1e-3, 1e-2):
        lb, Gs = oracle(w, 1.0, eta)
        res["oracle"].append(dict(w=w, eta=eta, lower_bound=lb, G_opt=Gs, sqrt2weta=np.sqrt(2 * w * eta), ratio=lb / np.sqrt(2 * w * eta)))
        print(f"w={w} eta={eta}: LB={lb:.5g} G*={Gs} sqrt(2w eta)={np.sqrt(2*w*eta):.5g} ratio={lb/np.sqrt(2*w*eta):.4f}")

rng = np.random.default_rng(11)
g, d, s = 0.9, 0.05, 0.95
def sim(eta, G, nr, T=20000, ntraj=2000):
    sigma = np.ones(ntraj); B = np.ones(ntraj); sh = np.ones(ntraj)
    loss = 0.0; cnt = 0
    for t in range(T):
        sigma *= np.where(rng.random(ntraj) < eta, -1, 1)
        B *= np.where(rng.random(ntraj) < d, -1, 1)
        if G > 0 and t % G == 0:
            z = np.where(rng.random((nr, ntraj)) < (1 + sigma * g) / 2, 1, -1).sum(0)
            sh = np.where(z > 0, 1, np.where(z < 0, -1, sh))
        y = np.where(rng.random(ntraj) < (1 + sigma * g * B) / 2, 1, -1)
        fl = (sh * y == -1) & (rng.random(ntraj) < s)
        B = np.where(fl, -B, B)
        if t >= 200:
            loss += np.mean(B == -1); cnt += 1
    return loss / cnt

L0 = sim(0.0, 0, 1)
res["L0"] = L0
print("L0 (eta=0, no refs):", L0)
Gs = [5, 10, 20, 40, 80, 160, 320, 640, 1280]
nrs = [1, 3, 5]
table = {}
for eta in (1e-4, 3e-4, 1e-3, 3e-3):
    table[eta] = {}
    for G in Gs:
        for nr in nrs:
            table[eta][(G, nr)] = sim(eta, G, nr)
pol = []
for w in (0.01, 0.1):
    for eta in (1e-4, 3e-4, 1e-3, 3e-3):
        best = min(((table[eta][(G, nr)] + w * nr / G), G, nr) for G in Gs for nr in nrs)
        excess = best[0] - L0
        pred = np.sqrt(2 * w * (1 - 2 * L0) * eta)
        lb, _ = oracle(w, 1.0, eta)
        pol.append(dict(w=w, eta=eta, best_cost=best[0], G=best[1], n_r=best[2], excess=excess, pred_sqrt=pred,
                        ratio_excess_to_pred=excess / pred, oracle_LB_total=lb))
        print(f"w={w} eta={eta}: best cost {best[0]:.5f} (G={best[1]}, n_r={best[2]}), excess {excess:.5f}, sqrt pred {pred:.5f}, ratio {excess/pred:.3f}, oracle LB {lb:.5f}")
# fitted slope of excess vs eta at each w
for w in (0.01, 0.1):
    xs = np.log([p["eta"] for p in pol if p["w"] == w]); ys = np.log([p["excess"] for p in pol if p["w"] == w])
    res[f"slope_w{w}"] = float(np.polyfit(xs, ys, 1)[0])
    print("w", w, "log-log slope of excess vs eta:", res[f"slope_w{w}"])
res["policy"] = pol
json.dump(res, open("out/c4_jump_price.json", "w"), indent=1, default=float)
