"""c8: R9 (TD-II reader with pushes). y = rho u.x + k rho uA.p + xi, x~N(0,I_D) fresh, p known push.
Shared coupling (uA=u): regression of y on p gives k rho u (all components, incl. background); regression on the
in-frame read v.x gives rho (u.v); together they give k, rho, u and s. Independent actuator coupling (uA != u):
pushes give k rho uA, nothing about the background of u (blind spot of TD-II Thm blind stays)."""
import json
import numpy as np
rng = np.random.default_rng(42)
D, rho, s, k, T = 6, 1.0, 0.5, 0.7, 400000
u = rng.normal(size=D); u /= np.linalg.norm(u)
v = u + 0.6 * rng.normal(size=D); v /= np.linalg.norm(v)      # misaligned frame
out = {}
for case in ("shared", "independent"):
    uA = u if case == "shared" else (lambda z: z / np.linalg.norm(z))(rng.normal(size=D))
    x = rng.normal(size=(T, D)); p = rng.normal(size=(T, D)) * 0.3
    y = rho * x @ u + k * rho * p @ uA + s * rng.normal(size=T)
    a = (x @ v)                                               # the only read (budget b=1)
    # regress y on [a, p]
    X = np.column_stack([a, p]); coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    b_in, krho_uA = coef[0], coef[1:]
    resid_var = np.var(y - X @ coef)
    # recover under the shared-coupling hypothesis
    u_dir = krho_uA / np.linalg.norm(krho_uA)
    cos_vu = u_dir @ v
    # b_in = rho (u.v) + cross terms? x independent of p, so b_in estimates rho u.v
    rho_hat = b_in / cos_vu
    k_hat = np.linalg.norm(krho_uA) / rho_hat
    # residual variance after the in-frame read and the pushes = rho^2 (1-(u.v)^2) + s^2
    s2_hat = resid_var - rho_hat ** 2 * (1 - cos_vu ** 2)
    out[case] = dict(true=dict(rho=rho, k=k, s2=s ** 2, u_dot_v=float(u @ v)),
                     est=dict(rho=float(rho_hat), k=float(k_hat), s2=float(s2_hat), dir_err=float(np.linalg.norm(u_dir - np.sign(u_dir @ u) * u)),
                              uA_dot_u=float(uA @ u)))
    print(case, out[case])
json.dump(out, open("out/c8_push_frame.json", "w"), indent=1)
