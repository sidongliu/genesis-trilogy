#!/usr/bin/env python3
"""Generate fig_loop.pdf for T-DOME Paper III.

Three-panel figure:
  (a) Self-referential Fisher information I_F(t) vs detection threshold
  (b) Lyapunov function V(t): calibrated agent vs uncalibrated agent
  (c) Cumulative thermodynamic cost decomposition

Parameters match Section 7 of Paper III and extend Paper II's qubit example:
  omega_0 = 1, lambda_z = 1, gamma_z = 0.5  (dephasing, non-Markov)
  lambda_x = 0.3, gamma_x = 5.0             (dissipative, ~Markov)
  k_star = 2 (foreground channels selected by ego)
  Drift: exponential frame drift theta(t) = theta_0 * exp(Lambda * t)
  Loop: eta = 0.5 (natural gradient learning rate)
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---- Physical parameters ----
lam_z, gam_z = 1.0, 0.5     # dephasing channel
lam_x, gam_x = 0.3, 5.0     # dissipative channel
k_star = 2                    # foreground channels
h_mu = 1.0                    # normalised spectral density
C_meta = 1.0                  # meta-observer budget (units of h_mu)
kBT_ln2 = 1.0                 # k_B T ln2, normalised

# Drift parameters (exponential, matching Paper II's Delusion Trap)
theta_0 = 0.02               # initial misalignment
Lambda = 0.08                 # drift rate
eta = 0.5                     # natural gradient learning rate
alpha_pe = 0.3                # persistent excitation constant

# Delusion trap time: t_del = (1/Lambda) * ln(pi/(4*theta_0))
t_del = (1.0 / Lambda) * np.log(np.pi / (4 * theta_0))
print(f"Delusion trap time: t_del = {t_del:.1f}")

# Time grid
dt = 0.02
t_max = 60.0
t = np.arange(0, t_max, dt)
N = len(t)

# ---- Optimal frame drift (environment changes) ----
# sigma*(t) drifts continuously; uncalibrated agent stays at sigma(0) = 0
sigma_star = theta_0 * np.exp(Lambda * t)  # optimal frame drifts exponentially

# ---- Fisher information parameters ----
I_F_env = lam_z / gam_z  # = 2.0 (per-component)
c_coupling = 0.8  # coupling constant

# ---- (b) Lyapunov function V(t) ----
# Uncalibrated agent: frame stays fixed, V = (sigma* - 0)^2
V_uncal = sigma_star**2

# Calibrated agent: simulate natural gradient loop
sigma_cal = np.zeros(N)
sigma_cal[0] = 0.0  # starts at same point as uncalibrated

for i in range(1, N):
    # Mismatch between current frame and optimal
    delta = sigma_cal[i-1] - sigma_star[i-1]

    # Self-referential Fisher information (data signal)
    I_F_local = c_coupling * delta**2 * I_F_env

    # Natural gradient update: sigma_dot = -eta * g^{-1} * grad L
    # grad L ~ delta * I_F_local (gradient of frame loss)
    # But g^{-1} ~ 1/I_F_local, so update simplifies to:
    # sigma_dot = -eta * delta (reparametrization-invariant!)
    # Plus noise floor for numerical stability
    sigma_cal[i] = sigma_cal[i-1] - eta * delta * dt

V_cal = (sigma_cal - sigma_star)**2

# ---- (a) Self-referential Fisher information ----
# For uncalibrated agent: I_F grows as mismatch^2
I_F_uncal = c_coupling * V_uncal * I_F_env

# Detection threshold (noise floor)
I_F_threshold = c_coupling * (0.03)**2 * I_F_env  # corresponds to ~0.03 rad mismatch
I_F_thresh_arr = I_F_threshold * np.ones_like(t)

# For calibrated agent (for reference, not plotted):
I_F_cal = c_coupling * V_cal * I_F_env

# ---- (c) Cumulative thermodynamic cost ----
# Sensing: W_sense = kBT*ln2 * h_mu * k_star per unit time
W_sense_rate = kBT_ln2 * h_mu * k_star  # = 2.0
W_sense_cum = W_sense_rate * t

# Computing: W_compute = kBT*ln2 * C_meta per unit time
W_compute_rate = kBT_ln2 * C_meta  # = 1.0
W_compute_cum = W_compute_rate * t

# Actuating: W_actuate = zeta * |d sigma/dt|^2 * dt (thermodynamic length cost)
# For calibrated agent, actuation cost comes from frame rotation
zeta_friction = lam_z  # friction tensor ~ coupling
dsigma_dt = np.gradient(sigma_cal, dt)
W_actuate_rate = zeta_friction * dsigma_dt**2
W_actuate_cum = np.cumsum(W_actuate_rate) * dt

W_total_cum = W_sense_cum + W_compute_cum + W_actuate_cum

# ---- Plotting ----
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4.5))

# Panel (a): Fisher information
ax1.semilogy(t, I_F_uncal, color="steelblue", linewidth=1.5,
             label=r"$\mathcal{I}_F(t)$")
ax1.axhline(I_F_threshold, color="red", linestyle="--", linewidth=1.2,
            label="detection threshold")
ax1.axvline(t_del, color="gray", linestyle=":", alpha=0.6, linewidth=1,
            label=r"$t_{\mathrm{del}}$")
# Mark detection time
detect_idx = np.argmax(I_F_uncal > I_F_threshold)
if detect_idx > 0:
    t_detect = t[detect_idx]
    ax1.axvline(t_detect, color="green", linestyle="-.", alpha=0.6,
                linewidth=1, label=r"$\Delta t_{\mathrm{detect}}$")
    print(f"Detection time: t_detect = {t_detect:.1f}")

ax1.set_xlabel(r"$t\;[\omega_0^{-1}]$", fontsize=12)
ax1.set_ylabel(r"$\mathcal{I}_F(\sigma;\{e_t\})$", fontsize=12)
ax1.set_title("(a) Self-referential Fisher information", fontsize=11)
ax1.set_xlim(0, t_max)
ax1.set_ylim(1e-5, 10)
ax1.legend(loc="lower right", fontsize=8.5)

# Panel (b): Lyapunov function
ax2.semilogy(t, np.maximum(V_cal, 1e-8), color="steelblue", linewidth=1.5,
             label="calibrated (with loop)")
ax2.semilogy(t, V_uncal, color="red", linewidth=1.5,
             label="uncalibrated (no loop)")
ax2.axhline((np.pi/4)**2, color="red", linestyle=":", alpha=0.4,
            linewidth=1)
ax2.axvline(t_del, color="gray", linestyle=":", alpha=0.6, linewidth=1,
            label=r"$t_{\mathrm{del}}$")
ax2.text(t_del + 1, 0.5, r"$t_{\mathrm{del}}$", fontsize=10, color="gray")
ax2.set_xlabel(r"$t\;[\omega_0^{-1}]$", fontsize=12)
ax2.set_ylabel(r"$V(\sigma) = d_{\mathrm{geo}}(\sigma, \sigma^*)^2$",
               fontsize=12)
ax2.set_title("(b) Lyapunov function", fontsize=11)
ax2.set_xlim(0, t_max)
ax2.set_ylim(1e-8, 10)
ax2.legend(loc="upper left", fontsize=9)

# Panel (c): Cumulative cost decomposition (stacked area)
ax3.fill_between(t, 0, W_sense_cum, alpha=0.35, color="green",
                 label="sensing")
ax3.fill_between(t, W_sense_cum, W_sense_cum + W_compute_cum,
                 alpha=0.35, color="darkorange", label="computing")
ax3.fill_between(t, W_sense_cum + W_compute_cum, W_total_cum,
                 alpha=0.35, color="purple", label="actuating")
ax3.plot(t, W_total_cum, color="black", linewidth=1.5,
         label=r"$W_{\mathrm{total}}$")
ax3.set_xlabel(r"$t\;[\omega_0^{-1}]$", fontsize=12)
ax3.set_ylabel(r"$W_{\mathrm{cum}}\;[k_BT]$", fontsize=12)
ax3.set_title("(c) Thermodynamic cost decomposition", fontsize=11)
ax3.set_xlim(0, t_max)
ax3.legend(loc="upper left", fontsize=9)

plt.tight_layout()
outpath = "/Users/sidongliu/Documents/Sidong/Genesis Trilogy/T-Dome/fig_loop.pdf"
plt.savefig(outpath, bbox_inches="tight")
print(f"\nFigure saved to {outpath}")
print(f"Final V_cal = {V_cal[-1]:.6e}, V_uncal = {V_uncal[-1]:.4f}")
print(f"Final costs: sense={W_sense_cum[-1]:.1f}, compute={W_compute_cum[-1]:.1f}, "
      f"actuate={W_actuate_cum[-1]:.4f}, total={W_total_cum[-1]:.1f}")
