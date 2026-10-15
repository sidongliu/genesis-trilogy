#!/usr/bin/env python3
"""Generate fig_ego.pdf for T-DOME Paper II.

Two-panel figure:
  (a) Decoherence amplitudes |p_z(t)| and |p_x(t)| with backflow bands
  (b) Cumulative backflow harvested by ego vs symmetric agent

Parameters match Section 6 of Paper II:
  omega_0 = 1, lambda_z = 1, gamma_z = 0.5  (underdamped, non-Markov)
  lambda_x = 0.3, gamma_x = 5.0             (overdamped, ~Markov)
  C_budget = 2 h_mu
  tau_par = C_budget / (h_mu * D) = 2/(4) = 0.5   [D = dim Cl(0,2) = 4]
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# Parameters
lam_z, gam_z = 1.0, 0.5
lam_x, gam_x = 0.3, 5.0
t = np.linspace(0, 15, 4000)
tau_par = 0.5  # Updated: C_budget/(h_mu * D) = 2/(4) = 0.5

# Decoherence functions
# z-channel: underdamped (discriminant > 0)
disc_z = 4 * lam_z * gam_z - gam_z**2  # = 1.75
Omega_z = 0.5 * np.sqrt(disc_z)
p_z = np.exp(-gam_z * t / 2) * (
    np.cos(Omega_z * t) + (gam_z / (2 * Omega_z)) * np.sin(Omega_z * t)
)

# x-channel: overdamped (discriminant < 0)
disc_x = 4 * lam_x * gam_x - gam_x**2  # = -19
kappa_x = 0.5 * np.sqrt(-disc_x)
p_x = np.exp(-gam_x * t / 2) * (
    np.cosh(kappa_x * t) + (gam_x / (2 * kappa_x)) * np.sinh(kappa_x * t)
)

abs_pz = np.abs(p_z)
abs_px = np.abs(p_x)

# Backflow detection: d|p_z|/dt > 0
dp_z = np.gradient(abs_pz, t)
backflow_mask = dp_z > 0

# Cumulative backflow harvested by ego agent
# Integral of max(d|p_z|^2/dt, 0) dt
d_pz2 = np.gradient(abs_pz**2, t)
backflow_rate = np.maximum(d_pz2, 0)
cum_backflow_ego = np.cumsum(backflow_rate) * (t[1] - t[0])

# Symmetric agent: paralyzed at tau_par, harvests nothing after
cum_backflow_sym = np.zeros_like(t)
mask_pre = t <= tau_par
if np.any(mask_pre):
    cum_backflow_sym[mask_pre] = np.cumsum(
        np.maximum(np.gradient(abs_pz[mask_pre]**2, t[mask_pre]), 0)
    ) * (t[1] - t[0])
    # After paralysis, frozen at last value
    last_val = cum_backflow_sym[mask_pre][-1]
    cum_backflow_sym[~mask_pre] = last_val

# ---- Plotting ----
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

# Panel (a): Decoherence amplitudes
ax1.plot(t, abs_pz, color="steelblue", linewidth=1.5, label=r"$|p_z(t)|$ (dephasing)")
ax1.plot(t, abs_px, color="darkorange", linewidth=1.5, label=r"$|p_x(t)|$ (dissipative)")

# Shade backflow intervals
in_backflow = False
start = 0
for i in range(1, len(t)):
    if backflow_mask[i] and not in_backflow:
        start = t[i]
        in_backflow = True
    elif not backflow_mask[i] and in_backflow:
        ax1.axvspan(start, t[i], alpha=0.15, color="green")
        in_backflow = False
if in_backflow:
    ax1.axvspan(start, t[-1], alpha=0.15, color="green")

ax1.axvline(tau_par, color="red", linestyle="--", linewidth=1.5,
            label=r"$\tau_{\mathrm{par}} = 0.5$")
ax1.set_xlabel(r"$t\;[\omega_0^{-1}]$", fontsize=12)
ax1.set_ylabel(r"$|p_\alpha(t)|$", fontsize=12)
ax1.set_title("(a) Decoherence amplitudes", fontsize=12)
ax1.set_xlim(0, 15)
ax1.set_ylim(0, 1.05)

# Custom legend with backflow patch
handles, labels = ax1.get_legend_handles_labels()
handles.append(Patch(facecolor="green", alpha=0.15, label="backflow"))
ax1.legend(handles=handles, loc="upper right", fontsize=9)

# Panel (b): Cumulative backflow harvested
ax2.plot(t, cum_backflow_ego, color="steelblue", linewidth=1.5,
         label="ego agent")
ax2.plot(t, cum_backflow_sym, color="red", linestyle="--", linewidth=1.5,
         label="symmetric agent")
ax2.axvline(tau_par, color="red", linestyle=":", alpha=0.5, linewidth=1)
ax2.set_xlabel(r"$t\;[\omega_0^{-1}]$", fontsize=12)
ax2.set_ylabel(r"cumulative backflow $[\beta\mathcal{S}]$", fontsize=12)
ax2.set_title("(b) Cumulative backflow harvested", fontsize=12)
ax2.set_xlim(0, 15)
ax2.legend(loc="upper left", fontsize=10)

plt.tight_layout()
plt.savefig("/Users/sidongliu/Documents/Sidong/Genesis Trilogy/T-Dome/fig_ego.pdf",
            bbox_inches="tight")
print(f"Figure saved. tau_par = {tau_par}")
print(f"Final cumulative backflow: ego = {cum_backflow_ego[-1]:.4f}, sym = {cum_backflow_sym[-1]:.4f}")
print(f"First backflow onset: t* ~ {t[np.argmax(backflow_mask)]:.1f}")
