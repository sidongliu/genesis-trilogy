#!/usr/bin/env python3
"""T-DOME Minimum Viable Demo: Delusion Trap & Phase Diagram.

Demonstrates the core T-DOME phenomena computationally:
  1. Delusion Trap (Paper II): a fixed-frame agent's foreground tracking
     improves while its true full-space error grows under background drift.
  2. Sentinel Detection (Paper III): residual-background correlation
     provides a drift signal growing quadratically with accumulated drift.
  3. Loop Calibration (Paper III): re-selecting foreground dimensions
     restores tracking, at a measurable thermodynamic cost.
  4. Phase Diagram: drift rate x calibration threshold reveals
     Stable / Looping / Chaotic regimes with a critical boundary.

Model
-----
  y(t) = w(t) . x + noise,  x ~ N(0, I_d)

  w(t) drifts on the unit sphere.  Drift is applied ONLY to background
  dimensions (indices k..d-1), so signal migrates monotonically from
  the ego's foreground to its blind spot.

  Ego prediction:  y_hat = w_ego . x[fg_idx]
  Full MSE = ||w_true[fg] - w_ego||^2 + ||w_true[bg]||^2 + sigma^2
           = (foreground tracking error) + (hidden sector) + (noise)

  The Delusion: foreground tracking error -> 0 (ego learns),
  but hidden sector grows -> full MSE increases.  The ego cannot
  distinguish hidden-sector signal from noise.

Author: Sidong Liu, PhD (iBioStratix Ltd)
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

# ================================================================
# Configuration
# ================================================================
D = 20          # Full dimension (the world)
K = 5           # Ego foreground dimension (the self)
NOISE = 0.1     # Observation noise std
LR = 0.01       # SGD learning rate
DECAY = 0.998   # Weight decay (prevents w explosion)
EMA_FAST = 0.02 # Fast EMA for mismatch signal
EMA_SLOW = 0.01 # Slow EMA for gradient buffer

BASE = os.path.dirname(os.path.abspath(__file__))


# ================================================================
# Environment: Drifting World
# ================================================================
class DriftingWorld:
    """Linear prediction target drifting from foreground to background.

    w(0) is supported entirely on dims 0..k-1.
    At each step, background dims receive random perturbations;
    foreground dims do not.  After renormalisation, ||w_fg|| shrinks
    and ||w_bg|| grows monotonically (up to fluctuations).
    """

    def __init__(self, d, k, drift_rate, seed=None):
        if seed is not None:
            np.random.seed(seed)
        self.d, self.k = d, k
        self.drift_rate = drift_rate
        self.w = np.zeros(d)
        self.w[:k] = np.random.randn(k)
        self.w /= np.linalg.norm(self.w)

    def step(self):
        noise = np.zeros(self.d)
        noise[self.k:] = self.drift_rate * np.random.randn(self.d - self.k)
        self.w += noise
        self.w /= np.linalg.norm(self.w)

    def observe(self):
        x = np.random.randn(self.d)
        y = x @ self.w + NOISE * np.random.randn()
        return x, y

    def fg_energy(self):
        """Fraction of ||w||^2 in the foreground."""
        return np.sum(self.w[:self.k] ** 2)


# ================================================================
# Agent 1: Fixed Ego (no loop, no sentinel)
# ================================================================
class FixedEgo:
    """Learns a linear model on a FIXED foreground subspace (dims 0..k-1).

    This agent embodies Paper II's "frozen gauge".  It can minimise
    foreground tracking error but is blind to background drift.
    """

    def __init__(self, k):
        self.k = k
        self.w = np.zeros(k)

    def step(self, x, y):
        z = x[: self.k]
        yhat = z @ self.w
        r = y - yhat
        self.w = DECAY * self.w + LR * r * z
        return r


# ================================================================
# Agent 2: Loop Agent (ego + sentinel + re-gauge)
# ================================================================
class LoopAgent:
    """Ego with sentinel monitoring and frame re-selection.

    Sentinel signal
    ---------------
    After ego predicts y_hat, compute residual r = y - y_hat.
    For each dimension i, track  grad_buf[i] = EMA of |r * x_i|.
    This approximates the absolute gradient of the loss w.r.t.
    a weight on dimension i --- high values mean dimension i
    carries learnable signal that the ego is NOT capturing.

    Mismatch = fraction of top-K gradient dims NOT in current foreground.
    Range [0,1]: 0 = perfect alignment, 1 = total misalignment.
    Rising mismatch => the ego's frame is stale.

    Re-gauge
    --------
    When mismatch EMA exceeds threshold:
      1. Select the top-K dimensions from grad_buf as new foreground.
      2. Transfer weights for dimensions that remain in the new fg.
      3. Zero weights for newly added dimensions.
      4. Halve grad_buf (partial memory, not full amnesia).
    """

    def __init__(self, d, k, threshold, cooldown=200):
        self.d, self.k = d, k
        self.threshold = threshold
        self.cooldown = cooldown        # settling time after re-gauge
        self.cooldown_counter = 0
        self.fg_idx = np.arange(k)
        self.w = np.zeros(k)
        self.mismatch_ema = 0.0
        self.calib_count = 0
        self.grad_buf = np.zeros(d)

    def step(self, x, y):
        z = x[self.fg_idx]
        yhat = z @ self.w
        r = y - yhat
        # SGD on foreground weights
        self.w = DECAY * self.w + LR * r * z

        # ---- Sentinel ----
        # Per-dimension absolute gradient proxy
        abs_grad = np.abs(r * x)
        self.grad_buf = (1 - EMA_SLOW) * self.grad_buf + EMA_SLOW * abs_grad

        # Mismatch: fraction of top-K gradient dims NOT in current foreground
        top_k = np.argsort(self.grad_buf)[-self.k:]
        overlap = len(np.intersect1d(top_k, self.fg_idx))
        mismatch = 1.0 - overlap / self.k
        self.mismatch_ema = (1 - EMA_FAST) * self.mismatch_ema + EMA_FAST * mismatch

        calibrated = False
        if self.cooldown_counter > 0:
            self.cooldown_counter -= 1
        elif self.mismatch_ema > self.threshold:
            self._regauge()
            calibrated = True
            self.calib_count += 1
            self.cooldown_counter = self.cooldown

        return r, calibrated

    def _regauge(self):
        new_fg = np.argsort(self.grad_buf)[-self.k:]
        # Transfer weights for surviving dims
        new_w = np.zeros(self.k)
        for i, dim in enumerate(new_fg):
            idx_in_old = np.where(self.fg_idx == dim)[0]
            if len(idx_in_old) > 0:
                new_w[i] = self.w[idx_in_old[0]]
        self.fg_idx = new_fg
        self.w = new_w
        self.mismatch_ema = 0.0
        self.grad_buf *= 0.5   # partial reset, not amnesia


# ================================================================
# Oracle metrics (computed externally, not visible to agents)
# ================================================================
def oracle_fg_err(agent_w, env_w_fg):
    """||w_ego - w_true[:K]||^2  (foreground tracking quality)."""
    return np.sum((agent_w - env_w_fg) ** 2)


def oracle_full_err_ego(agent_w, env_w, k):
    """Full-space expected MSE for fixed ego (dims 0..k-1)."""
    fg_err = np.sum((agent_w - env_w[:k]) ** 2)
    bg_err = np.sum(env_w[k:] ** 2)
    return fg_err + bg_err


def oracle_full_err_loop(agent, env_w):
    """Full-space expected MSE for loop agent."""
    w_full = np.zeros(len(env_w))
    w_full[agent.fg_idx] = agent.w
    fg_err = np.sum((w_full[agent.fg_idx] - env_w[agent.fg_idx]) ** 2)
    bg_idx = np.setdiff1d(np.arange(len(env_w)), agent.fg_idx)
    bg_err = np.sum(env_w[bg_idx] ** 2)
    return fg_err + bg_err


# ================================================================
# Experiment 1: Delusion Trap Time Series
# ================================================================
def run_delusion_trap(drift_rate=0.02, T=5000, seed=42):
    """Single run showing delusion trap + loop correction."""
    np.random.seed(seed)
    env = DriftingWorld(D, K, drift_rate, seed=seed)
    ego = FixedEgo(K)
    loop = LoopAgent(D, K, threshold=0.25)

    hist = dict(
        ego_fg=[], ego_full=[], loop_full=[],
        fg_frac=[], mismatch=[], calib=[]
    )

    for t in range(T):
        env.step()
        x, y = env.observe()
        ego.step(x, y)
        _, calibrated = loop.step(x, y)
        if calibrated:
            hist['calib'].append(t)

        hist['ego_fg'].append(oracle_fg_err(ego.w, env.w[:K]))
        hist['ego_full'].append(oracle_full_err_ego(ego.w, env.w, K))
        hist['loop_full'].append(oracle_full_err_loop(loop, env.w))
        hist['fg_frac'].append(env.fg_energy())
        hist['mismatch'].append(loop.mismatch_ema)

    hist['threshold'] = loop.threshold
    for k_ in hist:
        if k_ not in ('calib', 'threshold'):
            hist[k_] = np.array(hist[k_])
    hist['calib'] = np.array(hist['calib'])
    hist['T'] = T
    return hist


def smooth_ema(arr, alpha=0.005):
    out = np.empty_like(arr)
    out[0] = arr[0]
    for i in range(1, len(arr)):
        out[i] = (1 - alpha) * out[i - 1] + alpha * arr[i]
    return out


def plot_delusion_trap(h, path):
    fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True,
                             gridspec_kw={'hspace': 0.25})

    # ---- Panel (a): The Delusion Trap ----
    ax = axes[0]
    ax.set_title(r"(a)  The Delusion Trap:  $\mathcal{S}_{\mathrm{vis}}$ "
                 r"improves while $\mathcal{S}_{\mathrm{full}}$ degrades",
                 fontsize=11)
    s = smooth_ema
    ax.plot(s(h['ego_fg']), color='#1565C0', lw=2,
            label=r'Foreground tracking error  $\|\mathbf{w}_{\mathrm{ego}}'
                  r'-\mathbf{w}_{\mathrm{fg}}^*\|^2$')
    ax.plot(s(h['ego_full']), color='#C62828', lw=2,
            label=r'True full-space error  (fg + hidden)')
    ax.fill_between(range(h['T']), s(h['ego_fg']), s(h['ego_full']),
                    color='#C62828', alpha=0.12,
                    label='Hidden sector (invisible to ego)')
    ax.set_ylabel('Squared error')
    ax.legend(loc='center left', fontsize=8.5)
    ax.grid(True, alpha=0.15)

    # ---- Panel (b): Sentinel Signal ----
    ax = axes[1]
    ax.set_title('(b)  Sentinel Mismatch Signal  &  Calibration Events',
                 fontsize=11)
    ax.plot(h['mismatch'], color='#6A1B9A', lw=0.8, alpha=0.85,
            label='Mismatch EMA  (frame staleness)')
    thresh = h.get('threshold', 0.25)
    ax.axhline(thresh, color='#E65100', lw=1.2, ls='--', alpha=0.7,
               label=f'Threshold $\\theta = {thresh}$')
    if len(h['calib']) > 0:
        ax.vlines(h['calib'], 0, thresh * 1.1,
                  color='#E65100', alpha=0.35, lw=0.8,
                  label=f"Re-gauge events ({len(h['calib'])})")
    ax.set_ylim(-0.02, min(1.0, thresh * 2.5))
    ax.set_ylabel('Mismatch  (0 = aligned, 1 = stale)')
    ax.legend(loc='upper left', fontsize=8.5)
    ax.grid(True, alpha=0.15)

    # ---- Panel (c): Ego vs Loop ----
    ax = axes[2]
    ax.set_title('(c)  Fixed Ego  vs  Loop Agent  (calibrated):  True Performance',
                 fontsize=11)
    ax.plot(s(h['ego_full']), color='#C62828', lw=2, alpha=0.8,
            label='Fixed Ego  (no loop)')
    ax.plot(s(h['loop_full']), color='#2E7D32', lw=2,
            label='Loop Agent  (calibrated)')
    ax2 = ax.twinx()
    ax2.plot(h['fg_frac'], color='grey', lw=0.8, ls='--', alpha=0.4)
    ax2.set_ylabel('Foreground energy fraction', color='grey', fontsize=8)
    ax2.tick_params(axis='y', labelcolor='grey')
    ax.set_xlabel('Time steps')
    ax.set_ylabel('True squared error')
    ax.legend(loc='upper left', fontsize=8.5)
    ax.grid(True, alpha=0.15)

    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {path}")


# ================================================================
# Experiment 2: Phase Diagram  (drift x threshold)
# ================================================================
def run_phase_diagram(drift_vals, cooldown_vals, T=3000, n_seeds=4):
    """2-D scan: drift rate x cooldown period.

    For each grid point, runs both a FixedEgo and a LoopAgent
    to compute the performance ratio (loop helps or hurts?).
    """
    n_d, n_c = len(drift_vals), len(cooldown_vals)
    loop_map = np.zeros((n_c, n_d))
    ego_map = np.zeros((n_c, n_d))
    rate_map = np.zeros((n_c, n_d))

    total = n_c * n_d
    done = 0

    for j, drift in enumerate(drift_vals):
        for i, cooldown in enumerate(cooldown_vals):
            seed_loop, seed_ego, seed_rate = [], [], []
            for s in range(n_seeds):
                sd = s * 10000 + i * 100 + j
                np.random.seed(sd)
                env = DriftingWorld(D, K, drift, seed=sd)
                ego = FixedEgo(K)
                agent = LoopAgent(D, K, threshold=0.25,
                                  cooldown=int(cooldown))

                err_loop = err_ego = 0.0
                half = T // 2
                for t in range(T):
                    env.step()
                    x, y = env.observe()
                    ego.step(x, y)
                    agent.step(x, y)
                    if t >= half:
                        err_loop += oracle_full_err_loop(agent, env.w)
                        err_ego += oracle_full_err_ego(ego.w, env.w, K)

                seed_loop.append(err_loop / half)
                seed_ego.append(err_ego / half)
                seed_rate.append(agent.calib_count / T)

            loop_map[i, j] = np.mean(seed_loop)
            ego_map[i, j] = np.mean(seed_ego)
            rate_map[i, j] = np.mean(seed_rate)
            done += 1
            if done % 20 == 0:
                print(f"    {done}/{total} grid points done")

    return loop_map, ego_map, rate_map


def plot_phase_diagram(drift_vals, cooldown_vals, loop_map, ego_map,
                       rate_map, path):
    from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))
    Drift, Cool = np.meshgrid(drift_vals, cooldown_vals)

    # (a) Absolute improvement: ego MSE - loop MSE
    # Positive = loop helps, negative = loop hurts
    rwg = LinearSegmentedColormap.from_list(
        'RdWtGn', ['#d73027', '#fdae61', 'white', '#a6d96a', '#1a9850'])
    improvement = ego_map - loop_map
    vmax = max(np.percentile(improvement, 97), 0.05)
    norm = TwoSlopeNorm(vmin=-vmax * 0.3, vcenter=0, vmax=vmax)
    im1 = ax1.pcolormesh(Drift, Cool, improvement, cmap=rwg, norm=norm,
                         shading='gouraud')
    cs = ax1.contour(Drift, Cool, improvement, levels=[0],
                     colors='black', linewidths=1.5, linestyles='--')
    ax1.clabel(cs, fmt=r'$\Delta=0$', fontsize=8)
    ax1.set_xlabel(r'Drift rate  $\Lambda$', fontsize=11)
    ax1.set_ylabel(r'Cooldown period  $\tau$  (steps)', fontsize=11)
    ax1.set_title(r'(a)  Performance Gain  ($\mathrm{MSE_{ego}} - \mathrm{MSE_{loop}}$)',
                  fontsize=11)
    ax1.set_yscale('log')
    cb1 = plt.colorbar(im1, ax=ax1)
    cb1.set_label('green = loop helps,  red = loop hurts', fontsize=8.5)

    # (b) Thermodynamic cost heatmap
    im2 = ax2.pcolormesh(Drift, Cool, rate_map, cmap='YlOrRd',
                         shading='gouraud')
    ax2.set_xlabel(r'Drift rate  $\Lambda$', fontsize=11)
    ax2.set_ylabel(r'Cooldown period  $\tau$  (steps)', fontsize=11)
    ax2.set_title('(b)  Calibration Frequency  (Thermodynamic Cost)',
                  fontsize=11)
    ax2.set_yscale('log')
    cb2 = plt.colorbar(im2, ax=ax2)
    cb2.set_label('Re-gauges per step', fontsize=9)

    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {path}")


# ================================================================
# Alpha_min extraction and plotting
# ================================================================
def extract_boundaries(drift_vals, cooldown_vals, ego_map, loop_map):
    """Extract optimal calibration boundary from the phase diagram.

    Uses Boltzmann-weighted geometric mean of cooldowns (soft argmin)
    for robustness to grid noise, then 3-point moving-average smoothing.

    Returns: (lambdas, tau_opt, alpha_opt).
    """
    log_cool = np.log(cooldown_vals)
    lambdas, tau_raw = [], []

    for j, drift in enumerate(drift_vals):
        col = loop_map[:, j]
        # Soft argmin: Boltzmann-weighted geometric mean
        min_mse = col.min()
        temp = 0.03  # temperature: ~3 grid points contribute
        weights = np.exp(-(col - min_mse) / temp)
        weights /= weights.sum()
        log_tau_star = np.average(log_cool, weights=weights)
        lambdas.append(drift)
        tau_raw.append(np.exp(log_tau_star))

    lambdas = np.array(lambdas)
    tau_raw = np.array(tau_raw)

    # Post-smooth: 3-point moving average in log space
    log_raw = np.log(tau_raw)
    n = len(log_raw)
    log_smooth = np.empty(n)
    log_smooth[0] = (2 * log_raw[0] + log_raw[1]) / 3
    log_smooth[-1] = (log_raw[-2] + 2 * log_raw[-1]) / 3
    for i in range(1, n - 1):
        log_smooth[i] = (log_raw[i-1] + log_raw[i] + log_raw[i+1]) / 3
    tau_opts = np.exp(log_smooth)
    alpha_opts = 1.0 / tau_opts
    return lambdas, tau_opts, alpha_opts


def plot_alpha_opt(lambdas, tau_opts, alpha_opts, path):
    """Standalone α_opt(Λ) curve — the quantitative signature.

    Shows optimal calibration frequency as a function of drift rate.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    # Left: τ_opt(Λ) — optimal cooldown (decreasing = needs tighter calibration)
    ax1.plot(lambdas, tau_opts, 'o-', color='#1565C0', lw=2, ms=5)
    ax1.set_xlabel(r'Drift rate  $\Lambda$', fontsize=11)
    ax1.set_ylabel(r'Optimal cooldown  $\tau_{\mathrm{opt}}(\Lambda)$  (steps)',
                   fontsize=11)
    ax1.set_title('Optimal recalibration period', fontsize=10.5)
    ax1.set_yscale('log')
    ax1.grid(True, alpha=0.2)
    ax1.text(0.95, 0.95, 'Faster drift\n' r'$\Rightarrow$ tighter calibration',
             transform=ax1.transAxes, ha='right', va='top',
             fontsize=9, bbox=dict(boxstyle='round', fc='white', alpha=0.8))

    # Right: α_opt(Λ) = 1/τ_opt — minimum calibration budget
    ax2.plot(lambdas, alpha_opts, 's-', color='#C62828', lw=2, ms=5)
    ax2.fill_between(lambdas, alpha_opts, 0, color='#C62828', alpha=0.08)
    ax2.set_xlabel(r'Drift rate  $\Lambda$', fontsize=11)
    ax2.set_ylabel(r'$\alpha_{\mathrm{opt}}(\Lambda) = 1/\tau_{\mathrm{opt}}$',
                   fontsize=11)
    ax2.set_title('Optimal calibration frequency', fontsize=10.5)
    ax2.set_yscale('log')
    ax2.grid(True, alpha=0.2)
    ax2.text(0.95, 0.05,
             r'$\alpha_{\mathrm{opt}}$ grows with $\Lambda$:'
             '\nmore drift demands more recalibration',
             transform=ax2.transAxes, ha='right', va='bottom',
             fontsize=9, bbox=dict(boxstyle='round', fc='white', alpha=0.8))

    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {path}")


def plot_phase_with_boundary(drift_vals, cooldown_vals, loop_map, ego_map,
                             rate_map, lambdas, tau_opts, path):
    """Phase diagram with τ_opt and Δ=0 boundaries overlaid."""
    from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))
    Drift, Cool = np.meshgrid(drift_vals, cooldown_vals)

    # (a) Performance gain + τ_opt curve
    rwg = LinearSegmentedColormap.from_list(
        'RdWtGn', ['#d73027', '#fdae61', 'white', '#a6d96a', '#1a9850'])
    improvement = ego_map - loop_map
    vmax = max(np.percentile(improvement, 97), 0.05)
    norm = TwoSlopeNorm(vmin=-vmax * 0.3, vcenter=0, vmax=vmax)
    im1 = ax1.pcolormesh(Drift, Cool, improvement, cmap=rwg, norm=norm,
                         shading='gouraud')
    # Δ=0 contour
    cs = ax1.contour(Drift, Cool, improvement, levels=[0],
                     colors='black', linewidths=1.2, linestyles='--')
    ax1.clabel(cs, fmt=r'$\Delta=0$', fontsize=7)
    # τ_opt curve
    ax1.plot(lambdas, tau_opts, 'k-', lw=2.5, zorder=5,
             label=r'$\tau_{\mathrm{opt}}(\Lambda)$')
    ax1.plot(lambdas, tau_opts, 'w:', lw=1, zorder=5)
    ax1.set_xlabel(r'Drift rate  $\Lambda$', fontsize=11)
    ax1.set_ylabel(r'Cooldown period  $\tau$  (steps)', fontsize=11)
    ax1.set_title(r'(a)  Performance Gain  with  $\tau_{\mathrm{opt}}$',
                  fontsize=11)
    ax1.set_yscale('log')
    ax1.legend(loc='upper right', fontsize=8.5)
    cb1 = plt.colorbar(im1, ax=ax1)
    cb1.set_label(r'$\Delta = \mathrm{MSE_{ego}} - \mathrm{MSE_{loop}}$',
                  fontsize=9)

    # (b) Cost + τ_opt
    im2 = ax2.pcolormesh(Drift, Cool, rate_map, cmap='YlOrRd',
                         shading='gouraud')
    ax2.plot(lambdas, tau_opts, 'k-', lw=2.5, zorder=5,
             label=r'$\tau_{\mathrm{opt}}(\Lambda)$')
    ax2.plot(lambdas, tau_opts, 'w:', lw=1, zorder=5)
    ax2.set_xlabel(r'Drift rate  $\Lambda$', fontsize=11)
    ax2.set_ylabel(r'Cooldown period  $\tau$  (steps)', fontsize=11)
    ax2.set_title(r'(b)  Cost at optimal $\tau_{\mathrm{opt}}$',
                  fontsize=11)
    ax2.set_yscale('log')
    ax2.legend(loc='upper right', fontsize=8.5)
    cb2 = plt.colorbar(im2, ax=ax2)
    cb2.set_label('Re-gauges per step', fontsize=9)

    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {path}")


def save_alpha_csv(lambdas, tau_opts, alpha_opts, path):
    """Export boundary data as CSV."""
    with open(path, 'w') as f:
        f.write('Lambda,tau_opt,alpha_opt\n')
        for l, t, a in zip(lambdas, tau_opts, alpha_opts):
            f.write(f'{l:.6f},{t:.2f},{a:.6f}\n')
    print(f"  Saved {path}")


# ================================================================
# Main
# ================================================================
if __name__ == '__main__':
    print('=' * 60)
    print('T-DOME  Minimum Viable Demo')
    print('=' * 60)

    # ---- Experiment 1 ----
    print('\n[1/2] Delusion Trap simulation  (T=5000) ...')
    h = run_delusion_trap(drift_rate=0.02, T=5000, seed=42)
    print(f'  Ego fg_err (last 500):  {h["ego_fg"][-500:].mean():.4f}')
    print(f'  Ego full_err (last 500): {h["ego_full"][-500:].mean():.4f}')
    print(f'  Loop full_err (last 500): {h["loop_full"][-500:].mean():.4f}')
    print(f'  Re-gauge events: {len(h["calib"])}')
    plot_delusion_trap(h, os.path.join(BASE, 'fig_delusion_trap.pdf'))

    # ---- Experiment 2 ----
    print('\n[2/3] Phase Diagram scan  (drift x cooldown) ...')
    drift_vals = np.linspace(0.005, 0.08, 16)
    cooldown_vals = np.logspace(np.log10(15), np.log10(800), 16)
    loop_map, ego_map, rate_map = run_phase_diagram(
        drift_vals, cooldown_vals, T=4000, n_seeds=6)
    plot_phase_diagram(drift_vals, cooldown_vals, loop_map, ego_map,
                       rate_map, os.path.join(BASE, 'fig_phase_diagram.pdf'))

    # ---- Experiment 3: optimal calibration boundary ----
    print('\n[3/3] Extracting optimal calibration curve ...')
    lambdas, tau_opts, alpha_opts = extract_boundaries(
        drift_vals, cooldown_vals, ego_map, loop_map)
    print(f'  Drift points: {len(lambdas)}')
    print(f'  tau_opt range: [{tau_opts.min():.0f}, {tau_opts.max():.0f}]')
    print(f'  alpha_opt range: [{alpha_opts.min():.5f}, {alpha_opts.max():.5f}]')
    plot_alpha_opt(lambdas, tau_opts, alpha_opts,
                   os.path.join(BASE, 'fig_alpha_opt.pdf'))
    plot_phase_with_boundary(drift_vals, cooldown_vals, loop_map, ego_map,
                             rate_map, lambdas, tau_opts,
                             os.path.join(BASE, 'fig_phase_with_boundary.pdf'))
    save_alpha_csv(lambdas, tau_opts, alpha_opts,
                   os.path.join(BASE, 'alpha_opt_curve.csv'))

    print('\nDone.')
