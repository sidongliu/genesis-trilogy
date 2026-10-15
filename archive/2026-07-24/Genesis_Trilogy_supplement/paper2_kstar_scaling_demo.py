#!/usr/bin/env python3
"""Paper II Companion Demo: Budget Forces Symmetry Breaking.

Demonstrates the SSB mechanism (Paper II) computationally:

  1. Under a hard processing budget (top-k coordinate updates per step),
     an agent must select which dimensions to track.  This selection
     breaks the environment's a priori symmetry --- SSB.

  2. The attention entropy H_attn (entropy of the update-frequency
     distribution) collapses from log(D) toward log(m) as the budget
     k tightens --- the agent confines its attention to the signal
     subspace.  This is a measurable order parameter for SSB.

  3. A budgeted selector that concentrates updates on signal-carrying
     dimensions systematically outperforms a random-allocation baseline
     with the same hard budget.

Model
-----
  y(t) = w*(t)^T x(t) + xi(t),   x(t) ~ N(0, I_D),  xi ~ N(0, sigma^2)

  w*(t) is SPARSE: only m out of D dimensions carry nonzero weight.
  The active support ROTATES every tau_switch steps (Experiment B),
  or remains fixed (Experiment A).

  Hard budget constraint: per step, the agent updates only k
  coordinates.  The selector ranks by importance (EMA of squared
  per-coordinate gradients); the random baseline picks k at random.

Does show
---------
  - Under budget constraints, attention entropy collapses from log(D)
    toward log(m) --- the agent confines its updates to the m-dimensional
    signal subspace.  This is spontaneous symmetry breaking (SSB).
  - For k <= m, H_attn < log(m): budget forces even tighter selection
    than the signal subspace.  For k > m, H_attn rises above log(m)
    as excess budget spills into noise dimensions.
  - A budgeted selector outperforms a random-allocation agent under
    the same hard budget.
  - The performance gap widens with slower drift (larger tau) and
    tighter budget (smaller k).

Does not show
-------------
  - Universality across environments, objectives, or architectures.
  - Tight constants or optimality of analytic bounds.
  - The delusion-correction cycle (addressed in Paper III demo).

Author: Sidong Liu, PhD (iBioStratix Ltd)
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ================================================================
# Configuration
# ================================================================

D = 64               # Ambient dimension
M = 8                # Signal dimensions (sparse support size)
T = 10_000           # Horizon per run (steps)
N_SEEDS = 10         # Independent replications
SIGMA_NOISE = 0.3    # Observation noise std
LR = 0.02            # Learning rate (SGD step size)
DECAY = 0.995        # Weight decay per step
ALPHA_ENTROPY = 0.002  # Signed-gradient EMA rate for Exp A (slow, stable)
ALPHA_PERF = 0.02      # Signed-gradient EMA rate for Exp B (fast adapt)
RESELECT_ENTROPY = 50  # Reselect period for Exp A (hysteresis/lock-in)
RESELECT_PERF = 1      # Reselect every step for Exp B (fast adapt)
ENTROPY_WINDOW = 1000  # Steps over which to measure H_attn

# Switching periods: controls drift rate Lambda = 1/tau
TAU_GRID = np.array([500, 1000, 2000])

# Budget scan: k values (how many coords updated per step)
K_BUDGET_GRID = np.array([2, 4, 6, 8, 10, 12, 16, 20, 24, 32, 48, 64])

BURNIN = 1000

BASE = os.path.dirname(os.path.abspath(__file__))


# ================================================================
# Environment: Sparse drifting linear prediction
# ================================================================

class SparseDriftWorld:
    """D-dimensional linear prediction with sparse rotating support.

    Only m out of D dimensions carry nonzero weight at any time.
    The active support rotates every tau_switch steps, modelling
    environmental drift.  Inactive dimensions have w*_i = 0.
    """

    def __init__(self, d, m, tau_switch, sigma_noise, seed=None):
        self.d = d
        self.m = m
        self.tau = tau_switch
        self.sigma_noise = sigma_noise
        self.rng = np.random.RandomState(seed)
        self.w = np.zeros(d)
        self._new_support()
        self.t = 0

    def _new_support(self):
        """Select new active dimensions and assign weights."""
        self.support = self.rng.choice(self.d, self.m, replace=False)
        self.w[:] = 0.0
        self.w[self.support] = self.rng.randn(self.m)
        self.w[self.support] /= np.linalg.norm(self.w[self.support])

    def step(self):
        """Advance one step; rotate support if due."""
        self.t += 1
        if self.t % self.tau == 0:
            self._new_support()

    def observe(self):
        """Sample (x, y) pair."""
        x = self.rng.randn(self.d)
        y = self.w @ x + self.sigma_noise * self.rng.randn()
        return x, y


# ================================================================
# Agent: Hard top-k selector (importance-weighted)
# ================================================================

class TopKSelectorAgent:
    """Updates only top-k coordinates per step (hard budget).

    Selection: rank coordinates by EMA of squared per-coordinate
    gradients (importance score).  This persistent importance tracking
    allows the agent to identify and commit to signal dimensions ---
    the computational analogue of spontaneous symmetry breaking.

    Tracks update counts for attention entropy computation.
    """

    def __init__(self, d, k, explore_prob=0.10, alpha_imp=0.02,
                 reselect_period=1, commitment=0.0):
        self.d = d
        self.k = min(k, d)
        self.w = np.zeros(d)
        self.u = np.zeros(d)  # EMA of signed gradient (directional)
        self.update_counts = np.zeros(d)
        self.explore_prob = explore_prob
        self.alpha_imp = alpha_imp
        self.reselect_period = reselect_period
        self.commitment = commitment  # w² weight in score (ego inertia)
        self.active_set = None  # Current locked-in top-k
        self.t_step = 0
        self.rng = np.random.RandomState(42)

    def reset_counts(self):
        """Reset update counts (for windowed entropy measurement)."""
        self.update_counts[:] = 0

    def step(self, x, y):
        """Predict, compute gradient, update top-k coords."""
        yhat = self.w @ x
        residual = y - yhat
        grad = residual * x  # Per-coordinate gradient

        # Update signed gradient EMA: u_i <- (1-a)*u_i + a*(r*x_i)
        # Signed accumulation: noise dims cancel (E=0), signal persists.
        self.u *= (1.0 - self.alpha_imp)
        self.u += self.alpha_imp * grad

        # Reselect active set every reselect_period steps (hysteresis)
        self.t_step += 1
        if (self.active_set is None
                or self.t_step % self.reselect_period == 0):
            score = np.abs(self.u) + self.commitment * self.w ** 2
            if self.k >= self.d:
                self.active_set = np.arange(self.d)
            else:
                self.active_set = np.argsort(score)[-self.k:]
                # Exploration: swap one with random dim
                if (self.rng.rand() < self.explore_prob
                        and self.k >= 2):
                    swap_pos = self.rng.randint(self.k)
                    rand_dim = self.rng.randint(self.d)
                    self.active_set[swap_pos] = rand_dim

        top_idx = self.active_set

        # Update only selected coordinates
        self.w *= DECAY
        self.w[top_idx] += LR * grad[top_idx]

        # Track update counts
        self.update_counts[top_idx] += 1

        return residual ** 2


# ================================================================
# Agent: Random-k baseline (budget-fair)
# ================================================================

class RandomKAgent:
    """Updates k randomly chosen coordinates per step.

    Same hard budget as TopKSelectorAgent (exactly k updates/step),
    but selection is uniform random instead of importance-weighted.
    This provides a budget-fair baseline: same mechanism, no SSB.
    """

    def __init__(self, d, k):
        self.d = d
        self.k = min(k, d)
        self.w = np.zeros(d)
        self.rng = np.random.RandomState(42)

    def step(self, x, y):
        """Predict, compute gradient, update random k coords."""
        yhat = self.w @ x
        residual = y - yhat
        grad = residual * x

        # Select k coords uniformly at random
        if self.k >= self.d:
            sel = np.arange(self.d)
        else:
            sel = self.rng.choice(self.d, self.k, replace=False)

        # Update only selected coordinates
        self.w *= DECAY
        self.w[sel] += LR * grad[sel]

        return residual ** 2


# ================================================================
# Metrics
# ================================================================

def attention_entropy(update_counts):
    """Shannon entropy of the update-frequency distribution.

    p_i = count_i / sum(counts).   H = -sum p_i ln p_i.
    """
    total = update_counts.sum()
    if total < 1:
        return np.log(len(update_counts))
    p = update_counts / total
    p = p[p > 0]
    return -np.sum(p * np.log(p))


def oracle_mse(agent_w, true_w):
    """Weight-space MSE: ||w_hat - w*||^2."""
    return np.sum((agent_w - true_w) ** 2)


# ================================================================
# Experiment A: Entropy collapse (fixed support)
# ================================================================

def run_entropy_collapse(d, m, k_grid, T, n_seeds, entropy_window):
    """Measure attention entropy H_attn for various budget k.

    Uses FIXED support (no rotation) to isolate the SSB mechanism.
    Entropy is measured over the last `entropy_window` steps only,
    after the agent's importance scores have converged.

    Returns: H_arr shape (n_k, n_seeds).
    """
    tau_fixed = T + 1  # Effectively infinite (no rotation)
    nk = len(k_grid)
    H_arr = np.zeros((nk, n_seeds))

    for ik, k in enumerate(k_grid):
        for s in range(n_seeds):
            seed = s * 10000 + ik * 100 + 3
            env = SparseDriftWorld(d, m, tau_fixed, SIGMA_NOISE,
                                  seed=seed)
            # Fixed support: slow EMA + hysteresis + commitment (w²)
            agent = TopKSelectorAgent(d, k, explore_prob=0.0,
                                      alpha_imp=ALPHA_ENTROPY,
                                      reselect_period=RESELECT_ENTROPY,
                                      commitment=1.0)
            agent.rng = np.random.RandomState(seed + 1)

            for t in range(T):
                if t == T - entropy_window:
                    agent.reset_counts()
                env.step()
                x, y = env.observe()
                agent.step(x, y)

            H_arr[ik, s] = attention_entropy(agent.update_counts)

    return H_arr


# ================================================================
# Experiment B: Performance gap (selector vs random-k)
# ================================================================

def run_performance_gap(d, m, tau_grid, k_grid, T, n_seeds):
    """Sweep (tau, k) x seeds to measure MSE gap.

    Both agents update exactly k dims per step (budget-fair).
    Selector uses importance-weighted top-k; baseline uses random-k.

    Returns:
      mse_selector: shape (n_tau, n_k, n_seeds)
      mse_random:   shape (n_tau, n_k, n_seeds)
    """
    nt = len(tau_grid)
    nk = len(k_grid)
    mse_sel = np.zeros((nt, nk, n_seeds))
    mse_rnd = np.zeros((nt, nk, n_seeds))

    total = nt * nk * n_seeds
    done = 0

    for jt, tau in enumerate(tau_grid):
        for ik, k in enumerate(k_grid):
            for s in range(n_seeds):
                seed = s * 10000 + jt * 1000 + ik * 10 + 7

                # --- Selector agent ---
                env = SparseDriftWorld(d, m, tau, SIGMA_NOISE,
                                      seed=seed)
                agent = TopKSelectorAgent(d, k, explore_prob=0.10,
                                          alpha_imp=ALPHA_PERF,
                                          reselect_period=RESELECT_PERF)
                agent.rng = np.random.RandomState(seed + 1)
                mse_sum_sel = 0.0
                count = 0
                for t in range(T):
                    env.step()
                    x, y = env.observe()
                    agent.step(x, y)
                    if t >= BURNIN:
                        mse_sum_sel += oracle_mse(agent.w, env.w)
                        count += 1
                mse_sel[jt, ik, s] = mse_sum_sel / max(count, 1)

                # --- Random-k agent (same environment) ---
                env2 = SparseDriftWorld(d, m, tau, SIGMA_NOISE,
                                       seed=seed)
                agent2 = RandomKAgent(d, k)
                agent2.rng = np.random.RandomState(seed + 2)
                mse_sum_rnd = 0.0
                count2 = 0
                for t in range(T):
                    env2.step()
                    x, y = env2.observe()
                    agent2.step(x, y)
                    if t >= BURNIN:
                        mse_sum_rnd += oracle_mse(agent2.w, env2.w)
                        count2 += 1
                mse_rnd[jt, ik, s] = mse_sum_rnd / max(count2, 1)

                done += 1
                if done % 20 == 0 or done == total:
                    print(f'  {done}/{total} trials complete',
                          flush=True)

    return mse_sel, mse_rnd


# ================================================================
# Plotting
# ================================================================

def plot_paper2_figure(H_arr, k_grid_entropy,
                       mse_sel, mse_rnd, tau_grid, k_grid,
                       D, M, path):
    """Two-panel figure: (a) entropy collapse, (b) performance gap."""

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))
    fig.subplots_adjust(wspace=0.30)

    n_seeds = H_arr.shape[1]
    z = 1.96

    colors = ['#1565C0', '#C62828', '#6A1B9A']
    markers = ['o', 's', 'D']

    # ================================================================
    # Panel (a): Attention entropy collapse (fixed support)
    # ================================================================
    ax1.set_title(r'(a) Attention entropy $H_{\mathrm{attn}}$ vs '
                  r'budget $k$ (fixed support)',
                  fontsize=11, pad=8)

    H_mean = H_arr.mean(axis=1)
    H_se = H_arr.std(axis=1) / np.sqrt(n_seeds)

    ax1.plot(k_grid_entropy, H_mean, 'o-', color='#1565C0',
             linewidth=2.0, markersize=5,
             label=r'$H_{\mathrm{attn}}$ (empirical)')
    ax1.fill_between(k_grid_entropy, H_mean - z * H_se,
                     H_mean + z * H_se,
                     color='#1565C0', alpha=0.15)

    # Reference lines
    # ln(D): no symmetry breaking (uniform updates)
    ax1.axhline(np.log(D), color='grey', linestyle=':', linewidth=0.8)
    ax1.text(D * 0.55, np.log(D) + 0.08,
             f'$\\ln D = \\ln {D}$', fontsize=9, color='grey')
    # ln(m): signal subspace confinement (SSB target)
    ax1.axhline(np.log(M), color='#2E7D32', linestyle='--',
                linewidth=1.5, alpha=0.8)
    ax1.text(D * 0.55, np.log(M) - 0.18,
             f'$\\ln m = \\ln {M}$', fontsize=9, color='#2E7D32')
    # ln(k) faint guide: full commitment limit
    k_fine = np.linspace(2, D, 200)
    ax1.plot(k_fine, np.log(k_fine), ':', color='#C62828',
             linewidth=0.8, alpha=0.4,
             label=r'$\ln k$ (full-lock limit)')
    # Vertical marker at k = m
    ax1.axvline(M, color='#2E7D32', linestyle='--', linewidth=0.8,
                alpha=0.4)

    ax1.set_xlabel(r'Budget $k$ (coords updated per step)', fontsize=10)
    ax1.set_ylabel(r'$H_{\mathrm{attn}} = -\sum p_i \ln p_i$',
                   fontsize=10)
    ax1.legend(fontsize=8.5, loc='lower right')
    ax1.set_xlim(0, D + 2)
    ax1.grid(True, alpha=0.15)

    # ================================================================
    # Panel (b): Performance gap (MSE_rnd - MSE_sel), rotating support
    # ================================================================
    ax2.set_title(r'(b) Selection advantage $\Delta\mathrm{MSE}'
                  r' = \mathrm{MSE}_{\mathrm{rnd}} - '
                  r'\mathrm{MSE}_{\mathrm{sel}}$',
                  fontsize=11, pad=8)

    for jt, tau in enumerate(tau_grid):
        gap_all = mse_rnd[jt] - mse_sel[jt]  # (n_k, n_seeds)
        gap_mean = gap_all.mean(axis=1)
        gap_se = gap_all.std(axis=1) / np.sqrt(n_seeds)

        label = (r'$\tau_{\mathrm{switch}} = '
                 + f'{int(tau)}$')
        ax2.plot(k_grid, gap_mean,
                 marker=markers[jt], linestyle='-',
                 color=colors[jt], linewidth=1.3,
                 markersize=4, label=label)
        ax2.fill_between(k_grid, gap_mean - z * gap_se,
                         gap_mean + z * gap_se,
                         color=colors[jt], alpha=0.10)

    ax2.axhline(0.0, color='grey', linestyle=':', linewidth=0.8)
    ax2.axvline(M, color='#2E7D32', linestyle='--', linewidth=0.8,
                alpha=0.5, label=f'$m = {M}$ (signal dims)')

    ax2.set_xlabel(r'Budget $k$ (coords updated per step)', fontsize=10)
    ax2.set_ylabel(r'$\Delta\mathrm{MSE}$', fontsize=10)
    ax2.legend(fontsize=7.5, loc='upper right')
    ax2.set_xlim(0, D + 2)
    ax2.grid(True, alpha=0.15)

    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved {os.path.basename(path)}')


def save_gap_csv(mse_sel, mse_rnd, tau_grid, k_grid, path):
    """Export performance gap data to CSV."""
    n_seeds = mse_sel.shape[2]
    with open(path, 'w') as f:
        f.write('tau_switch,k_budget,mse_sel_mean,mse_rnd_mean,'
                'gap_mean,gap_se\n')
        for jt, tau in enumerate(tau_grid):
            for ik, k in enumerate(k_grid):
                sel_m = mse_sel[jt, ik].mean()
                rnd_m = mse_rnd[jt, ik].mean()
                gap = mse_rnd[jt, ik] - mse_sel[jt, ik]
                gm = gap.mean()
                gse = gap.std() / np.sqrt(n_seeds)
                f.write(f'{int(tau)},{k},{sel_m:.6f},{rnd_m:.6f},'
                        f'{gm:.6f},{gse:.6f}\n')
    print(f'  Saved {os.path.basename(path)}')


# ================================================================
# Main
# ================================================================

if __name__ == '__main__':
    print('=' * 60)
    print('Paper II Companion Demo: Budget Forces Symmetry Breaking')
    print('=' * 60)

    print(f'\nParameters:')
    print(f'  D = {D}  |  m = {M}  |  T = {T:,}  |  seeds = {N_SEEDS}')
    print(f'  sigma_noise = {SIGMA_NOISE}  |  lr = {LR}  |  '
          f'decay = {DECAY}')
    print(f'  alpha_entropy = {ALPHA_ENTROPY}  |  '
          f'alpha_perf = {ALPHA_PERF}  |  '
          f'entropy_window = {ENTROPY_WINDOW}')
    print(f'  tau_switch grid = {list(TAU_GRID.astype(int))}')
    print(f'  k_budget grid = {list(K_BUDGET_GRID.astype(int))}')

    # ---- Experiment A: Entropy collapse (fixed support) ----
    print(f'\n[1/2] Attention entropy collapse (fixed support) ...')
    H_arr = run_entropy_collapse(D, M, K_BUDGET_GRID,
                                 T, N_SEEDS, ENTROPY_WINDOW)
    H_mean = H_arr.mean(axis=1)
    print(f'  H_attn range: [{H_mean.min():.2f}, '
          f'{H_mean.max():.2f}]  '
          f'(ln D = {np.log(D):.2f}, ln m = {np.log(M):.2f})')
    for ik, k in enumerate(K_BUDGET_GRID):
        print(f'    k={k:3d}: H_attn = {H_mean[ik]:.3f}'
              f'  (ln k = {np.log(k):.2f})')

    # ---- Experiment B: Performance gap (rotating support) ----
    total_trials = (len(TAU_GRID) * len(K_BUDGET_GRID) * N_SEEDS)
    print(f'\n[2/2] Performance gap sweep: '
          f'{len(TAU_GRID)} tau x '
          f'{len(K_BUDGET_GRID)} budgets x {N_SEEDS} seeds '
          f'= {total_trials} trials ...')
    mse_sel, mse_rnd = run_performance_gap(
        D, M, TAU_GRID, K_BUDGET_GRID, T, N_SEEDS)

    for jt, tau in enumerate(TAU_GRID):
        gap_low = (mse_rnd[jt, 0] - mse_sel[jt, 0]).mean()
        gap_high = (mse_rnd[jt, -1] - mse_sel[jt, -1]).mean()
        print(f'  tau={int(tau)}: gap from {gap_low:.4f} '
              f'(k={K_BUDGET_GRID[0]}) to {gap_high:.4f} '
              f'(k={K_BUDGET_GRID[-1]})')

    # ---- Plot ----
    print(f'\nGenerating figures ...')
    plot_paper2_figure(H_arr, K_BUDGET_GRID,
                       mse_sel, mse_rnd, TAU_GRID, K_BUDGET_GRID,
                       D, M,
                       os.path.join(BASE,
                                    'fig_paper2_kstar_scaling.pdf'))

    save_gap_csv(mse_sel, mse_rnd, TAU_GRID, K_BUDGET_GRID,
                 os.path.join(BASE, 'kstar_scaling_data.csv'))

    print('\nDone.')
