#!/usr/bin/env python3
"""Paper I Companion Demo: Memory Surpasses a Markov Ceiling.

Demonstrates the Markovian Ceiling (Paper I, Theorem 1) computationally:

  1. A 2-hidden-state HMM with aliased observations creates a minimal
     partially-observed environment where optimal action requires temporal
     integration beyond any fixed observation window.

  2. Optimal Markov-k agents (using the last k observations) exhibit a
     performance ceiling: their accuracy saturates once the correlation
     length ell >> k, because they cannot accumulate more evidence.

  3. An optimal Memory agent (full Bayes filter over all observations)
     continues to improve with correlation length, breaking through
     every finite-order Markov ceiling.

Model
-----
  Hidden state:  s_t in {0, 1},  p(s_{t+1} = s_t) = 1 - epsilon.
  Observation:   o_t in {0, 1},  p(o=0 | s=0) = 0.5 + delta,
                                  p(o=0 | s=1) = 0.5 - delta.
  Action:        a_t in {0, 1},  reward r_t = 1[a_t = s_t].

  Correlation length proxy: ell ~ 1/epsilon.
  Aliasing: delta controls observability (delta=0 => max aliasing).

  Both Markov-k and Memory agents use the TRUE model parameters and
  compute optimal posteriors; the comparison is purely about how much
  history is available, not about learning algorithms.

Does show
---------
  - A reproducible regime where finite-order Markov agents plateau while
    a memory/belief agent improves (the "Markov ceiling" signature).
  - The gap Delta_R = R_mem - R_Markov-k increases with correlation length.
  - Higher k pushes the ceiling higher but can never eliminate it.

Does not show
-------------
  - Universality across environments, objectives, or architectures.
  - Tight constants or optimality of analytic bounds.
  - That the Bayes filter is the unique optimal memory strategy.

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

T = 100_000          # Horizon per run (steps)
N_SEEDS = 10         # Independent replications
DELTA = 0.05         # Observation asymmetry: p(o=0|s=0) = 0.5 + delta
                     # Small enough that single observations are near-useless;
                     # accumulation over many steps is required.

# Markov window sizes to compare
K_VALUES = [1, 2, 4, 8]

# Epsilon grid: transition noise, log-spaced
# ell ~ 1/eps: small eps => long correlations => ceiling becomes visible
N_EPS = 16
EPS_GRID = np.logspace(-3, -0.7, N_EPS)   # [0.001, ..., 0.2]
ELL_GRID = 1.0 / EPS_GRID                  # correlation lengths

# Reward averaging
BURNIN = 2000        # Discard initial transient

BASE = os.path.dirname(os.path.abspath(__file__))


# ================================================================
# Environment: Aliased Hidden Markov Model
# ================================================================

class AliasedHMM:
    """Two-hidden-state HMM with aliased observations.

    Hidden state s_t in {0,1} evolves as a Markov chain with
    persistence probability 1-epsilon.  Observations o_t in {0,1}
    are drawn with p(o=0|s=0) = 0.5+delta, p(o=0|s=1) = 0.5-delta.

    When delta is small, a single observation gives very little
    information about the hidden state.  The agent must integrate
    observations over time to build confident estimates.
    """

    def __init__(self, epsilon, delta, seed=None):
        self.epsilon = epsilon
        self.delta = delta
        self.rng = np.random.RandomState(seed)
        # Emission matrix: E[s, o] = p(o | s)
        self.E = np.array([[0.5 + delta, 0.5 - delta],
                           [0.5 - delta, 0.5 + delta]])
        # Initial hidden state (uniform)
        self.state = self.rng.randint(2)

    def step(self):
        """Advance hidden state and emit observation. Returns o_t."""
        if self.rng.rand() < self.epsilon:
            self.state = 1 - self.state
        p_o0 = self.E[self.state, 0]
        obs = 0 if self.rng.rand() < p_o0 else 1
        return obs

    def reward(self, action):
        """Returns 1.0 if action matches hidden state, else 0.0."""
        return 1.0 if action == self.state else 0.0


# ================================================================
# Bayes filter utilities
# ================================================================

def bayes_predict(b, epsilon):
    """Prediction step: propagate belief through transition model.
    b = p(s=1 | past observations).
    Returns b_predict = p(s_{t+1}=1 | past observations).
    """
    return epsilon * (1.0 - b) + (1.0 - epsilon) * b


def bayes_update(b_pred, obs, E):
    """Update step: incorporate new observation.
    b_pred = p(s=1) prior, obs = observed value, E = emission matrix.
    Returns posterior p(s=1 | obs, prior).
    """
    L0 = E[0, obs]  # p(obs | s=0)
    L1 = E[1, obs]  # p(obs | s=1)
    unnorm_1 = L1 * b_pred
    unnorm_0 = L0 * (1.0 - b_pred)
    b_post = unnorm_1 / (unnorm_0 + unnorm_1)
    return np.clip(b_post, 1e-15, 1.0 - 1e-15)


def run_kstep_filter(obs_window, epsilon, E):
    """Run a Bayes filter over a window of k observations,
    starting from uniform prior.  Returns final belief p(s=1).
    """
    b = 0.5
    for i, o in enumerate(obs_window):
        if i > 0:
            b = bayes_predict(b, epsilon)
        b = bayes_update(b, o, E)
    return b


# ================================================================
# Simulation: single trial
# ================================================================

def run_single_trial(epsilon, delta, k_values, T, seed):
    """Run one trial.  Returns dict of average rewards.

    All agents use the TRUE model parameters and compute exact
    Bayesian posteriors --- the only difference is how many
    observations each agent retains.

    - 'memory':   full Bayes filter (all observations)
    - 'markov_k': optimal k-observation agent (sliding window)
    """
    env = AliasedHMM(epsilon, delta, seed=seed)
    E = env.E
    k_max = max(k_values) if k_values else 0

    # Observation history buffer (ring buffer of size k_max)
    obs_buf = np.zeros(max(k_max, 1), dtype=int)
    buf_idx = 0

    # Full Bayes filter state
    b_full = 0.5

    # Accumulators
    names = ['memory'] + [f'markov_{k}' for k in k_values]
    reward_sums = {n: 0.0 for n in names}
    count = 0

    for t in range(T):
        obs = env.step()

        # ---- Full memory agent (Bayes filter) ----
        if t > 0:
            b_full = bayes_predict(b_full, epsilon)
        b_full = bayes_update(b_full, obs, E)
        a_mem = 1 if b_full > 0.5 else 0
        r_mem = env.reward(a_mem)

        # ---- Store observation ----
        obs_buf[buf_idx % max(k_max, 1)] = obs
        buf_idx += 1

        # ---- Markov-k agents (sliding window) ----
        r_mk = {}
        for k in k_values:
            if buf_idx >= k:
                # Extract last k observations
                window = []
                for j in range(k):
                    idx = (buf_idx - k + j) % max(k_max, 1)
                    window.append(obs_buf[idx])
                b_k = run_kstep_filter(window, epsilon, E)
                a_k = 1 if b_k > 0.5 else 0
            else:
                # Not enough observations yet — random
                a_k = int(np.random.randint(2))
            r_mk[k] = env.reward(a_k)

        # ---- Accumulate after burnin ----
        if t >= BURNIN:
            reward_sums['memory'] += r_mem
            for k in k_values:
                reward_sums[f'markov_{k}'] += r_mk[k]
            count += 1

    return {n: reward_sums[n] / count for n in names}


# ================================================================
# Sweep over epsilon grid x seeds
# ================================================================

def run_sweep(eps_grid, delta, k_values, T, n_seeds):
    """Run full parameter sweep.

    Returns dict: name -> array of shape (n_eps, n_seeds).
    """
    names = ['memory'] + [f'markov_{k}' for k in k_values]
    results = {n: np.zeros((len(eps_grid), n_seeds)) for n in names}

    total = len(eps_grid) * n_seeds
    done = 0

    for i, eps in enumerate(eps_grid):
        for s in range(n_seeds):
            seed = s * 10000 + i * 100 + 42
            trial = run_single_trial(eps, delta, k_values, T, seed)
            for n in names:
                results[n][i, s] = trial[n]
            done += 1
            if done % 10 == 0 or done == total:
                print(f'  {done}/{total} trials complete', flush=True)

    return results


# ================================================================
# Plotting
# ================================================================

def plot_markov_ceiling(results, ell_grid, k_values, path):
    """Two-panel figure: (a) R_bar vs ell, (b) Delta_R vs ell."""

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))
    fig.subplots_adjust(wspace=0.30)

    n_seeds = results['memory'].shape[1]
    z = 1.96  # 95% CI

    # ---- Color palette (matches tdome_demo.py) ----
    colors = {
        1: '#C62828',
        2: '#E65100',
        4: '#6A1B9A',
        8: '#2E7D32',
    }
    markers = {1: 's', 2: '^', 4: 'D', 8: 'v'}
    mem_color = '#1565C0'

    # ---- Panel (a): Average reward vs correlation length ----
    ax1.set_title(r'(a) Average reward $\bar{R}$ vs correlation length',
                  fontsize=11, pad=8)

    # Memory agent
    mem_mean = results['memory'].mean(axis=1)
    mem_se = results['memory'].std(axis=1) / np.sqrt(n_seeds)
    ax1.plot(ell_grid, mem_mean, 'o-', color=mem_color,
             linewidth=2.0, markersize=4, label='Memory (Bayes filter)',
             zorder=5)
    ax1.fill_between(ell_grid, mem_mean - z * mem_se,
                     mem_mean + z * mem_se,
                     color=mem_color, alpha=0.15)

    # Markov-k agents
    for k in k_values:
        mk_mean = results[f'markov_{k}'].mean(axis=1)
        mk_se = results[f'markov_{k}'].std(axis=1) / np.sqrt(n_seeds)
        ax1.plot(ell_grid, mk_mean, marker=markers[k], linestyle='-',
                 color=colors[k], linewidth=1.3, markersize=3.5,
                 label=f'Markov-{k} (window)')
        ax1.fill_between(ell_grid, mk_mean - z * mk_se,
                         mk_mean + z * mk_se,
                         color=colors[k], alpha=0.10)

    # Chance level
    ax1.axhline(0.5, color='grey', linestyle=':', linewidth=0.8,
                label='Chance (0.5)')

    ax1.set_xscale('log')
    ax1.set_xlabel(r'Correlation length $\ell = 1/\epsilon$', fontsize=10)
    ax1.set_ylabel(r'Average reward $\bar{R}$', fontsize=10)
    ax1.legend(fontsize=8, loc='center right')
    ax1.grid(True, alpha=0.15)

    # ---- Panel (b): Gap Delta_R vs correlation length ----
    ax2.set_title(r'(b) Memory advantage $\Delta\bar{R} = \bar{R}_{'
                  r'\mathrm{mem}} - \bar{R}_{\mathrm{Markov\text{-}k}}$',
                  fontsize=11, pad=8)

    for k in k_values:
        gap_all = results['memory'] - results[f'markov_{k}']
        gap_mean = gap_all.mean(axis=1)
        gap_se = gap_all.std(axis=1) / np.sqrt(n_seeds)
        ax2.plot(ell_grid, gap_mean, marker=markers[k], linestyle='-',
                 color=colors[k], linewidth=1.3, markersize=3.5,
                 label=f'$k = {k}$')
        ax2.fill_between(ell_grid, gap_mean - z * gap_se,
                         gap_mean + z * gap_se,
                         color=colors[k], alpha=0.10)

    ax2.axhline(0.0, color='grey', linestyle=':', linewidth=0.8)

    ax2.set_xscale('log')
    ax2.set_xlabel(r'Correlation length $\ell = 1/\epsilon$', fontsize=10)
    ax2.set_ylabel(r'$\Delta\bar{R}$', fontsize=10)
    ax2.legend(fontsize=8.5, loc='upper left')
    ax2.grid(True, alpha=0.15)

    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved {os.path.basename(path)}')


def save_summary_csv(results, ell_grid, k_values, path):
    """Export summary statistics to CSV."""
    n_seeds = results['memory'].shape[1]
    header = 'ell,epsilon,R_memory'
    for k in k_values:
        header += f',R_markov_{k}'
    header += '\n'

    with open(path, 'w') as f:
        f.write(header)
        for i, ell in enumerate(ell_grid):
            eps = 1.0 / ell
            row = f'{ell:.2f},{eps:.6f},{results["memory"][i].mean():.6f}'
            for k in k_values:
                row += f',{results[f"markov_{k}"][i].mean():.6f}'
            f.write(row + '\n')
    print(f'  Saved {os.path.basename(path)}')


# ================================================================
# Main
# ================================================================

if __name__ == '__main__':
    print('=' * 60)
    print('Paper I Companion Demo: Markov Ceiling')
    print('=' * 60)

    print(f'\nParameters:')
    print(f'  T = {T:,}  |  seeds = {N_SEEDS}  |  delta = {DELTA}')
    print(f'  k values = {K_VALUES}')
    print(f'  epsilon grid: {N_EPS} points in '
          f'[{EPS_GRID[0]:.4f}, {EPS_GRID[-1]:.4f}]')
    print(f'  ell grid: [{ELL_GRID[-1]:.1f}, {ELL_GRID[0]:.1f}]')

    # ---- Sweep ----
    n_agents = 1 + len(K_VALUES)
    print(f'\n[1/2] Running sweep: {N_EPS} epsilon values x '
          f'{N_SEEDS} seeds x {n_agents} agents ...')
    results = run_sweep(EPS_GRID, DELTA, K_VALUES, T, N_SEEDS)

    # Summary at longest correlation
    print(f'\n  At longest correlation (ell={ELL_GRID[0]:.0f}):')
    print(f'    Memory:   R_bar = '
          f'{results["memory"][0].mean():.4f}')
    for k in K_VALUES:
        mk = results[f'markov_{k}'][0].mean()
        gap = results['memory'][0].mean() - mk
        print(f'    Markov-{k}: R_bar = {mk:.4f}  '
              f'(gap = {gap:+.4f})')

    # Summary at shortest correlation
    print(f'\n  At shortest correlation (ell={ELL_GRID[-1]:.1f}):')
    print(f'    Memory:   R_bar = '
          f'{results["memory"][-1].mean():.4f}')
    for k in K_VALUES:
        mk = results[f'markov_{k}'][-1].mean()
        gap = results['memory'][-1].mean() - mk
        print(f'    Markov-{k}: R_bar = {mk:.4f}  '
              f'(gap = {gap:+.4f})')

    # ---- Plot ----
    print(f'\n[2/2] Generating figures ...')
    plot_markov_ceiling(results, ELL_GRID, K_VALUES,
                        os.path.join(BASE, 'fig_paper1_markov_ceiling.pdf'))

    save_summary_csv(results, ELL_GRID, K_VALUES,
                     os.path.join(BASE, 'markov_ceiling_data.csv'))

    print('\nDone.')
