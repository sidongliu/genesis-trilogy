"""Monte Carlo check of the Gaussian residual Fisher model."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
FIGURE_PATH = ROOT / "figures" / "fig_fisher_residual.pdf"
DATA_PATH = ROOT / "data" / "fisher_residual_data.csv"

SEED = 20260715
A = 1.2
SIGMA_E = 0.35
EPSILON = 0.04
SAMPLE_RATE = 50.0
J_MIN = 20.0
N_REPLICATES = 4000
N_TIMES = 320


def cumulative_trapezoid(values: np.ndarray, times: np.ndarray) -> np.ndarray:
    increments = 0.5 * (values[1:] + values[:-1]) * np.diff(times)
    return np.concatenate(([0.0], np.cumsum(increments)))


def main() -> None:
    rng = np.random.default_rng(SEED)
    t_vis = np.pi / (4.0 * EPSILON)
    t_det = (3.0 * SIGMA_E**2 * J_MIN / (4.0 * A**2 * SAMPLE_RATE * EPSILON**2)) ** (1.0 / 3.0)

    times = np.linspace(0.0, t_vis, N_TIMES)
    theta = EPSILON * times
    mean = A * theta**2
    residuals = rng.normal(mean[:, None], SIGMA_E, size=(N_TIMES, N_REPLICATES))
    scores = (residuals - mean[:, None]) * (2.0 * A * theta[:, None]) / SIGMA_E**2

    fisher_empirical = np.mean(scores**2, axis=1)
    fisher_analytic = 4.0 * A**2 * theta**2 / SIGMA_E**2
    accumulated_empirical = SAMPLE_RATE * cumulative_trapezoid(fisher_empirical, times)
    accumulated_analytic = 4.0 * A**2 * SAMPLE_RATE * EPSILON**2 * times**3 / (3.0 * SIGMA_E**2)

    crossing = np.flatnonzero(accumulated_empirical >= J_MIN)
    t_det_empirical = times[crossing[0]] if crossing.size else np.nan

    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(
        DATA_PATH,
        np.column_stack(
            (
                times,
                theta,
                fisher_analytic,
                fisher_empirical,
                accumulated_analytic,
                accumulated_empirical,
            )
        ),
        delimiter=",",
        header="time,theta,fisher_analytic,fisher_empirical,accumulated_analytic,accumulated_empirical",
        comments="",
    )

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.1))
    axes[0].plot(times, fisher_analytic, color="#2457a6", linewidth=2.2, label="analytic")
    axes[0].scatter(times[::8], fisher_empirical[::8], s=13, color="#d1495b", alpha=0.8, label="Monte Carlo score")
    axes[0].set_xlabel("drift time $t$")
    axes[0].set_ylabel("per-sample $\\mathcal{I}_1(\\theta(t))$")
    axes[0].set_title("Quadratic local Fisher signal")
    axes[0].legend(frameon=False)

    axes[1].plot(times, accumulated_analytic, color="#2457a6", linewidth=2.2, label="analytic $J_T$")
    axes[1].plot(times, accumulated_empirical, color="#d1495b", linewidth=1.4, alpha=0.85, label="Monte Carlo $J_T$")
    axes[1].axhline(J_MIN, color="#4c956c", linestyle="--", linewidth=1.4, label="$J_{\\min}$")
    axes[1].axvline(t_det, color="#4c956c", linestyle=":", linewidth=1.5, label="$T_{\\mathrm{det}}$")
    axes[1].axvline(t_vis, color="#6b6b6b", linestyle="--", linewidth=1.3, label="$t_{\\mathrm{vis}}$")
    axes[1].set_xlabel("window length $T$")
    axes[1].set_ylabel("accumulated local information $J_T$")
    axes[1].set_title("Threshold before visibility crossover")
    axes[1].set_yscale("log")
    axes[1].set_ylim(0.5, accumulated_analytic[-1] * 1.2)
    axes[1].legend(frameon=False, fontsize=8)

    for axis in axes:
        axis.grid(alpha=0.2)
    fig.tight_layout()
    FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE_PATH, bbox_inches="tight")
    print(f"analytic T_det={t_det:.4f}, empirical T_det={t_det_empirical:.4f}, t_vis={t_vis:.4f}")


if __name__ == "__main__":
    main()
