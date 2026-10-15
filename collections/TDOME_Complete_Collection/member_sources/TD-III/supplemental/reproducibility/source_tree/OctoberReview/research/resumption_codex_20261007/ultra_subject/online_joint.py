"""Bounded online gain/sign/body inference under equal supplied resource caps.

One body read, one food read and the same fresh Gaussian gate packets per use.
Thirty-two known-input reference ports are supplied per 4096 uses to every
policy. Beliefs, physical gain and body persist through port-budget boundaries.
The calculation reports task risk and Bayesian food mismatch, not apparatus net.
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np

L = 4096
M = 32
N = 5.018975748824816
R = 0.8 ** 7
GS = np.array([0.90, 0.98]) * math.sqrt(1 - R * R)
VQ = 1 / (2 * N + 1)
LAM2 = math.pi ** 2 * (2 * N + 1) / (16 * N * (N + 1))
SUCCESS = (1 + math.exp(-2 * LAM2 * VQ)) / 2
GAIN_SWITCH = 3e-4
SIGN_SWITCH = 2e-4
BATH = 0.05
DELTA = 2.0 ** -44
STATES = np.array([(g, z, b) for g in range(2)
                   for z in (-1, 1) for b in (-1, 1)])
THETA = GS[STATES[:, 0]] * STATES[:, 1]
BODY = STATES[:, 2]


def entropy(p):
    p = np.clip(p, 1e-15, 1 - 1e-15)
    return -p * np.log(p) - (1 - p) * np.log1p(-p)


def transitions(action):
    out = np.empty((8, 8))
    body_flip = BATH if not action else BATH + SUCCESS * (1 - 2 * BATH)
    for i, (g, z, b) in enumerate(STATES):
        for j, (gp, zp, bp) in enumerate(STATES):
            out[i, j] = ((GAIN_SWITCH if g != gp else 1 - GAIN_SWITCH)
                         * (SIGN_SWITCH if z != zp else 1 - SIGN_SWITCH)
                         * (body_flip if b != bp else 1 - body_flip))
    return out


T = np.stack([transitions(False), transitions(True)])


def quantize(p, delta=DELTA):
    logratio = np.log(p[:, :7]) - np.log(p[:, 7:8])
    logratio = np.round(logratio / delta) * delta
    logs = np.column_stack((logratio, np.zeros(len(p))))
    logs -= logs.max(axis=1, keepdims=True)
    q = np.exp(logs)
    return q / q.sum(axis=1, keepdims=True)


def certificate():
    # Projective diameter: max log(T_ij T_kl / T_il T_kj).
    diameters = []
    for matrix in T:
        logt = np.log(matrix)
        diameter = max(float(np.ptp(logt[i] - logt[j]))
                       for i in range(8) for j in range(8))
        diameters.append(diameter)
    diameter = max(diameters)
    contraction = math.tanh(diameter / 4)
    tmin = float(T.min())
    lo = (1 - GS.max()) / 2
    hi = (1 + GS.max()) / 2
    # One body and at most one reference observation before quantization.
    lower = tmin * (lo / hi) ** 2
    radius = -math.log(lower)
    coordinate_bits = math.ceil(math.log2(2 * radius / DELTA + 3))
    hilbert_floor = DELTA / (1 - contraction)
    tv_floor = math.tanh(hilbert_floor / 4)
    rng = np.random.default_rng(57213)
    max_violation = -math.inf
    max_quantization_error = 0.0
    for _ in range(3000):
        p, q = rng.dirichlet(np.ones(8), size=2)
        distance = np.ptp(np.log(p) - np.log(q))
        matrix = T[rng.integers(2)]
        updated_p, updated_q = p @ matrix, q @ matrix
        image_distance = np.ptp(np.log(updated_p) - np.log(updated_q))
        max_violation = max(max_violation, image_distance - contraction * distance)
        qp = quantize(updated_p[None])[0]
        max_quantization_error = max(max_quantization_error,
                                      float(np.ptp(np.log(qp) - np.log(updated_p))))
    return {
        'transition_row_residual': float(np.max(abs(T.sum(axis=2) - 1))),
        'minimum_transition': tmin, 'projective_diameter': diameter,
        'contraction': contraction, 'delta': DELTA,
        'posterior_lower_before_quantization': lower,
        'log_coordinate_radius': radius, 'posterior_register_bits': 7 * coordinate_bits,
        'budget_register_bits': 6, 'clock_states': L,
        'asymptotic_hilbert_bound': hilbert_floor,
        'asymptotic_total_variation_bound': tv_floor,
        'body_one_step_excess_risk_bound': 2 * tv_floor,
        'random_contraction_max_violation': max_violation,
        'random_quantization_max_error': max_quantization_error,
        'gate_success': SUCCESS, 'gate_lambda2': LAM2, 'gate_occupation': N,
        'supplied_record_purity_per_use': (4 * 7 + M * 7 / L) * math.log(2),
        'supplied_reference_preparation_and_separate_erasure_per_use':
            2 * M * math.log(2) / L,
        'processor_reset_cost': 'not implemented or inferred from task risk',
    }


def run(policy, threshold, steps, paths, seed):
    rng = np.random.default_rng(seed)
    latent = rng.integers(8, size=paths)
    belief = np.full((paths, 8), 1 / 8)
    exact_same_history = belief.copy()
    used = np.zeros(paths, dtype=int)
    burn = 2 * L
    per_path = np.zeros((paths, 5))
    max_budget = 0
    max_history_tv = 0.0
    normalization = 0.0
    for time in range(steps):
        if time % L == 0:
            used[:] = 0
        real_theta = THETA[latent]
        real_body = BODY[latent]
        y = np.where(rng.random(paths) < (1 + real_theta * real_body) / 2, 1, -1)
        likelihood = (1 + y[:, None] * THETA * BODY) / 2
        belief *= likelihood
        belief /= belief.sum(axis=1, keepdims=True)
        exact_same_history *= likelihood
        exact_same_history /= exact_same_history.sum(axis=1, keepdims=True)
        sign_probability = belief[:, STATES[:, 1] == 1].sum(axis=1)
        if policy == 'periodic':
            request = np.full(paths, time % (L // M) == 0)
        elif policy == 'adaptive':
            request = np.minimum(sign_probability, 1 - sign_probability) > threshold
        else:
            request = np.zeros(paths, dtype=bool)
        request &= used < M
        reference = np.where(rng.random(paths) < (1 + real_theta) / 2, 1, -1)
        like_ref = np.where(request[:, None], (1 + reference[:, None] * THETA) / 2, 1.0)
        belief *= like_ref
        belief /= belief.sum(axis=1, keepdims=True)
        exact_same_history *= like_ref
        exact_same_history /= exact_same_history.sum(axis=1, keepdims=True)
        used += request
        max_budget = max(max_budget, int(used.max()))
        belief = quantize(belief)
        theta_bar = belief @ THETA
        gain_estimate = belief @ GS[STATES[:, 0]]
        command = belief @ BODY < 0
        theta_loss = entropy((1 + SUCCESS * abs(theta_bar)) / 2) - belief @ entropy((1 + SUCCESS * GS[STATES[:, 0]]) / 2)
        gain_error = (gain_estimate - GS[STATES[latent, 0]]) ** 2
        normalization = max(normalization, float(np.max(abs(belief.sum(axis=1) - 1))))
        max_history_tv = max(max_history_tv, float(np.max(abs(belief - exact_same_history).sum(axis=1) / 2)))
        # Action, then bath, gain switch and sign switch: exactly T[command].
        action_flip = command & (rng.random(paths) < SUCCESS)
        next_body = real_body * np.where(action_flip, -1, 1) * np.where(rng.random(paths) < BATH, -1, 1)
        gain_index = STATES[latent, 0] ^ (rng.random(paths) < GAIN_SWITCH)
        next_sign = STATES[latent, 1] * np.where(rng.random(paths) < SIGN_SWITCH, -1, 1)
        latent = 4 * gain_index + 2 * (next_sign == 1) + (next_body == 1)
        if time >= burn:
            per_path[:, 0] += next_body == -1
            per_path[:, 1] += request
            per_path[:, 2] += theta_loss
            per_path[:, 3] += gain_error
            per_path[:, 4] += command
        belief = np.where(command[:, None], belief @ T[1], belief @ T[0])
        exact_same_history = np.where(command[:, None], exact_same_history @ T[1], exact_same_history @ T[0])
    per_path /= steps - burn
    means = per_path.mean(axis=0)
    ses = per_path.std(axis=0, ddof=1) / math.sqrt(paths)
    keys = ['body_loss', 'reference_rate', 'food_bayes_mismatch', 'gain_mse', 'body_switch_rate']
    return {'policy': policy, 'threshold': threshold, 'steps': steps, 'paths': paths,
            'seed': seed, 'mean': dict(zip(keys, map(float, means))),
            'se': dict(zip(keys, map(float, ses))),
            'max_references_per_4096': max_budget,
            'posterior_normalization_residual': normalization,
            'max_quantized_vs_exact_same_history_tv': max_history_tv}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--steps', type=int, default=32768)
    parser.add_argument('--paths', type=int, default=128)
    parser.add_argument('--seed', type=int, default=7100726)
    parser.add_argument('--chosen-only', action='store_true')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    results = [run('periodic', 0, args.steps, args.paths, args.seed)]
    results += [run('adaptive', threshold, args.steps, args.paths, args.seed)
                for threshold in ((.04,) if args.chosen_only else (.01, .02, .04))]
    out = {'certificate': certificate(), 'runs': results,
           'meaning': 'bounded continuing joint filter and task risk; no apparatus-net claim'}
    encoded = json.dumps(out, indent=2)
    if args.output:
        args.output.write_text(encoded + '\n')
    print(encoded)


if __name__ == '__main__':
    main()
