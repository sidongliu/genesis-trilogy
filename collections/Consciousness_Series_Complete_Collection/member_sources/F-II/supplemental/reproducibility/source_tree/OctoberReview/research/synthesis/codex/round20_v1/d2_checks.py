"""Independent D2 implementation after reading Claude v3; not a blind audit.

No imports from the source simulation. Four counterfactual policies share one
exogenous world. Healthy physical records are reused across counterfactuals;
faulted and healthy instrument records are independently drawn inside a fault
window. This is a valid coupling that differs from the source implementation's RNG coupling. Each of the
R trials has its own reference outcomes. Outputs JSON to stdout, writes nothing.
Only selected rules / faults, three new seeds; no exhaustive parameter sweep.
"""

import json
from dataclasses import dataclass

import numpy as np


def grid(value):
    return np.rint(512.0 * np.clip(value, -1.0, 1.0)) / 512.0


@dataclass(frozen=True)
class Case:
    name: str
    references: tuple
    fault: str
    weighted: bool = False
    recipient: int = 0
    communicating: bool = True


def moments(values):
    return dict(mean=float(np.mean(values)),
                se=float(np.std(values, ddof=1) / np.sqrt(len(values))))


def experiment(case, seed, trials=240, epochs=2600):
    # mode 0: no fault, 1: fault, 2: fault + isolation, 3: no fault + isolation.
    children = np.random.SeedSequence(seed).spawn(7)
    world_rng = np.random.default_rng(children[0])
    onset_rng = np.random.default_rng(children[1])
    startup_rng = np.random.default_rng(children[2])
    read_rng = [[np.random.default_rng(children[3 + 2*i + j])
                 for j in range(2)] for i in range(2)]
    start = onset_rng.integers(800, 2400, size=trials)
    assert epochs >= int(np.max(start)) + 200
    plus = np.full(trials, 2048, dtype=np.int64)
    event = 2 * world_rng.integers(0, 2, size=trials) - 1
    gain_est = np.empty((4, 2, trials))
    bias_est = np.empty_like(gain_est)
    window_correct = np.zeros((4, trials))
    window_count = np.zeros(trials, dtype=int)
    minimum_reference_unique = 10**9
    phase_g = (0.0, 1.7)
    phase_b = (0.4, 2.9)
    sender = 1 - case.recipient

    def instruments(k):
        gain = np.array([.55 + .17*np.sin(2*np.pi*k/420 + phase_g[i])
                         + .03*np.sin(2*np.pi*k/67 + phase_g[i])
                         for i in range(2)])
        bias = np.array([.10*np.sin(2*np.pi*k/250)
                         + .03*np.sin(2*np.pi*k/53)
                         + .10*np.sin(2*np.pi*k/300 + phase_b[i])
                         + .025*np.sin(2*np.pi*k/43 + phase_b[i])
                         for i in range(2)])
        return gain, bias

    gain, bias = instruments(0)
    for i in range(2):
        p = startup_rng.binomial(2048, (1 + gain[i] + bias[i])/2, size=trials)
        m = startup_rng.binomial(2048, (1 - gain[i] + bias[i])/2, size=trials)
        gain_est[:, i] = grid((p - m)/2048)
        bias_est[:, i] = grid((p + m)/2048 - 1)

    for k in range(epochs):
        event *= np.where(world_rng.random(trials) < .01, -1, 1)
        target = .6 * event
        x = (plus - 2048) / 2048.0
        active = (k >= start) & (k < start + 100)
        gain, bias = instruments(k)
        observed = np.empty((4, 2, trials))
        for i in range(2):
            physical_g = np.full((2, trials), gain[i])
            physical_b = np.full((2, trials), bias[i])
            if i == sender:
                if case.fault == 'bias': physical_b[1, active] += .30
                elif case.fault == 'zerogain': physical_g[1, active] = 0
                elif case.fault == 'lowgain': physical_g[1, active] = .05
                else: raise ValueError(case.fault)
            physical_b = np.clip(physical_b, -.98+abs(physical_g), .98-abs(physical_g))
            assert np.max(abs(physical_b)+abs(physical_g)) <= .9800000001
            nr = case.references[i]
            ns = 512 - 2*nr
            raw = np.empty((2, trials))
            raw_gain = np.empty_like(raw)
            raw_bias = np.empty_like(raw)
            for fault_flag in range(2):
                rng = read_rng[i][fault_flag]
                count = rng.binomial(ns, (1 + physical_g[fault_flag]*x
                                          + physical_b[fault_flag])/2)
                raw[fault_flag] = (2*count - ns)/ns
                if nr:
                    p = rng.binomial(nr, (1 + physical_g[fault_flag]
                                          + physical_b[fault_flag])/2)
                    m = rng.binomial(nr, (1 - physical_g[fault_flag]
                                          + physical_b[fault_flag])/2)
                    assert p.shape == m.shape == (trials,)
                    minimum_reference_unique = min(minimum_reference_unique,
                                                   len(np.unique(p-m)))
                    raw_gain[fault_flag] = (p-m)/nr
                    raw_bias[fault_flag] = (p+m-nr)/nr
            use_fault = active if i == sender else np.zeros(trials, dtype=bool)
            for mode in range(4):
                mask = use_fault if mode in (1, 2) else np.zeros(trials, dtype=bool)
                observed[mode, i] = np.where(mask, raw[1], raw[0])
                if nr:
                    rg = np.where(mask, raw_gain[1], raw_gain[0])
                    rb = np.where(mask, raw_bias[1], raw_bias[0])
                    gain_est[mode, i] = grid(.75*gain_est[mode, i] + .25*rg)
                    bias_est[mode, i] = grid(.75*bias_est[mode, i] + .25*rb)
        decoded = grid((observed-bias_est) / np.maximum(.30, gain_est))
        old_bias = bias_est.copy()
        if case.communicating:
            for i in range(2):
                partner = 1-i
                residual = observed[:, i] - gain_est[:, i]*decoded[:, partner] - old_bias[:, i]
                trust = np.clip(gain_est[:, partner]/.4, 0, 1)**2 if case.weighted else 1
                updated = grid(old_bias[:, i] + .25*trust*residual)
                if i == case.recipient:
                    updated[2] = np.where(active, old_bias[2, i], updated[2])
                    updated[3] = np.where(active, old_bias[3, i], updated[3])
                bias_est[:, i] = updated
        prediction = np.where(decoded[:, case.recipient] >= 0, 1, -1)
        window_correct += (prediction == event) * active
        window_count += active
        plus = (world_rng.binomial(plus, (1+.65+.10*target)/2)
                + world_rng.binomial(4096-plus, (1-.65+.10*target)/2))

    assert np.all(window_count == 100)
    risk = window_correct / 100
    assert np.array_equal(risk[2], risk[3]), 'isolated receiver must be invariant to partner fault'
    total = risk[0] - risk[1]
    support = risk[0] - risk[2]
    harm = risk[2] - risk[1]
    assert np.max(abs(total-support-harm)) < 1e-15
    if not case.communicating:
        assert np.array_equal(risk[0], risk[1])
    return dict(case=case.name, seed=seed, trials=trials,
                correct_no_fault=moments(risk[0]), correct_fault=moments(risk[1]),
                correct_isolated=moments(risk[2]), contagion=moments(total),
                support=moments(support), coupling_harm=moments(harm),
                isolation_exact=True, decomposition_exact=True,
                reference_sample_shape=[trials],
                minimum_reference_count_differences=minimum_reference_unique), total, support, harm


def main():
    cases = [Case('mixed16_zero', (16,16), 'zerogain'),
             Case('weighted16_zero', (16,16), 'zerogain', weighted=True),
             Case('mixed4_bias', (4,4), 'bias'),
             Case('receiver_anchor_zero', (4,0), 'zerogain'),
             Case('sender_anchor_zero', (0,4), 'zerogain'),
             Case('uncoupled64_zero', (64,64), 'zerogain', communicating=False)]
    results, pooled = [], {}
    for case in cases:
        vectors = []
        for seed in (2026100401, 2026100417, 2026100479):
            item, total, support, harm = experiment(case, seed)
            results.append(item)
            vectors.append((total,support,harm))
        pooled[case.name] = {label: moments(np.concatenate([v[j] for v in vectors]))
                             for j,label in enumerate(('contagion','support','coupling_harm'))}
    print(json.dumps(dict(scope='Independent implementation after reading v3, selected cases only',
                          seeds=[2026100401,2026100417,2026100479], epochs=2600,
                          trials_per_seed=240, quantization_values=1025,
                          bits_per_separately_encoded_scalar=11,
                          no_opponent_code_imported=True, results=results, pooled=pooled),
                     ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
