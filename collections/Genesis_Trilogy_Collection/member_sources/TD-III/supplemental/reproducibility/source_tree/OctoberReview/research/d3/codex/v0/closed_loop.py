"""D3 finite bit-bank feedback, independent of Claude's implementations.

Stdout JSON only. Count compression is exact for this exchangeable classical
bank; it does not give the observer access to the hidden count. All policies
receive only their own instrument counts and initial calibration counts.
"""
import json
import numpy as np
from scipy.stats import binom

CONFIG = {
    'seed': 610316, 'trajectories': 160, 'epochs': 1200, 'burn_in': 200,
    'bank_bits': 4096, 'identity_weight': 0.65, 'control_weight': 0.25,
    'environment_weight': 0.10, 'feedback_gain': 0.8,
    'reads_per_epoch': 512, 'initial_reference_reads_each': 2048,
    'adaptive_signal_reads': 256, 'adaptive_reference_reads_each': 128,
    'ewma_alpha': 0.25, 'quantization_intervals': 1024, 'gain_floor': 0.30,
}


def quantize(z, cfg):
    n = cfg['quantization_intervals']
    return -1 + 2 * np.rint((np.clip(z, -1, 1) + 1) * n / 2) / n


def sensor(k, regime):
    if regime == 'static':
        return 0.55, 0.0
    if regime == 'fast_drift':
        k *= 8
    return (0.55 + 0.17*np.sin(2*np.pi*k/420) + 0.03*np.sin(2*np.pi*k/67),
            0.14*np.sin(2*np.pi*k/300) + 0.025*np.sin(2*np.pi*k/43))


def bath(k):
    return 0.60 + 0.25*np.sin(2*np.pi*k/177)


def calibration_counts(rng, g, b, n, size):
    plus = rng.binomial(n, (1 + g + b)/2, size=size)
    minus = rng.binomial(n, (1 - g + b)/2, size=size)
    return plus, minus


def update_estimates(gh, bh, plus, minus, n, alpha, cfg):
    # This function sees no hidden world, true sensor, or bath parameter.
    raw_g = (plus - minus)/n
    raw_b = (plus + minus)/n - 1
    return (quantize((1-alpha)*gh + alpha*raw_g, cfg),
            quantize((1-alpha)*bh + alpha*raw_b, cfg))


def decide(signal_count, n, gh, bh, cfg):
    # Only counts, stored estimates and declared constants enter the policy.
    observed = 2*signal_count/n - 1
    estimate = quantize((observed-bh)/np.maximum(gh, cfg['gain_floor']), cfg)
    return estimate, -cfg['feedback_gain']*estimate


def summarize(values):
    values = np.asarray(values)
    return {'mean': float(values.mean()),
            'trajectory_mean_standard_error': float(values.std(ddof=1)/np.sqrt(len(values)))}


def simulate(regime, cfg):
    rng = np.random.default_rng(cfg['seed'] + {'drift':0, 'static':1, 'fast_drift':2}[regime])
    names = ['fixed_calibration', 'blockwise_calibration', 'persistent_calibration']
    m, reps = cfg['bank_bits'], cfg['trajectories']
    hidden = np.full((len(names), reps), int(0.60*m), dtype=int)
    g0, b0 = sensor(0, regime)
    p0, m0 = calibration_counts(rng, g0, b0, cfg['initial_reference_reads_each'], reps)
    ig, ib = update_estimates(np.zeros(reps), np.zeros(reps), p0, m0,
                             cfg['initial_reference_reads_each'], 1.0, cfg)
    gh, bh = np.tile(ig, (3, 1)), np.tile(ib, (3, 1))
    metric_names = ['world_loss', 'estimation_loss', 'control_square', 'gain_error_square', 'bias_error_square']
    accumulated = {name: np.zeros((3, reps)) for name in metric_names}
    a, c, r = (cfg[key] for key in ('identity_weight', 'control_weight', 'environment_weight'))
    max_mean_identity_residual = 0.0
    for k in range(cfg['epochs']):
        g, b = sensor(k, regime)
        assert abs(g) + abs(b) < 1 and g >= cfg['gain_floor']
        d = bath(k)
        for j, name in enumerate(names):
            # Environment/experiment layer only: hidden count is not sent to policy.
            x = 2*hidden[j]/m - 1
            if name == 'fixed_calibration':
                ns = cfg['reads_per_epoch']
            else:
                ns = cfg['adaptive_signal_reads']
                nr = cfg['adaptive_reference_reads_each']
                plus, minus = calibration_counts(rng, g, b, nr, reps)
                alpha = 1.0 if name == 'blockwise_calibration' else cfg['ewma_alpha']
                gh[j], bh[j] = update_estimates(gh[j], bh[j], plus, minus, nr, alpha, cfg)
            signal = rng.binomial(ns, (1+g*x+b)/2)
            xhat, u = decide(signal, ns, gh[j], bh[j], cfg)
            assert np.all(np.abs(u) <= cfg['feedback_gain'])
            if k >= cfg['burn_in']:
                for label, val in (
                    ('world_loss', x*x/4), ('estimation_loss', (xhat-x)**2/4),
                    ('control_square', u*u), ('gain_error_square', (gh[j]-g)**2),
                    ('bias_error_square', (bh[j]-b)**2),
                ):
                    accumulated[label][j] += val
            p_plus = (1+a+c*u+r*d)/2
            p_minus = (1-a+c*u+r*d)/2
            assert np.all((p_plus >= 0) & (p_plus <= 1))
            assert np.all((p_minus >= 0) & (p_minus <= 1))
            expectation = 2*(hidden[j]*p_plus+(m-hidden[j])*p_minus)/m - 1
            max_mean_identity_residual = max(max_mean_identity_residual,
                float(np.max(np.abs(expectation-(a*x+c*u+r*d)))))
            old = hidden[j].copy()
            hidden[j] = rng.binomial(old, p_plus) + rng.binomial(m-old, p_minus)
    count = cfg['epochs']-cfg['burn_in']
    results = {}
    for j, name in enumerate(names):
        results[name] = {key: summarize(val[j]/count) for key, val in accumulated.items()}
        results[name]['read_allocation_signal_plus_minus'] = ([512, 0, 0] if j == 0 else [256, 128, 128])
    # Replicate-level differences, not iid estimates across correlated epochs.
    results['persistent_minus_blockwise_world_loss'] = summarize(
        (accumulated['world_loss'][2]-accumulated['world_loss'][1])/count)
    results['persistent_minus_fixed_world_loss'] = summarize(
        (accumulated['world_loss'][2]-accumulated['world_loss'][0])/count)
    return {'regime': regime, 'policies': results,
            'conditional_mean_identity_max_residual': max_mean_identity_residual,
            'hidden_bank_count_range_at_end': [int(hidden.min()), int(hidden.max())]}


def exact_old_static_risk():
    ns, nr, x, g, b, floor = 400, 200, 0.4, 0.5, 0.1, 0.125
    cp = np.arange(nr+1)[:, None]
    cm = np.arange(nr+1)[None, :]
    gh, bh = (cp-cm)/nr, (cp+cm)/nr-1
    weights = binom.pmf(cp, nr, (1+g+b)/2)*binom.pmf(cm, nr, (1-g+b)/2)
    mean, second = 0., 0.
    for s in range(ns+1):
        estimate = np.clip((2*s/ns-1-bh)/np.maximum(gh, floor), -1, 1)
        loss = (estimate-x)**2/4
        ps = binom.pmf(s, ns, (1+g*x+b)/2)
        mean += ps*np.sum(weights*loss)
        second += ps*np.sum(weights*loss*loss)
    return {'parameters': {'x':x,'g':g,'b':b,'reads':[ns,nr,nr],'gain_floor':floor},
            'exact_risk': float(mean), 'monte_carlo_standard_error_for_50000':float(np.sqrt((second-mean*mean)/50000))}


def analytic_bound(cfg):
    alpha = cfg['ewma_alpha']
    beta = 1-alpha
    nr, ns = cfg['adaptive_reference_reads_each'], cfg['adaptive_signal_reads']
    delta = 2/cfg['quantization_intervals']
    ng = 0.17*2*np.pi/420 + 0.03*2*np.pi/67
    nb = 0.14*2*np.pi/300 + 0.025*2*np.pi/43
    noise = np.sqrt(alpha/(2*nr*(2-alpha)))
    bg = beta*ng/alpha + noise + delta/(2*alpha)
    bb = beta*nb/alpha + noise + delta/(2*alpha)
    e = min(2., (1/np.sqrt(ns)+bg+bb)/cfg['gain_floor']+delta/2)
    a, c, r = (cfg[key] for key in ('identity_weight','control_weight','environment_weight'))
    k = c*cfg['feedback_gain']
    q = abs(a-k)
    world_rms = min(1., (k*e+r*0.85+1/np.sqrt(cfg['bank_bits']))/(1-q))
    forcing = k*e+r*0.85
    tighter = min(1., (q*forcing+np.sqrt(forcing**2+(1-q*q)/cfg['bank_bits']))/(1-q*q))
    return {'applicable_regimes':['drift','static'],
            'sensor_per_epoch_drift_bounds':[float(ng),float(nb)],
            'asymptotic_calibration_rms_bounds':[float(bg),float(bb)],
            'asymptotic_estimation_rms_bound':float(e), 'closed_loop_contraction':q,
            'asymptotic_world_normalized_risk_bound':float(world_rms**2/4),
            'improved_asymptotic_world_risk_bound':float(tighter**2/4),
            'scope':'Expected limsup bound for normal drift or static, NOT fast_drift; not simultaneous all-time confidence; transient terms are in report.'}


report = {
    'configuration':CONFIG,
    'static_reconciliation':exact_old_static_risk(),
    'analytic_guarantee':analytic_bound(CONFIG),
    'experiments':[simulate(regime, CONFIG) for regime in ('drift','static','fast_drift')],
    'resources':{
        'observer_calibration_grid_values_each':1025,
        'observer_calibration_bits_each':11,
        'actuator_grid_values':1025, 'actuator_index_bits':11,
        'classical_counter_bits_peak_sum':26,
        'reference_preparations_per_epoch_adaptive':256,
        'signal_random_index_choices_per_epoch_adaptive':256,
        'physical_world_bits':4096,
        'per_epoch_world_channel_applications':4096,
        'expected_world_reset_branch_uses':(1-CONFIG['identity_weight'])*CONFIG['bank_bits'],
        'maximum_world_reset_branch_uses':4096,
        'note':'Counters, arithmetic workspace, random-index generator and clock additionally charged as described in report. Not finite total lifetime cost; no heat/work value inferred.'},
    'scope':'One diagonal quantum/classical controlled platform; stationary and drifting calibration comparisons, not workspace formation, consciousness, or a universal self-model necessity theorem.'
}
print(json.dumps(report, indent=2))
