#!/usr/bin/env python3
"""Recalculate Foundation II's archived numeric tables from three supplied JSON inputs.

Extracted from the numeric portion of make_f2_section.py. No manuscript text,
TeX table, or PDF is generated. Only Python's standard library is required.
Run in an isolated copy, or give --output a new path to preserve the archive.
"""
from pathlib import Path
import argparse
import json

HERE = Path(__file__).resolve().parent

R = json.load(open(HERE / '../../d1d2/claude_v3/d2_v3_output.json'))
FX = json.load(open(HERE / '../../d1d2/claude_v3/d2_v3_output_fixedwindow.json'))
CX = json.load(open(HERE / '../../synthesis/codex/round20_v1/d2_output.json'))
B, C, D, V1, V1C = (R['baseline'], R['contagion'], R['decomposition'], R['v1scan'], R['v1common'])
used = {}

def dc(cfg, kd, who):
    return C[f'{cfg}|{kd}|{who}']['dC_win']

def se(cfg, kd, who):
    return C[f'{cfg}|{kd}|{who}']['se_win']

def har(cfg, kd, who):
    return D[f'{cfg}|{kd}|{who}']['harm']

def sup(cfg, kd, who):
    return D[f'{cfg}|{kd}|{who}']['support']

def pca(v):
    return 1 - v['E'] / v['A12']
T1 = [('Initial references only (no further calibration)', 'ref|0'), ('References, 4 per sign per epoch, no coupling', 'ref|4'), ('References, 16 per sign', 'ref|16'), ('References, 64 per sign', 'ref|64'), ('References, 128 per sign', 'ref|128'), ('Agreement only, no references', 'mutual|0'), ('References (16) and agreement', 'mixed|16')]
used['table1'] = {k: dict(A=V1[k]['A12'], C=V1[k]['C1'], E=V1[k]['E'], P_correct_given_agree=pca(V1[k])) for _, k in T1}
ident_resid = max((abs(v['C1'] + v['C2'] - (1 + v['A12'] - 2 * v['E'])) for grp in (R['baseline'], V1, V1C) for v in grp.values() if all((k in v for k in ('A12', 'C1', 'C2', 'E')))))
used['identity_max_residual_over_saved_rows'] = ident_resid
used['table2'] = {f'{k}|{cs}': V1C[f'{k}|{cs}'] for k in ('mutual', 'ref64') for cs in ('0.0', '1.0', '2.0')}
rows = [('Agreement only, no references', 'bias (mutual, no refs)'), ('References (4), agreement, both observers', 'mixed (4,4)'), ('References (16), agreement, both', 'mixed (16,16)'), ('References (64), agreement, both', 'mixed (64,64)'), ('References (16), weighted agreement, both', 'mixedw (16,16)'), ('References (64), weighted agreement, both', 'mixedw (64,64)'), ('References (4) at observer 1 only, agreement', 'mixed (4,0)'), ('References (4) at observer 2 only, agreement', 'mixed (0,4)'), ('Leader (references 64) and follower', 'leader (n_r=64)'), ('References (64), no coupling', 'ref n_r=64 (no comm.)')]
used['table3'] = {k: {kd: [dc(k, kd, 1), dc(k, kd, 0), se(k, kd, 1), se(k, kd, 0)] for kd in ('bias', 'zerogain')} for _, k in rows}
refrows = [k for _, k in rows if k not in ('bias (mutual, no refs)',) and (not k.startswith('leader')) and (not k.startswith('ref n_r'))]
max_bias_ref = max((abs(dc(k, 'bias', w)) for k in refrows for w in (0, 1)))
leader_bias = dc('leader (n_r=64)', 'bias', 0)
fixed_bias = FX['contagion']['bias (mutual, no refs)|bias|1']['dC_win']
maxse = max((se(k, kd, w) for _, k in rows for kd in ('bias', 'zerogain') for w in (0, 1)))
cp = CX['pooled']
harm_eq, harm_w = (cp['mixed16_zero']['coupling_harm']['mean'], cp['weighted16_zero']['coupling_harm']['mean'])
tot_eq, tot_w = (cp['mixed16_zero']['contagion']['mean'], cp['weighted16_zero']['contagion']['mean'])
reduction = 1 - harm_w / harm_eq
used.update(max_bias_transfer_rows_with_refs=max_bias_ref, leader_bias=leader_bias, fixed_window_bias=fixed_bias, table3_max_se=maxse, second_implementation=dict(seeds=CX['seeds'], epochs=CX['epochs'], trials_per_seed=CX['trials_per_seed'], harm_equal=harm_eq, harm_weighted=harm_w, total_equal=tot_eq, total_weighted=tot_w, harm_reduction=reduction))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=HERE / 'f2_numbers.json')
    args = parser.parse_args()
    args.output.write_text(json.dumps(used, indent=1) + '\n')
    print('Wrote numeric results to', args.output)
