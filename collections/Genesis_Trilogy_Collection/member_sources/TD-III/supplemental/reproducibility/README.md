# TD-III reproducibility bundle

Detection, drift prices, symmetry and sign inversion, body moments, block margins, continuing filtering, body repair, two Model-A implementations and Model-B authority. These experiments retain different assumptions; their outputs are not one apparatus certificate.

SOURCE_MAP.json lists every original and distributed SHA-256, with the paper owner and any final-newline-only normalization. The source_tree contains the actual inputs, not links to a private workspace. Original contributor/path names are retained as provenance, not statements of independent verification.

Use Python with NumPy, SciPy and mpmath; tested versions are recorded in SOURCE_MAP.json. Run a check only in an isolated copy:

    python3 run_check.py SCRIPT.py --output-dir /path/to/a/new/run

The output directory must not exist. The wrapper copies the input tree before running, retaining relative imports and data paths. It saves stdout, stderr and RUN.json. Do not run scripts in the frozen source_tree: some write beside themselves.

Large historical Monte Carlo runs can be expensive. Packaging did not rerun every experiment. Archived outputs identify the evidence underlying the paper; their hashes do not certify the calculations. Individual seeds, horizons and resource assumptions remain in each script and output. Do not replace them with an unlabelled shortened run.

## Manuscript result to check to archived output

Labels below remain valid when section numbering changes. A support entry identifies the stated finite computation, not a proof of every general claim. SOURCE_MAP.json gives full paths, exact output keys and both provenance hashes. No script or archived output was changed for this mapping.

| Manuscript label(s) | Script | Archived output(s) | Scope |
|---|---|---|---|
| `prop:even`, `cor:three_records`, `prop:detect`, `tab:detect` | `check_detection.py` | `check_detection_output.json` | Even/odd Gaussian information and entropy formulas, necessary horizons and fixed-seed likelihood-ratio diagnostics. Does not implement the residual-variance-test row. |
| `cor:before_failure`, `tab:detect` | `check_variance_test.py` | `check_variance_test_output.json` | The residual-variance-test row only: Brownian paths, chi-square null threshold and grid-based type-II Monte Carlo. Necessary times are supplied constants from check_detection, not independently recomputed here. |
| `thm:driftlaw`, `cor:driftlaw`, `cor:c3`, `cor:thermo`, `tab:price`, `rem:driftlaw_scope` | `check_drift_law.py` | `check_drift_law_output.json` | Gaussian information recursion, cost brackets and price table; finite schedule/value-iteration diagnostics, stationary Kalman and a clipped-binomial comparison. The clipped model is a separate numerical comparison, not a proof of an all-policy physical Bernoulli floor or of the stopped-score lemma. |
| `eq:fisher_probe`, `prop:degenerate`, `conj:smooth` | `check_extra.py` | `check_extra_output.json` | Probe metric and natural-gradient limit; an indicative Gaussian-price crossing; Riccati slopes for stationary smoother-drift policies only, not the open adaptive lower bound. |
| `prop:memory_drift`, `prop:decode_error`, `prop:task_risk`, `prop:two_optima`, `tab:two_optima` | `check_section9.py` | `check_section9_output.json` | Analytic bounds including 0.4518, and direct enumeration of the M=20 task-risk example. Does not rerun Model A/B Monte Carlo. |
| `def:joint`, `prop:purity_copy`, `subsec:J_ledger` | `c1_reductions.py` | `c1_reductions.json` | Reductions and elementary feedback/copy-versus-blank resource numbers, using common.py. Not a proof or simulation of the complete apparatus ledger. |
| `thm:gauge`, `prop:nostandard` | `c2_gauge_enum.py` | `c2_gauge_enum.json` | Finite binary record/loss enumeration and linear Gaussian likelihood comparisons under joint relabeling. The all-policy assertion rests on the written covariance proof. |
| `thm:inversion` | `c3_sign_inversion.py` | `c3_sign_inversion.json` | Finite-horizon Bayes-optimal losses and natural-policy long runs. Its stored bound_general uses a maximum conditional bias (the later r_* type bound), not the sharper b_* formula of the current all-policy theorem. The explicit eta<=d comparison is identified separately in the script. |
| `prop:sqrtlaw` | `c4_jump_price.py` | `c4_jump_price.json` | Exogenous/periodic-reference benchmarks and noisy periodic-policy scans. Does not check the later adaptive reference bound, the 0.068 counterexample or prop:adaptive_exact. |
| `prop:J0`, `prop:tax`, `thm:ledger` | `c5_driftlaw_J0.py` | `c5_driftlaw_J0.json` | Gaussian free-information floor and reading-tax factor; a separate bounded body-moment tracker. The latter is not a joint Fisher-information-rate certificate for overlapping body products. |
| `prop:loopmoments` | `c7_closedloop_moments.py` | `c7_closedloop_moments.json` | Direct finite binary-path enumeration of moment identities and their inversion; arbitrary initial body means, with the current theorem retaining its stationary/nondegeneracy assumptions. |
| `subsec:J_consequences` | `c8_push_frame.py` | `c8_push_frame.json` | The D=6 shared-versus-independent sensing/action push regression example, not a general learning theorem for arbitrary couplings. |
| `subsec:J_viability`, `tab:viability` | `economy.py` | `economy_output.json` | Exact bounded-block posterior/reset/drive calculations: food-only optimum then body cost for the SAME drive state, including the external record-purity entries. Not the continuing filter or cyclic apparatus net. |
| `subsec:online_joint`, `prop:finite_filter`, `app:filter_certificates` | `online_joint.py` | `online_joint_output.json`, `online_joint_confirmation.json` | Continuing eight-state filter, periodic/adaptive reference comparisons and projective constants. The random binary64 quantization diagnostics are not a rigorous machine certificate. Two archived outputs use different seed/horizon options, listed below. |
| `prop:body_event`, `prop:body_blackout`, `prop:body_memory_value`, `tab:body_access_ledger`, `app:filter_certificates` | `independent_checks.py` | `check_results.json` | Exact rational first-return/eigenmode and memory-food paths, CP-extension and selected supplied-resource entries. The rational s=49/50 witness is separated from the actual manuscript drive; this is not a complete cyclic resource ledger. |
| `prop:repair_witness`, `eq:short_repair`, `prop:adaptive_exact`, `prop:sqrtlaw` | `check_subject.py` | `check_subject_output.json` | Static two-slot repair witness and exact rational quiet-threshold Bellman certificate. sqrtlaw here names the adaptive_exact certificate; it is not c4’s asymptotic scan. Does NOT compute prop:repair_rates, tab:repair_drift or the continuing enquiry examples. |
| `eq:bank_step`, `subsec:model_A`, `tab:loop`, `prop:memory_drift`, `prop:decode_error` | `closed_loop.py` | `closed_loop_output.json` | Count-compressed Model A comparison (static, normal, x8 only), resource counts and an older unrolled analytic bound. Not the source of the x4 table row or the sharp 0.4518 calculation; see d3_repro/check_section9. |
| `subsec:model_A`, `tab:loop`, `app:loop_protocol` | `d3_repro.py` | `d3_repro_output.json` | Bit-by-bit Model A comparison; these four stored regimes supply the printed table, with additional x2/x16 scans retained. Uses platform_e0.py. Distinct streams across policies, not a paired-policy standard-error claim. |
| `subsec:model_B`, `tab:cells`, `app:loop_protocol` | `chain_v2.py` | `chain_v2_output.json` | Three-block Model B alignment/authority comparison, all fault cells and command-cap check. Not the different Foundation-II calibration experiment. |

Helper modules are dependency entries, not independent tests. online_joint recorded options are preserved per output in JSON.

- lem:stopped_score conditional-average bound: the article gives the general proof. The separate public supplemental/stopped_score directory supplies a finite-tree illustration and a counterexample to an unconditional-only budget; it is not a discrete-prior van Trees proof.
- prop:access_moments, prop:enquiry_moments, prop:repair_rates and tab:repair_drift: not computed by the bundled check_subject.py; their written proofs and other research evidence are not silently attributed to that script.
- The 0.068 adaptive-reference counterexample and nonideal 0.0687253 number are not computed by c4_jump_price.py or check_subject.py.
