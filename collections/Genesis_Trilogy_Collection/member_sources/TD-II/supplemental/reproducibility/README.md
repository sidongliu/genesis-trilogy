# TD-II reproducibility bundle

Resource-matched reader trajectories, threshold calculation (which imports td2_loop), and analytic checks.

SOURCE_MAP.json lists every original and distributed SHA-256, with the paper owner and any final-newline-only normalization. The source_tree contains the actual inputs, not links to a private workspace. Original contributor/path names are retained as provenance, not statements of independent verification.

Use Python with NumPy, SciPy and mpmath; tested versions are recorded in SOURCE_MAP.json. Run a check only in an isolated copy:

    python3 run_check.py SCRIPT.py --output-dir /path/to/a/new/run

The output directory must not exist. The wrapper copies the input tree before running, retaining relative imports and data paths. It saves stdout, stderr and RUN.json. Do not run scripts in the frozen source_tree: some write beside themselves.

Large historical Monte Carlo runs can be expensive. Packaging did not rerun every experiment. Archived outputs identify the evidence underlying the paper; their hashes do not certify the calculations. Individual seeds, horizons and resource assumptions remain in each script and output. Do not replace them with an unlabelled shortened run.

## Manuscript result to check to archived output

Labels below remain valid when section numbering changes. A support entry identifies the stated finite computation, not a proof of every general claim. SOURCE_MAP.json gives full paths, exact output keys and both provenance hashes. No script or archived output was changed for this mapping.

| Manuscript label(s) | Script | Archived output(s) | Scope |
|---|---|---|---|
| `thm:capability`, `thm:rate`, `cor:price`, `tab:rate`, `thm:blind`, `cor:even`, `thm:drift` | `td2_checks.py` | `td2_checks_output.json` | Finite-prior capability/top-k tests; Beta/Legendre table and tilted equality; Gaussian blind-spot covariance; isotropic rotation moment diagnostics. Not a proof of every theorem or a test of the separate quantum-reference results. |
| `prop:stationary`, `tab:loop`, `app:protocol` | `td2_loop.py` | `td2_loop_output.json` | Resource-matched trajectories and exact stationary formulas; all nine eta rows, common random world and declared start-up/read budgets. The trajectory implementation uses algebraically equivalent noiseless functionals, not independent noisy hardware reads. |
| `tab:threshold`, `prop:stationary` | `td2_threshold.py` | `td2_threshold_output.json` | Stationary-risk crossings from td2_loop.risks/best_alpha, not a second Monte Carlo experiment. Uses optimization floor 1e-4, unlike the 0.002 floor of the trajectory comparison. |

Helper modules are dependency entries, not independent tests.

- TD-II quantum-reference and signed-price results: proofs remain in the manuscript; none of these three scripts is claimed to test them.
