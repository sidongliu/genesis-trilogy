# F-II reproducibility bundle

Calibration/fault experiments, selected independent checks, and aggregation of the three archived JSON inputs. The numeric-only aggregator recalculates f2_numbers.json; it generates no manuscript text, TeX tables or PDF. Fixed-window data is distinct from a current script's default run.

SOURCE_MAP.json lists every original and distributed SHA-256, with the paper owner and any final-newline-only normalization. The source_tree contains the actual inputs, not links to a private workspace. Original contributor/path names are retained as provenance, not statements of independent verification.

Use Python with NumPy, SciPy and mpmath; tested versions are recorded in SOURCE_MAP.json. Run a check only in an isolated copy:

    python3 run_check.py SCRIPT.py --output-dir /path/to/a/new/run

The output directory must not exist. The wrapper copies the input tree before running, retaining relative imports and data paths. It saves stdout, stderr and RUN.json. Do not run scripts in the frozen source_tree: some write beside themselves.

Large historical Monte Carlo runs can be expensive. Packaging did not rerun every experiment. Archived outputs identify the evidence underlying the paper; their hashes do not certify the calculations. Individual seeds, horizons and resource assumptions remain in each script and output. Do not replace them with an unlabelled shortened run.

To reproduce the supplied numeric aggregate in an isolated working tree:

    python3 run_check.py calculate_f2_numbers.py --output-dir /path/to/a/new/f2_numbers_run

The resulting source_tree retains the relative JSON input paths. The archived 4,894-byte JSON is byte-identical to the output of this split script. SOURCE_MAP records the extraction and the original generator hash; the original writing tool is not included in this release.
