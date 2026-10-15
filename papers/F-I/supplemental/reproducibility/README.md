# F-I reproducibility bundle

Permission bridge, finite hull and character checks. The inherited text outputs are included with their original scripts.

SOURCE_MAP.json lists every original and distributed SHA-256, with the paper owner and any final-newline-only normalization. The source_tree contains the actual inputs, not links to a private workspace. Original contributor/path names are retained as provenance, not statements of independent verification.

Use Python with NumPy, SciPy and mpmath; tested versions are recorded in SOURCE_MAP.json. Run a check only in an isolated copy:

    python3 run_check.py SCRIPT.py --output-dir /path/to/a/new/run

The output directory must not exist. The wrapper copies the input tree before running, retaining relative imports and data paths. It saves stdout, stderr and RUN.json. Do not run scripts in the frozen source_tree: some write beside themselves.

Large historical Monte Carlo runs can be expensive. Packaging did not rerun every experiment. Archived outputs identify the evidence underlying the paper; their hashes do not certify the calculations. Individual seeds, horizons and resource assumptions remain in each script and output. Do not replace them with an unlabelled shortened run.
