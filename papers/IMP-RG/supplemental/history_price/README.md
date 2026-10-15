# Exact finite-comparison price interval

Run `python3 check_history_price.py` with NumPy and SciPy installed. It enumerates the 4001 fixed allocations through analytic binomial decision boundaries, solves the specified 64-state history-policy chain, and replaces `HISTORY_PRICE.json`.

The exact threshold is the finite Bayes-sum minimum minus the stated chain's stationary error. The displayed decimals are binary64 evaluations, not directed-rounding certificates. The closed-form lower interval remains available without using the sharper numerical value. This comparison assumes the paper's equal read/preparation prices and supplied port menu; it does not equate all physical apparatus costs or prove adaptive optimality.
