# Stopped reading-score illustration

`check_stopped_score.py` is a finite-tree check of the conditional-average information budget in Lemma `lem:stopped_score`. It uses Python's standard library. Run a copy in a scratch directory if the archived `stopped_average.json` is to be preserved: the script writes that filename beside itself.

The adaptive design depends on acquired records, not on the unknown parameter. The stopped-score isometry and conditional Bayesian bound hold although the pointwise Fisher bound is violated at one parameter value. A second exact rational example shows why an unconditional potential-read average cannot replace a conditional budget after selection.

The two-point mixing law tests only the reading-score identity. It is not a van Trees proof for a discrete prior or a simulation of the drift apparatus. The article gives the general argument with regular likelihoods.
