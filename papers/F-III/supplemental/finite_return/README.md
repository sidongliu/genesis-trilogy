# Finite semiclassical return

This package reproduces the X-basis terminal acquisition in Foundation III's finite-return protocol. It retains the six data qubits, eight purification columns, clock path, scalar radiation and JT mean-action equations. It is not a branchwise quantum-gravity simulation or a sustained thermal-renewal model.

Python 3, NumPy and SciPy are required. From this directory run:

```sh
python3 scripts/prefix_coefficient.py
python3 scripts/full_return.py base
python3 scripts/full_return.py edge
python3 scripts/full_return.py audit
python3 scripts/full_return.py tight
python3 scripts/full_return.py null
python3 scripts/full_return.py background
```

The full integrations are substantially slower than the prefix calculation. Each command replaces its named JSON in `results/`; base/background also write trajectory arrays. The shipped JSON files are the numerical outputs used for the manuscript.

- `base`: learned/odd/even policies for both unknown plant signs, task loss, completion and complete work/junction residuals.
- `edge`: matched alpha=0/1 runs and geometric-only replays with the same mechanical history; pre/post-acquisition masks separate the mechanism.
- `audit`, `tight`: RK45 and tighter DOP853 tolerances. Agreement is a floating-point consistency check, not interval certification.
- `null`: sensor coupling zero. A remaining total port difference is not labeled gravitational.
- `background`: a separate initial-geometry comparison, not the same-initial-state intervention.
- `prefix`: full-circuit insertion derivatives and the exact rational shortest-path coefficient.

No outcome is postselected. Preparations, spent cells and their purifiers remain in the state; the ledger does not interpret their energy gain as an independent heat exhaust.
