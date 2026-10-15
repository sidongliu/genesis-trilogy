# Contact-bound two-walker band

`bound_walk.py` implements both free coined steps followed by the coincidence phase exp(+0.4 i). The relative lattice is periodic, r=(xA−xB)/2, and K is conjugate to (xA+xB)/2. NumPy and SciPy are required.

Run `python3 bound_walk.py` from this directory. It replaces `ROOT_QR_NUMERICS.json`, containing the finite-size checks and the overlap-tracked band on 0≤K≤2. The sampled maximum is not a certified global group-velocity maximum. The manuscript uses the locally isolated band's rest value and curvature, not an inferred endpoint at K=0.68.
