# QR-ME

Recovering Mass-Energy Equivalence within Algebraic Lorentzian Read-out: Spectral Mass, Lattice Windows and Finite Records

Author: Sidong Liu.

Compile `QRAIF_Mass_Energy_manuscript.tex` in this directory with xelatex:

```sh
latexmk -norc -use-make- -xelatex -interaction=nonstopmode -halt-on-error -no-shell-escape QRAIF_Mass_Energy_manuscript.tex
```

Requires a current TeX Live distribution and the packages named by the source. XeLaTeX documents may require TeX Gyre and Latin Modern fonts. All project-local dependencies are supplied with relative paths. Supplementary code has its own dependency and run instructions; some historical full runs are costly. Build in a copy to retain the reference PDF. Recompiled PDFs may differ in metadata, IDs and font embedding.
