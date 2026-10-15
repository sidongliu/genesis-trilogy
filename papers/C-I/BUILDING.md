# C-I

Geometry Relative to a Cut: Selection, Generation and Modular Inseparability

Author: Sidong Liu.

Compile `CI_Inseparability.tex` in this directory with xelatex:

```sh
latexmk -norc -use-make- -xelatex -interaction=nonstopmode -halt-on-error -no-shell-escape CI_Inseparability.tex
```

Requires a current TeX Live distribution and the packages named by the source. XeLaTeX documents may require TeX Gyre and Latin Modern fonts. All project-local dependencies are supplied with relative paths. Supplementary code has its own dependency and run instructions; some historical full runs are costly. Build in a copy to retain the reference PDF. Recompiled PDFs may differ in metadata, IDs and font embedding.
