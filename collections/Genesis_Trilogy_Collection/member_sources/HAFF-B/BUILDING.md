# HAFF-B

Coupling, Records, and Stable Relations: Formation, Protected Content and the Lifetime of a Geometry

Author: Sidong Liu.

Compile `HAFF_B_manuscript.tex` in this directory with pdflatex:

```sh
latexmk -norc -use-make- -pdf -interaction=nonstopmode -halt-on-error -no-shell-escape HAFF_B_manuscript.tex
```

Requires a current TeX Live distribution and the packages named by the source. XeLaTeX documents may require TeX Gyre and Latin Modern fonts. All project-local dependencies are supplied with relative paths. Supplementary code has its own dependency and run instructions; some historical full runs are costly. Build in a copy to retain the reference PDF. Recompiled PDFs may differ in metadata, IDs and font embedding.
