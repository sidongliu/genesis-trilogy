# TD-II

Reference Frames Selected by History: The Capability and Cost of a Bounded Agent's Self-Frame

Author: Sidong Liu.

Compile `TDOME_II_manuscript.tex` in this directory with pdflatex:

```sh
latexmk -norc -use-make- -pdf -interaction=nonstopmode -halt-on-error -no-shell-escape TDOME_II_manuscript.tex
```

Requires a current TeX Live distribution and the packages named by the source. XeLaTeX documents may require TeX Gyre and Latin Modern fonts. All project-local dependencies are supplied with relative paths. Supplementary code has its own dependency and run instructions; some historical full runs are costly. Build in a copy to retain the reference PDF. Recompiled PDFs may differ in metadata, IDs and font embedding.
