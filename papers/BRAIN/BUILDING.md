# BRAIN

The Brain as a Plastic Observable-Algebra Network: Learning Shared Access from Sensorimotor History

Author: Sidong Liu.

Compile `Brain_Anchor_v2.tex` in this directory with xelatex:

```sh
latexmk -norc -use-make- -xelatex -interaction=nonstopmode -halt-on-error -no-shell-escape Brain_Anchor_v2.tex
```

Requires a current TeX Live distribution and the packages named by the source. XeLaTeX documents may require TeX Gyre and Latin Modern fonts. All project-local dependencies are supplied with relative paths. Supplementary code has its own dependency and run instructions; some historical full runs are costly. Build in a copy to retain the reference PDF. Recompiled PDFs may differ in metadata, IDs and font embedding.
