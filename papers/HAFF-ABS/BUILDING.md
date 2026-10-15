# HAFF-ABS

The Holographic Alaya-Field Framework: World, Interface and Subject

Author: Sidong Liu.

Compile `HAFF_Abstract.tex` in this directory with pdflatex:

```sh
latexmk -norc -use-make- -pdf -interaction=nonstopmode -halt-on-error -no-shell-escape HAFF_Abstract.tex
```

Requires a current TeX Live distribution and the packages named by the source. XeLaTeX documents may require TeX Gyre and Latin Modern fonts. All project-local dependencies are supplied with relative paths. Supplementary code has its own dependency and run instructions; some historical full runs are costly. Build in a copy to retain the reference PDF. Recompiled PDFs may differ in metadata, IDs and font embedding.
