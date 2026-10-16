# HAFF-D

Selection, Maintenance and Gravitational Response: Modular Constraints and a Physical Return

The complete technical appendices are a companion supplement. Keep both sources and PDFs together. The pair uses reciprocal xr-hyper references; alternate builds until auxiliary files settle. The first pass may lack the other auxiliary file; the final pass must have no unresolved references.

Build in a working copy:

```sh
for round in 1 2 3; do
  pdflatex -interaction=nonstopmode -halt-on-error HAFF_D_manuscript.tex
  pdflatex -interaction=nonstopmode -halt-on-error HAFF_D_supplement.tex
done
```

The reference PDF is supplied; regenerated IDs, dates or font embedding may differ. Scientific scripts have their own stated assumptions and dependencies.
