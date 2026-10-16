# QR-C

Representing Control in the World: Retained History, Physical Sources and Renewal in Q-RAIF

The complete technical appendices are a companion supplement. Keep both sources and PDFs together. The pair uses reciprocal xr-hyper references; alternate builds until auxiliary files settle. The first pass may lack the other auxiliary file; the final pass must have no unresolved references.

Build in a working copy:

```sh
for round in 1 2 3; do
  pdflatex -interaction=nonstopmode -halt-on-error QRAIF_C_manuscript.tex
  pdflatex -interaction=nonstopmode -halt-on-error QRAIF_C_supplement.tex
done
```

The reference PDF is supplied; regenerated IDs, dates or font embedding may differ. Scientific scripts have their own stated assumptions and dependencies.
