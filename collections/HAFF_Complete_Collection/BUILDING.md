# The Holographic Alaya-Field Framework

Member PDFs, TeX and scientific supplements are kept once in `../../papers/`; this volume does not duplicate them. Keep the repository directory layout when downloading.

From the repository root:

```sh
python3 tools/assemble_collection.py HAFF_Complete_Collection --output rebuilt.pdf
```

Requires PyMuPDF. Rebuild the introduction with XeLaTeX if needed, then use the assembler. `HAFF_Complete_Collection.tex` is a pdfpages convenience entry; it preserves pages but need not preserve the reference PDF's hyperlinks. The linked D/QR-C technical supplements remain in their paper directories.
