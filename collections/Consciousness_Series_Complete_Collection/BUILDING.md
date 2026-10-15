# The Consciousness Series

Author: Sidong Liu.

The reference collection preserves the introduction and member paper pages by native PDF concatenation. The supplied source files supplies the actual introduction, all component PDFs and each member's source and dependencies.

Install PyMuPDF and run `python3 assemble.py --output rebuilt.pdf`. Add `--compile-frontmatter` to rebuild `frontmatter.tex` with XeLaTeX and latexmk. To rebuild the member papers, follow `member_sources/ID/BUILDING.md`; replace the corresponding ID.pdf and update its SHA-256 in assembly.json when intentionally using a new build.

`Consciousness_Series_Complete_Collection.tex` is a newly generated pdfpages convenience entry, not the source of the reference PDF. The supported reference assembly route is assemble.py, which preserves internal links and page geometry. Tool versions, PDF IDs and timestamps can prevent byte-identical regeneration. The source packages contain no old monolithic collection body.
