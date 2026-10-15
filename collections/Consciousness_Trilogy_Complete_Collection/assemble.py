#!/usr/bin/env python3
"""Assemble unchanged component pages while preserving internal PDF links.

Requires PyMuPDF (the reference environment used version 1.28.0).
Run from any directory: python3 assemble.py [--output OUTPUT.pdf]
--compile-frontmatter rebuilds the introduction with XeLaTeX first.
Member source bundles live in member_sources; build each using BUILDING.md.
PDF byte identity depends on tool versions and generated PDF IDs/timestamps;
the reference PDF is included separately as the published manuscript.
"""
from pathlib import Path
import argparse, hashlib, json, subprocess
import fitz
from urllib.parse import unquote

HERE=Path(__file__).resolve().parent

def import_pages_with_links(destination, source, offset):
    # insert_pdf does not copy named destinations. Resolve each source named
    # citation/equation target first and encode the original target explicitly.
    destination.insert_pdf(source, links=False)
    for index, page in enumerate(source):
        for link in page.get_links():
            target = dict(link)
            for key in ('xref', 'id', 'nameddest'):
                target.pop(key, None)
            if link['kind'] in (fitz.LINK_GOTO, fitz.LINK_NAMED):
                if link.get('page', -1) < 0 or 'to' not in link:
                    raise RuntimeError('Unresolved original internal destination: '+str(link))
                target['kind'] = fitz.LINK_GOTO
                target['page'] = link['page'] + offset
            elif link['kind'] != fitz.LINK_URI:
                raise RuntimeError('Unsupported original link kind: '+str(link))
            destination[offset+index].insert_link(target)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=HERE/'rebuilt.pdf')
    parser.add_argument('--compile-frontmatter',action='store_true')
    args=parser.parse_args()
    recipe=json.loads((HERE/'assembly.json').read_text())
    if args.compile_frontmatter:
        subprocess.run(['latexmk','-norc','-use-make-','-xelatex',
                        '-interaction=nonstopmode','-halt-on-error','-no-shell-escape',
                        'frontmatter.tex'],cwd=HERE,check=True)
    doc=fitz.open()
    with fitz.open(HERE/'frontmatter.pdf') as front:
        doc.insert_pdf(front)
    toc=[[1,'Cover',1],[1,'Contents',2]]
    for item in recipe['members']:
        path=HERE/item['pdf']
        if hashlib.sha256(path.read_bytes()).hexdigest()!=item['sha256']:
            raise RuntimeError('Component hash changed: '+item['pdf'])
        with fitz.open(path) as source:
            offset=len(doc)
            import_pages_with_links(doc,source,offset)
            toc.append([1,item['title'],offset+1])
            for level,title,page in source.get_toc():
                if page>0:toc.append([level+1,title,offset+page])
    # Turn frontmatter's portable self-PDF contents links into native destinations.
    for page in doc:
        for link in page.get_links():
            uri=unquote(link.get('uri', link.get('file', '')))
            while unquote(uri)!=uri:
                uri=unquote(uri)
            prefix=recipe['official_pdf']+'#'
            if uri.startswith(prefix):
                anchor=uri[len(prefix):].removeprefix('nameddest=')
                if not anchor.startswith('page='):
                    raise RuntimeError('Unknown contents anchor: '+uri)
                target=int(anchor.removeprefix('page='))-1
                if not 0 <= target < len(doc):
                    raise RuntimeError('Contents destination outside volume: '+uri)
                page.delete_link(link)
                page.insert_link({'kind':fitz.LINK_GOTO,'from':link['from'],
                                  'page':target,'to':fitz.Point(0,0)})
    doc.set_toc(toc)
    doc.set_metadata({'title':recipe['title'],'author':'Sidong Liu',
                      'subject':'Genesis Trilogy: collected independent papers'})
    if args.output.resolve()==(HERE/recipe['official_pdf']).resolve():
        raise RuntimeError('Choose a fresh output; retain the reference manuscript.')
    doc.save(args.output,garbage=4,deflate=True)
    doc.close()
    print(args.output)

if __name__=='__main__':main()
