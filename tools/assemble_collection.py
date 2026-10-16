#!/usr/bin/env python3
"""Assemble a collection from the repository's single canonical papers tree.
Requires PyMuPDF. Run in a downloaded copy, with the repository layout intact.
"""
from pathlib import Path
from urllib.parse import unquote
import argparse,hashlib,json,re
import fitz

ROOT=Path(__file__).resolve().parents[1]
SUPPLEMENTS={'HAFF_D_supplement.pdf':'../../papers/HAFF-D/HAFF_D_supplement.pdf',
             'QRAIF_C_supplement.pdf':'../../papers/QR-C/QRAIF_C_supplement.pdf'}

def remote(page,link):
    if not link.get('xref') or page.parent.xref_get_key(link['xref'],'A/S')!=('name','/GoToR'):return None
    doc=page.parent;kind,file=doc.xref_get_key(link['xref'],'A/F');fields=['A/F']
    if kind=='dict':
        fields=[];values=[]
        for field in ['A/F/F','A/F/UF']:
            typ,value=doc.xref_get_key(link['xref'],field)
            if typ=='string':fields.append(field);values.append(value)
        assert values and len(set(values))==1
        file=values[0]
    else:assert kind=='string',(kind,file)
    return file,fields

def import_pages(doc,src,offset):
    doc.insert_pdf(src,links=False)
    for i,page in enumerate(src):
        for link in page.get_links():
            info=remote(page,link)
            if info:
                file,fields=info;target=SUPPLEMENTS.get(Path(unquote(file)).name,file)
                raw=src.xref_object(link['xref'],compressed=False)
                raw=re.sub(r'/P\s+\d+\s+\d+\s+R\b','',raw)
                assert not re.search(r'\b\d+\s+\d+\s+R\b',raw)
                x=doc.get_new_xref();doc.update_object(x,raw)
                destpage=doc[offset+i];doc.xref_set_key(x,'P',str(destpage.xref)+' 0 R')
                for field in fields:doc.xref_set_key(x,field,fitz.get_pdf_str(target))
                kind,arr=doc.xref_get_key(destpage.xref,'Annots')
                if kind=='null':arr='[]'
                elif kind=='xref':arr=doc.xref_object(int(arr.split()[0]),compressed=False)
                else:assert kind=='array'
                doc.xref_set_key(destpage.xref,'Annots',arr.rstrip()[:-1]+' '+str(x)+' 0 R ]')
            else:
                new=dict(link)
                for key in ['xref','id','nameddest']:new.pop(key,None)
                if link['kind'] in [fitz.LINK_GOTO,fitz.LINK_NAMED]:
                    assert link.get('page',-1)>=0 and 'to' in link
                    new.update(kind=fitz.LINK_GOTO,page=link['page']+offset)
                else:assert link['kind']==fitz.LINK_URI
                doc[offset+i].insert_link(new)

def assemble(folder,output):
    recipe=json.loads((folder/'assembly.json').read_text())
    assert output.resolve()!=(folder/recipe['official_pdf']).resolve(),'Preserve the reference PDF; choose another output'
    doc=fitz.open();front=fitz.open(folder/'frontmatter.pdf');doc.insert_pdf(front,links=False)
    offset=len(front);toc=[[1,'Introduction and reading map',recipe['intro_page']]]
    for row in recipe['members']:
        source=(folder/row['pdf']).resolve()
        assert hashlib.sha256(source.read_bytes()).hexdigest()==row['sha256'],source
        src=fitz.open(source);import_pages(doc,src,offset)
        toc.append([1,row['title'],offset+1])
        toc.extend([level+1,title,offset+page] for level,title,page in src.get_toc() if page>0)
        offset+=len(src);src.close()
    for i,page in enumerate(front):
        for link in page.get_links():
            uri=unquote(link.get('uri',link.get('file','')))
            match=re.fullmatch(re.escape(recipe['official_pdf'])+r'#(?:nameddest=)?page=(\d+)',uri)
            if match:
                p=int(match[1])-1;assert 0<=p<len(doc)
                doc[i].insert_link(dict(kind=fitz.LINK_GOTO,**{'from':link['from']},page=p,to=fitz.Point(0,0)))
            else:
                assert link['kind']==fitz.LINK_URI,link
                doc[i].insert_link(dict(kind=fitz.LINK_URI,**{'from':link['from']},uri=link['uri']))
    doc.set_toc(toc);doc.set_metadata(recipe['metadata']);doc.save(output,garbage=4,deflate=True)
    doc.close();front.close()

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('collection',help='Collection directory name under collections/')
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();assemble(ROOT/'collections'/args.collection,args.output.resolve())
