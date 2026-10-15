"""Run a mapped historical check in a new, isolated copy of its inputs.

No research tree or distributed frozen input is changed. The output
directory must not already exist; scripts may take hours on full settings.
"""
from pathlib import Path
import argparse,json,shutil,subprocess,sys

here=Path(__file__).resolve().parent
parser=argparse.ArgumentParser()
parser.add_argument('check',help='script basename, or mapped relative source path')
parser.add_argument('--output-dir',required=True,type=Path)
args, script_args=parser.parse_known_args()
if script_args[:1]==['--']:script_args=script_args[1:]
mapping=json.loads((here/'SOURCE_MAP.json').read_text())
entries=[e for e in mapping['files'] if e['source'].endswith('.py') and (e['source']==args.check or Path(e['source']).name==args.check)]
if len(entries)!=1:parser.error('Choose one mapped script path; basename is missing or ambiguous.')
output=args.output_dir.resolve()
if output.exists():parser.error('Output directory already exists; choose a fresh run destination.')
output.mkdir(parents=True)
tree=output/'source_tree'
shutil.copytree(here/'source_tree',tree)
script=tree/entries[0]['source']
(script.parent/'out').mkdir(exist_ok=True)
with (output/'stdout.txt').open('w') as stdout, (output/'stderr.txt').open('w') as stderr:
    result=subprocess.run([sys.executable,str(script),*script_args],cwd=script.parent,stdout=stdout,stderr=stderr)
(output/'RUN.json').write_text(json.dumps({'check':entries[0],'exit_code':result.returncode,'argv':script_args,'output_tree':str(tree)},indent=2)+'\n')
print('Exit',result.returncode,'; run artifacts in',output)
sys.exit(result.returncode)
