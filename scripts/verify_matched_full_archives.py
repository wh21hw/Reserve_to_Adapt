"""Verify one downloaded full-budget result/checkpoint pair against manifest."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import zipfile
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from matched_full_audit import validate_full_logs

def digest_stream(stream):
    sha=hashlib.sha256()
    for chunk in iter(lambda:stream.read(1024**2),b''): sha.update(chunk)
    return sha.hexdigest()

parser=argparse.ArgumentParser()
parser.add_argument('--arm',choices=('structure_off','structure_on'),required=True)
args=parser.parse_args()
root=Path(__file__).resolve().parents[1]/'pipeline-results'
prefix=f'matched-full-a2w-seed1-v1-{args.arm}'
manifest=json.loads((root/f'{prefix}-archive-manifest.json').read_text())
assert {row['file'] for row in manifest}=={f'{prefix}-results.zip',f'{prefix}-checkpoint.zip'}
for row in manifest:
    path=root/row['file']
    assert path.stat().st_size==row['bytes']
    with path.open('rb') as stream: assert digest_stream(stream)==row['sha256']
with zipfile.ZipFile(root/f'{prefix}-results.zip') as archive:
    assert archive.testzip() is None
    read=lambda name:json.loads(archive.read(name))
    lines=lambda name:[json.loads(line) for line in archive.read(name).splitlines()]
    config,summary=read('config.json'),read('summary.json')
    report=validate_full_logs(config,lines('history.jsonl'),lines('batches.jsonl'),summary)
    independent=read('independent-audit.json')
    assert independent['independent_checkpoint_evaluation_verified'] and independent['complete_budget']
    assert independent['collector_optimizer_steps']==0 and independent['final_epoch_verified']==70
    assert independent['independent_final_metrics']==summary['final']['metrics']
    assert independent['checkpoint_sha256']==summary['checkpoint_sha256']
    report['independent_checkpoint_evaluation_verified']=True
with zipfile.ZipFile(root/f'{prefix}-checkpoint.zip') as archive:
    assert archive.testzip() is None
    assert archive.namelist()==['last.pt']
    with archive.open('last.pt') as stream: assert digest_stream(stream)==summary['checkpoint_sha256']
print('LOCAL_FULL_ARCHIVE_VERIFICATION_PASS',args.arm,json.dumps(report))
