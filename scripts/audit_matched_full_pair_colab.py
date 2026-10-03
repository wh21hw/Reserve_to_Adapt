"""Only run once both independent complete-budget collectors have passed."""
import json
from pathlib import Path
import sys
sys.path.insert(0,'/content')
from matched_full_pair_audit import validate_full_pair

root=Path('/content/imp-runs/matched-structure-full-a2w-seed1-v1')
def load(arm):
    path=root/arm
    read=lambda name:json.loads((path/name).read_text())
    lines=lambda name:[json.loads(x) for x in (path/name).read_text().splitlines()]
    return (read('config.json'),lines('history.jsonl'),lines('batches.jsonl'),read('summary.json')),read('independent-audit.json')
off,io=load('structure_off')
on,inn=load('structure_on')
report=validate_full_pair(off,on,io,inn)
with Path('/content/matched-full-a2w-seed1-pair-audit-v1.json').open('x') as stream:
    json.dump(report,stream,indent=2,allow_nan=False)
print('FULL_PAIR_AUDIT_PASS',json.dumps(report),flush=True)
