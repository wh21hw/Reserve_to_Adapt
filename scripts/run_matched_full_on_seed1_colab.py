"""ONE fixed on-arm; requires independently audited completed off-arm."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
for filename, expected in {
    'matched_full_budget.py':'e764da6a387160c5c98d906c1e9aed0dbd1afafb2e70458a443fe072c80852e3',
    'train_matched_full_a2w_seed1.py':'0d351d40b13bbe021754c1271dfea33de3788cc64fca7c9ccca81c1829ef3826',
}.items():
    assert hashlib.sha256((Path('/content')/filename).read_bytes()).hexdigest()==expected
off = Path('/content/imp-runs/matched-structure-full-a2w-seed1-v1/structure_off')
audit = json.loads((off/'independent-audit.json').read_text())
assert audit['independent_checkpoint_evaluation_verified'] and audit['complete_budget']
assert audit['final_epoch_verified']==70 and audit['new_optimizer_updates_verified']==924
assert hashlib.sha256((off/'last.pt').read_bytes()).hexdigest()==audit['checkpoint_sha256']
with Path('/content/matched-full-a2w-seed1-on-console-v1.log').open('x') as stream:
    process=subprocess.Popen([sys.executable,'-u','/content/train_matched_full_a2w_seed1.py','--arm','structure_on'],
                             stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
    for line in process.stdout:
        print(line,end='',flush=True)
        stream.write(line); stream.flush()
    status=process.wait()
if status: raise RuntimeError(f'Full on-arm failed ({status}); no automatic retry')
