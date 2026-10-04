"""Declared ten-epoch K1 ce_ep intersection arm; no retry or grid."""
import json
import os
from pathlib import Path
import subprocess

baseline=Path('/content/imp-runs/officehome-capacity-10e-v1/estimated/launch.json')
launch=json.loads(baseline.read_text())
output=Path('/content/imp-runs/officehome-imp-intersection-10e-v3')
output.mkdir(exist_ok=False)
command=list(launch['command'])
command[2]='/content/train_imp_intersection_entry.py'
command[command.index('--log_dir')+1]=str(output)
env=dict(os.environ,RTA_SEED='1',RTA_EPOCHS='10',KONLY_SOURCE_PRIOR=launch['source_prior'],
    RTA_MODEL_PATH='/content/osda-datasets/resnet50-19c8e357.pth',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',
    IMP_INITIAL_ASSIGNMENTS='/content/imp-runs/officehome-capacity-10e-v1/initial-imp-assignments.npz')
launch.update(method='Frozen initial IMP intersection of ce_ep candidates only',command=command,
    training_changes='Only ce_ep sample selection; K1 and all other settings unchanged',
    caveat='Independent method version, not K-only; initial geometry is not refreshed')
(output/'launch.json').write_text(json.dumps(launch,indent=2))
with (output/'console.log').open('x') as log:
    p=subprocess.Popen(command,cwd='/content/rta-legacy-l4-bridge-v1',env=env,
        stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
    for line in p.stdout:
        log.write(line);log.flush();print(line,end='',flush=True)
    if p.wait():
        raise RuntimeError('Intersection arm failed; preserve evidence')
history=[json.loads(x) for x in (output/'officehome-pr2rw_seed1/history.jsonl').read_text().splitlines()]
if [x['epoch'] for x in history]!=list(range(1,11)):
    raise ValueError('Expected ten complete epochs')
print('IMP_INTERSECTION_ARM_COMPLETE',flush=True)
