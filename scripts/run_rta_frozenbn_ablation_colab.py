"""One declared ten-epoch arm; reuse completed controls and the existing estimate."""
import argparse
import json
import os
from pathlib import Path
import subprocess

parser=argparse.ArgumentParser()
parser.add_argument('--arm',choices=['fixed4','estimated'],default='fixed4')
args=parser.parse_args()
pair=Path('/content/imp-runs/officehome-frozenbn-capacity-10e-v1')
control=pair/args.arm
launch=json.loads((control/'launch.json').read_text())
history=[json.loads(line) for line in (control/'officehome-pr2rw_seed1/history.jsonl').read_text().splitlines()]
expected=4
if args.arm=='estimated':
    estimate=json.loads((pair/'prior/capacity.json').read_text())
    if not estimate['converged'] or estimate['target_labels_used'] or estimate['K']!=2:
        raise ValueError('Expected existing unlabeled K2 estimate; no new inference or selection')
    expected=2
    held=Path('/content/imp-runs/officehome-rta-frozenbn-10e-v1')
    completed=json.loads((held/'summary.json').read_text())
    comparison=json.loads((held/'launch.json').read_text())
    if not completed['held_buffers_unchanged'] or any(comparison[key]!=launch[key]
            for key in ('Q','seed','epochs','source_prior')):
        raise ValueError('Missing completed matched hold-BN K4 comparison')
if launch['K']!=expected or [row['epoch'] for row in history]!=list(range(1,11)):
    raise ValueError('Expected complete matched update-BN control')
output=Path('/content/imp-runs/officehome-rta-frozenbn-'+('k2-' if args.arm=='estimated' else '')+'10e-v1')
output.mkdir(exist_ok=False)
command=list(launch['command']);command[command.index('--log_dir')+1]=str(output)
env=dict(os.environ,OPENBLAS_NUM_THREADS='2',OMP_NUM_THREADS='2',RTA_SEED='1',RTA_EPOCHS='10',
    RTA_FREEZE_ENCODER_BN='1',KONLY_SOURCE_PRIOR=launch['source_prior'],
    RTA_MODEL_PATH='/content/osda-datasets/resnet50-19c8e357.pth')
manifest=dict(launch,command=command,arm='rta-frozen-encoder-bn',control=str(control),
    research_change='Only encoder BN running-statistics policy in RTA; affine/weights/head train normally',
    RTA_FREEZE_ENCODER_BN=True,adaptive_K=args.arm=='estimated',
    capacity_result=str(pair/'prior/capacity.json') if args.arm=='estimated' else None)
(output/'launch.json').write_text(json.dumps(manifest,indent=2))
with (output/'console.log').open('x') as log:
    p=subprocess.Popen(command,cwd='/content/rta-legacy-l4-bridge-v1',env=env,
        stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
    for line in p.stdout:print(line,end='',flush=True);log.write(line);log.flush()
    if p.wait():raise RuntimeError('BN ablation failed; preserve evidence, no automatic retry')
history=[json.loads(line) for line in (output/'officehome-pr2rw_seed1/history.jsonl').read_text().splitlines()]
if [row['epoch'] for row in history]!=list(range(1,11)):raise RuntimeError('Incomplete budget')
print('RTA_FROZEN_ENCODER_BN_ARM_COMPLETE',flush=True)
