"""One new ten-epoch fixed-K arm; reuse the completed ordinary-BN RTA control."""
import json
import os
from pathlib import Path
import subprocess

control=Path('/content/imp-runs/officehome-frozenbn-capacity-10e-v1/fixed4')
launch=json.loads((control/'launch.json').read_text())
history=[json.loads(line) for line in (control/'officehome-pr2rw_seed1/history.jsonl').read_text().splitlines()]
if launch['K']!=4 or [row['epoch'] for row in history]!=list(range(1,11)):
    raise ValueError('Expected completed fixedK4 control')
output=Path('/content/imp-runs/officehome-rta-frozenbn-10e-v1')
output.mkdir(exist_ok=False)
command=list(launch['command']);command[command.index('--log_dir')+1]=str(output)
env=dict(os.environ,OPENBLAS_NUM_THREADS='2',OMP_NUM_THREADS='2',RTA_SEED='1',RTA_EPOCHS='10',
    RTA_FREEZE_ENCODER_BN='1',KONLY_SOURCE_PRIOR=launch['source_prior'],
    RTA_MODEL_PATH='/content/osda-datasets/resnet50-19c8e357.pth')
manifest=dict(launch,command=command,arm='rta-frozen-encoder-bn',control=str(control),
    research_change='Only encoder BN running-statistics policy in RTA; affine/weights/head train normally',
    RTA_FREEZE_ENCODER_BN=True,adaptive_K=False)
(output/'launch.json').write_text(json.dumps(manifest,indent=2))
with (output/'console.log').open('x') as log:
    p=subprocess.Popen(command,cwd='/content/rta-legacy-l4-bridge-v1',env=env,
        stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
    for line in p.stdout:print(line,end='',flush=True);log.write(line);log.flush()
    if p.wait():raise RuntimeError('BN ablation failed; preserve evidence, no automatic retry')
history=[json.loads(line) for line in (output/'officehome-pr2rw_seed1/history.jsonl').read_text().splitlines()]
if [row['epoch'] for row in history]!=list(range(1,11)):raise RuntimeError('Incomplete budget')
print('RTA_FROZEN_ENCODER_BN_ARM_COMPLETE',flush=True)
