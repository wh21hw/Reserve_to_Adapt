"""VisDA source-only stage; capacity and RTA are separate later executions."""
import json
import os
from pathlib import Path
import subprocess

data=Path('/content/osda-visda-syn2real-v1')
ready=json.loads((data/'data-preparation.json').read_text())
if (ready['source_samples'],ready['target_samples'])!=(79765,55388):
    raise ValueError('Unexpected prepared VisDA sample protocol')
source=data/'source-known-6.txt';target=data/'target-unlabeled-paths.txt'
source_rows=source.read_text().splitlines();target_rows=target.read_text().splitlines()
known=sorted({int(line.rsplit(None,1)[1]) for line in source_rows})
if known!=[1,2,3,6,10,11] or len(source_rows)!=79765 or len(target_rows)!=55388:
    raise ValueError('Unexpected known IDs or list sizes')
if any(len(line.split())!=1 for line in target_rows):
    raise ValueError('Expected unlabeled target paths only')
root=Path('/content/imp-runs/visda-frozenbn-capacity-10e-v1')
root.mkdir(exist_ok=False)
command=['/content/rta-py38/bin/python','-u','/content/train_visda_frozenbn_source_v1.py',
    '--code-root','/content/rta-legacy-l4-bridge-v1','--source',str(source),
    '--target',str(target),'--data-root',str(data),'--weights','/content/osda-datasets/resnet50-19c8e357.pth',
    '--output',str(root/'source'),'--seed','1','--epochs','3','--freeze-backbone-bn','--log-every','200']
manifest=dict(task='VisDA Synthetic->Real',C=6,source_rows=79765,target_rows=55388,known_original_ids=known,
    seed=1,source_epochs=3,backbone='ResNet50',source_encoder_bn_frozen=True,
    optimizer='Same source SGD as OfficeHome; encoder/head lr5e-5/5e-4; momentum.9/nesterov/wd5e-4',
    target_labels_used=False,command=command,capacity_stage_started=False,RTA_stage_started=False,
    caveat='User-specified ResNet; paper table/text backbone and original VisDA epoch budget unresolved')
(root/'source-launch.json').write_text(json.dumps(manifest,indent=2))
env=dict(os.environ,OPENBLAS_NUM_THREADS='2',OMP_NUM_THREADS='2')
with (root/'source-console.log').open('x') as log:
    p=subprocess.Popen(command,cwd='/content/rta-legacy-l4-bridge-v1',env=env,
        stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
    for line in p.stdout:print(line,end='',flush=True);log.write(line);log.flush()
    if p.wait():raise RuntimeError('VisDA source stage failed; preserve evidence, no automatic retry')
print('VISDA_FROZENBN_SOURCE_STAGE_COMPLETE',flush=True)
