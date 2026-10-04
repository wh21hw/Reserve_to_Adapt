"""Fixed additional source block, two three-epoch arms, no grid or target data."""
import argparse
import json
import os
from pathlib import Path
import subprocess

ROOT=Path('/content/imp-runs/source-frozenbn-block0to4-v1')
DATA=Path('/content/osda-officehome-pr2rw-v1')
parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['prepare','ce','frozenbn']);args=parser.parse_args()
if args.stage=='prepare':
    ROOT.mkdir(parents=True,exist_ok=False)
    rows=[r.rsplit(None,1) for r in (DATA/'product_0-24_train_all.txt').read_text().splitlines() if r.strip()]
    hidden=list(range(5));known=[r for r in rows if int(r[1]) not in hidden]
    if len(set(int(r[1]) for r in known))!=20:raise ValueError('Expected C20 supervision')
    (ROOT/'known-source.txt').write_text(''.join('%s %s\n'%tuple(r) for r in known))
    (ROOT/'extract-all-source.txt').write_text(''.join(r[0]+' 0\n' for r in rows))
    config=dict(hidden_ids=hidden,known_classes=20,known_samples=len(known),extract_source_samples=len(rows),source_only=True,
        real_target_used=False,seed=1,epochs_per_arm=3,arms=['source-ce','source-frozenbn'],selection='Fixed additional class block0..4 before paired results; not chosen by best performance')
    (ROOT/'protocol.json').write_text(json.dumps(config,indent=2));print('VALIDATION_PREPARED',json.dumps(config),flush=True)
else:
    arm=ROOT/('source-ce' if args.stage=='ce' else 'source-frozenbn')
    arm.mkdir(exist_ok=False)
    command=['/content/rta-py38/bin/python','-u','/content/train_source_bn_validation_v1.py','--code-root','/content/rta-legacy-l4-bridge-v1',
        '--source',str(ROOT/'known-source.txt'),'--target',str(ROOT/'extract-all-source.txt'),'--data-root',str(DATA),
        '--weights','/content/osda-datasets/resnet50-19c8e357.pth','--output',str(arm/'source'),'--seed','1','--epochs','3','--save-backbone']
    if args.stage=='frozenbn':command.append('--freeze-backbone-bn')
    (arm/'launch.json').write_text(json.dumps(dict(command=command),indent=2))
    with (arm/'console.log').open('x') as log:
        p=subprocess.Popen(command,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,env=dict(os.environ,OPENBLAS_NUM_THREADS='2',OMP_NUM_THREADS='2'))
        for line in p.stdout:
            print(line,end='',flush=True);log.write(line);log.flush()
        if p.wait():raise RuntimeError('Validation arm failed; preserve evidence')
    print('VALIDATION_ARM_COMPLETE',args.stage,flush=True)
