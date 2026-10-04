"""Separated stages of the declared real-domain ten-epoch pair."""
import argparse
import json
import os
from pathlib import Path
import subprocess

ROOT=Path('/content/imp-runs/officehome-frozenbn-capacity-10e-v1')
PRIOR=ROOT/'prior'
DATA=Path('/content/osda-officehome-pr2rw-v1')
parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['source','infer','fixed4','estimated']);args=parser.parse_args()
def run(command,log_path,env):
    with log_path.open('x') as log:
        process=subprocess.Popen(command,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,cwd='/content/rta-legacy-l4-bridge-v1',env=env)
        for line in process.stdout:
            print(line,end='',flush=True);log.write(line);log.flush()
        if process.wait():raise RuntimeError('Stage failed; preserve evidence, no automatic retry')
env=dict(os.environ,OPENBLAS_NUM_THREADS='2',OMP_NUM_THREADS='2')
if args.stage=='source':
    ROOT.mkdir(parents=True,exist_ok=False);PRIOR.mkdir()
    command=['/content/rta-py38/bin/python','-u','/content/train_frozenbn_real_officehome_v1.py',
        '--code-root','/content/rta-legacy-l4-bridge-v1','--source',str(DATA/'product_0-24_train_all.txt'),
        '--target',str(DATA/'real_world_0-64_test.txt'),'--data-root',str(DATA),
        '--weights','/content/osda-datasets/resnet50-19c8e357.pth','--output',str(PRIOR/'source'),
        '--seed','1','--epochs','3','--freeze-backbone-bn']
    (PRIOR/'launch.json').write_text(json.dumps(dict(command=command,C=25,seed=1,epochs=3,
        freeze_encoder_bn=True,retention_weight=0,target_labels_used=False),indent=2))
    run(command,PRIOR/'console.log',env)
elif args.stage=='infer':
    import numpy as np
    from source_precision_capacity import estimate_source_cost_capacity
    output=PRIOR/'capacity.json'
    if output.exists():raise FileExistsError('Preserve capacity result')
    f=np.load(PRIOR/'source/features.npz')
    result,settings=estimate_source_cost_capacity(f['source'],f['source_labels'],f['target'])
    report=dict(task='OfficeHome Pr->Rw',seed=1,C=25,K=result['K'],converged=result['converged'],settings=settings,
        counts=result['counts'].tolist(),noise_count=result['noise_count'],history=result['history'],
        target_labels_used=False,semantic_unknown_count=None,source_prior='C25 source CE3, frozen encoder BN only during pretraining')
    output.write_text(json.dumps(report,indent=2,allow_nan=False));print('FROZENBN_TARGET_CAPACITY',json.dumps(report),flush=True)
else:
    estimate=json.loads((PRIOR/'capacity.json').read_text())
    if not estimate['converged'] or estimate['K']<1:raise ValueError('No positive inferred capacity; no forced K1')
    if args.stage=='estimated' and estimate['K']==4:raise ValueError('Estimated K equals fixed4; do not duplicate identical arm')
    k=4 if args.stage=='fixed4' else int(estimate['K'])
    output=ROOT/args.stage;output.mkdir(exist_ok=False)
    command=['/content/rta-py38/bin/python','-u','/content/train_legacy_task_entry.py','--task','officehome-pr2rw',
        '--shared_classes','25','--all_classes',str(25+k),'--virtual-clusters','29',
        '--source',str(DATA/'product_0-24_train_all.txt'),'--target',str(DATA/'real_world_0-64_test.txt'),
        '--data_dir',str(DATA),'--log_dir',str(output),'--name','seed1','--batch_size','64','--learning_rate','0.00005']
    env.update(RTA_SEED='1',RTA_EPOCHS='10',KONLY_SOURCE_PRIOR=str(PRIOR/'source/source-final.pt'),
        RTA_MODEL_PATH='/content/osda-datasets/resnet50-19c8e357.pth')
    (output/'launch.json').write_text(json.dumps(dict(arm=args.stage,K=k,C=25,Q=29,seed=1,epochs=10,source_epochs=3,
        source_prior=str(PRIOR/'source/source-final.pt'),command=command,
        training_changes='Pair only differs in K; BN freeze only source pretraining, original RTA train mode',
        caveat='Exploratory Q29/K4 and budget not confirmed author configuration; not strict reproduction',
        selection='Target-label metrics for oracle-best epoch only; not K or threshold selection'),indent=2))
    run(command,output/'console.log',env)
    history=[json.loads(row) for row in (output/'officehome-pr2rw_seed1/history.jsonl').read_text().splitlines()]
    if [row['epoch'] for row in history]!=list(range(1,11)):raise RuntimeError('Incomplete ten-epoch arm')
    print('FROZENBN_RTA_ARM_COMPLETE',args.stage,flush=True)
