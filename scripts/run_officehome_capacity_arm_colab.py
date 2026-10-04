"""One declared matched OfficeHome ten-epoch arm, no automatic retry."""
import json
import os
from pathlib import Path
import subprocess

arm=os.environ['OFFICEHOME_CAPACITY_ARM']
if arm not in ('fixed4','estimated'):
    raise ValueError('Only the declared pair is allowed')
prior=Path('/content/imp-runs/source-precision-officehome-v1/seed1/source')
estimate=json.loads((prior.parent/'source-cost-target-capacity.json').read_text())
if not estimate['converged'] or estimate['K']<1:
    raise ValueError('Do not silently force a positive inferred K')
k=4 if arm=='fixed4' else int(estimate['K'])
output=Path('/content/imp-runs/officehome-capacity-10e-v1')/arm
output.mkdir(parents=True,exist_ok=False)
command=['/content/rta-py38/bin/python','-u','/content/train_legacy_task_entry.py',
    '--task','officehome-pr2rw','--shared_classes','25','--all_classes',str(25+k),
    '--virtual-clusters','29','--source','/content/osda-officehome-pr2rw-v1/product_0-24_train_all.txt',
    '--target','/content/osda-officehome-pr2rw-v1/real_world_0-64_test.txt',
    '--data_dir','/content/osda-officehome-pr2rw-v1','--log_dir',str(output),
    '--name','seed1','--batch_size','64','--learning_rate','0.00005']
env=dict(os.environ,RTA_SEED='1',RTA_EPOCHS='10',KONLY_SOURCE_PRIOR=str(prior/'source-final.pt'),
    RTA_MODEL_PATH='/content/osda-datasets/resnet50-19c8e357.pth',OPENBLAS_NUM_THREADS='2',OMP_NUM_THREADS='2')
(output/'launch.json').write_text(json.dumps(dict(arm=arm,K=k,C=25,Q=29,seed=1,epochs=10,
    source_epochs=3,source_prior=str(prior/'source-final.pt'),command=command,
    training_changes='Only K differs between arms; original task-adapted RTA objectives and prediction',
    caveat='Exploratory; OfficeHome Q29 and budget not confirmed author configuration',
    selection='target labels only evaluate oracle-best epoch, not estimate K'),indent=2))
with (output/'console.log').open('x') as stream:
    process=subprocess.Popen(command,cwd='/content/rta-legacy-l4-bridge-v1',env=env,
        stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
    for line in process.stdout:
        print(line,end='',flush=True); stream.write(line); stream.flush()
    if process.wait():
        raise RuntimeError('Capacity arm failed; no automatic retry')
history=[json.loads(line) for line in (output/'officehome-pr2rw_seed1/history.jsonl').read_text().splitlines()]
if [row['epoch'] for row in history]!=list(range(1,11)):
    raise RuntimeError('Expected exactly ten complete epochs')
print('OFFICEHOME_CAPACITY_ARM_COMPLETE',arm,flush=True)
