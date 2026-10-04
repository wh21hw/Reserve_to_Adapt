"""One ten-epoch capacity arm, same seed1 prior; never overwrite prior results."""
import json
import os
from pathlib import Path
import subprocess

k = int(os.environ.get('CAPACITY_CONTROL_K', '2'))
if k not in (2, 7):
    raise ValueError('Only predeclared K2/K7 arms')
prior = Path('/content/imp-runs/fusion-imp-rta-v1/seed1/source')
summary = json.loads((prior/'summary.json').read_text())
if not summary['complete'] or summary['epochs'] != 3:
    raise RuntimeError('Source prior incomplete')
output = Path('/content/imp-runs/fusion-capacity-diagnostic-v1')/('K%d' % k)
environment = dict(os.environ, RTA_SEED='1', FUSION_EPOCHS='10',
                   FUSION_FIXED_K=str(k), FUSION_DIAGNOSTICS='1',
                   KONLY_SOURCE_PRIOR=str(prior/'source-final.pt'),
                   FUSION_FEATURES=str(prior/'features.npz'),
                   RTA_MODEL_PATH='/content/osda-datasets/resnet50-19c8e357.pth',
                   OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2')
environment.pop('FUSION_BUILD_ONLY', None)
command = ['/content/rta-py38/bin/python','-u','/content/train_fusion_imp_rta_entry.py',
           '--source','/content/amazon_0-9_train_all.txt',
           '--target','/content/webcam_0-9_20-30_test.txt',
           '--data_dir','/content/osda-datasets/','--log_dir',str(output/'rta')+'/',
           '--batch_size','64','--learning_rate','0.00005','--shared_classes','10','--name','seed1']
# One focused construction check, no SGD; do not create run directory on failure.
check = subprocess.run(command, env=dict(environment,FUSION_BUILD_ONLY='1'),cwd='/content',
                       stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
print(check.stdout,flush=True)
check.check_returncode()
output.mkdir(parents=True,exist_ok=False)
(output/'launch.json').write_text(json.dumps(dict(K=k,seed=1,epochs=10,source_epochs=3,
    command=command,source_prior=str(prior),design='capacity control; unchanged IMP virtual updates',
    target_labels_for_diagnostics=False,selection='seed1 chosen post hoc from fusion v1; best epoch target oracle'),indent=2))
with (output/'console.log').open('x') as stream:
    process = subprocess.Popen(command, env=environment,cwd='/content',
                               stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
    for line in process.stdout:
        print(line,end='',flush=True)
        stream.write(line)
        stream.flush()
    if process.wait():
        raise RuntimeError('Capacity diagnostic failed; no automatic retry')
folder = output/'rta/a2w_seed1'
history = [json.loads(line) for line in (folder/'history.jsonl').read_text().splitlines()]
if [row['epoch'] for row in history] != list(range(1,11)):
    raise RuntimeError('Incomplete ten epoch arm')
print('CAPACITY_DIAGNOSTIC_COMPLETE',k,flush=True)
