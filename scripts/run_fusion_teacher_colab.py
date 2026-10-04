"""Declared seed1 K7 ten-epoch geometric pseudo-label arm; stop on failure."""
import json
import os
from pathlib import Path
import subprocess

prior = Path('/content/imp-runs/fusion-imp-rta-v1/seed1/source')
output = Path('/content/imp-runs/fusion-unknown-teacher-v1')
environment = dict(os.environ,RTA_SEED='1',FUSION_EPOCHS='10',
                   FUSION_FIXED_K='7',FUSION_DIAGNOSTICS='1',
                   KONLY_SOURCE_PRIOR=str(prior/'source-final.pt'),
                   FUSION_FEATURES=str(prior/'features.npz'),
                   RTA_MODEL_PATH='/content/osda-datasets/resnet50-19c8e357.pth',
                   OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
environment.pop('FUSION_BUILD_ONLY',None)
command = ['/content/rta-py38/bin/python','-u','/content/train_fusion_teacher_entry.py',
           '--source','/content/amazon_0-9_train_all.txt',
           '--target','/content/webcam_0-9_20-30_test.txt',
           '--data_dir','/content/osda-datasets/','--log_dir',str(output/'rta')+'/',
           '--batch_size','64','--learning_rate','0.00005','--shared_classes','10','--name','seed1']
check = subprocess.run(command,env=dict(environment,FUSION_BUILD_ONLY='1'),cwd='/content',
                       stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
print(check.stdout,flush=True)
check.check_returncode()
# One focused teacher interface check, not a second training/smoke run.
test = '''import torch
from unknown_prototype_teacher import UnknownPrototypeTeacher
t = UnknownPrototypeTeacher(2)
t.initialize(torch.tensor([[0.,0.],[2.,2.]]),'/content')
labels = t.assign(torch.tensor([[0.1,0.],[1.9,2.]]),torch.zeros(2,4))
assert labels.tolist()==[2,3] and bool(torch.isfinite(t.centers).all())
print('TEACHER_INTERFACE_PASS')
'''
checked = subprocess.run(['/content/rta-py38/bin/python','-c',test],env=environment,cwd='/content',
                         stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
print(checked.stdout,flush=True)
checked.check_returncode()
output.mkdir(exist_ok=False)
(output/'launch.json').write_text(json.dumps(dict(seed=1,K=7,epochs=10,source_epochs=3,
    command=command,teacher_momentum=0.9,change='only unknown pseudo-label assignment and its EMA bank',
    comparison='/content/imp-runs/fusion-capacity-diagnostic-v1/K7',
    source_prior=str(prior),target_labels_for_teacher=False,
    selection='posthoc seed1; best epoch target oracle'),indent=2))
with (output/'console.log').open('x') as stream:
    process = subprocess.Popen(command,env=environment,cwd='/content',
                               stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
    for line in process.stdout:
        print(line,end='',flush=True)
        stream.write(line)
        stream.flush()
    if process.wait():
        raise RuntimeError('Teacher training failed; preserve evidence, no automatic retry')
folder = output/'rta/a2w_seed1'
history = [json.loads(line) for line in (folder/'history.jsonl').read_text().splitlines()]
if [row['epoch'] for row in history] != list(range(1,11)):
    raise RuntimeError('Teacher arm did not complete ten epochs')
teacher = [json.loads(line) for line in (folder/'teacher-history.jsonl').read_text().splitlines()]
if [row['epoch'] for row in teacher] != list(range(5,11)):
    raise RuntimeError('Missing teacher postwarm diagnostics')
print('UNKNOWN_TEACHER_COMPLETE',flush=True)
