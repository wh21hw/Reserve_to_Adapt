"""One supported-capacity arm; change only count inference, not virtual updates."""
import json
import os
from pathlib import Path
import subprocess

prior = Path('/content/imp-runs/fusion-imp-rta-v1/seed1/source')
inference = '''import json,torch
torch.set_num_threads(2)
from fusion_imp_rta import initial_structure
r,s=initial_structure('/content/imp-runs/fusion-imp-rta-v1/seed1/source/features.npz',10)
mass=r['effective_counts'][10:]
minimum=s['prior_strength']
keep=mass>=minimum
print(json.dumps(dict(raw_K=r['candidate_count'],supported_K=int(keep.sum()),minimum_support=minimum,
                     candidate_effective_counts=mass.tolist(),keep=keep.tolist(),target_labels_used=False)))
'''
checked = subprocess.run(['/content/rta-py38/bin/python','-c',inference],cwd='/content',
                         stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
print(checked.stdout,flush=True)
checked.check_returncode()
settings = json.loads(checked.stdout.strip().splitlines()[-1])
if settings['supported_K'] < 1:
    raise RuntimeError('No supported capacity; do not force K=1')
output = Path('/content/imp-runs/fusion-supported-capacity-v1')
environment = dict(os.environ,RTA_SEED='1',FUSION_EPOCHS='10',
                   FUSION_FIXED_K=str(settings['supported_K']),FUSION_DIAGNOSTICS='1',
                   KONLY_SOURCE_PRIOR=str(prior/'source-final.pt'),
                   FUSION_FEATURES=str(prior/'features.npz'),
                   RTA_MODEL_PATH='/content/osda-datasets/resnet50-19c8e357.pth',
                   OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
environment.pop('FUSION_BUILD_ONLY',None)
command = ['/content/rta-py38/bin/python','-u','/content/train_fusion_imp_rta_entry.py',
           '--source','/content/amazon_0-9_train_all.txt',
           '--target','/content/webcam_0-9_20-30_test.txt',
           '--data_dir','/content/osda-datasets/','--log_dir',str(output/'rta')+'/',
           '--batch_size','64','--learning_rate','0.00005','--shared_classes','10','--name','seed1']
output.mkdir(exist_ok=False)
(output/'launch.json').write_text(json.dumps(dict(seed=1,epochs=10,source_epochs=3,
    command=command,count_inference=settings,change='only K counting; virtual directions remain raw IMP',
    unknown_pseudo_labels='original self-label',prediction='original flat argmax',
    caveat='minimum support heuristic borrowing source prior strength; not exact DP posterior',
    selection='posthoc seed1; best epoch target oracle'),indent=2))
with (output/'console.log').open('x') as stream:
    process = subprocess.Popen(command,env=environment,cwd='/content',
                               stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
    for line in process.stdout:
        print(line,end='',flush=True)
        stream.write(line)
        stream.flush()
    if process.wait():
        raise RuntimeError('Supported capacity failed; no automatic retry')
folder = output/'rta/a2w_seed1'
history = [json.loads(line) for line in (folder/'history.jsonl').read_text().splitlines()]
if [row['epoch'] for row in history] != list(range(1,11)):
    raise RuntimeError('Incomplete supported-capacity arm')
config_file = folder/'config.json'
config = json.loads(config_file.read_text())
config.update(K_policy='supported initial IMP estimate fixed',count_inference=settings)
config_file.write_text(json.dumps(config,indent=2))
print('SUPPORTED_CAPACITY_COMPLETE',settings['supported_K'],flush=True)
