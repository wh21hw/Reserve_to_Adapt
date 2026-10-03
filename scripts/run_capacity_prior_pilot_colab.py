"""Run ONE predeclared two-epoch arm; never launch a chained experiment."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import torch

arm = 'uniform18'
if not torch.cuda.is_available() or 'L4' not in torch.cuda.get_device_name():
    raise RuntimeError('Expected L4')
preflight=json.loads(Path('/content/capacity-prior-preflight-v1.json').read_text())
assert preflight['shared_handoff_matches_proto18'] and preflight['optimizer_steps_executed']==0
code=Path('/content/rta-capacity-prior-pilot-v1')
output=Path('/content/imp-runs/rta-capacity-prior-pilot-v1')/arm
output.mkdir(parents=True, exist_ok=False)
audit=dict(arm=arm, seed=1, adaptation_epochs=2, fixed_final_epoch=6,
    comparison='Controlled post-warmup handoff, NOT uninterrupted official resume',
    checkpoint_selection='Fixed warm epoch4, never oracle-best',
    metrics='Fixed final primary; target-label oracle-best exploratory only',
    capacity_prior='Unknown logits + log(2/K); known unchanged; reference2 is original control capacity, not tuned',
    limitations='Group mass control only; flat entropy, per-slot CE, argmax remain capacity-dependent',
    gpu=torch.cuda.get_device_name(), code_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest()
        for p in code.glob('*.py')})
(output/'audit.json').write_text(json.dumps(audit,indent=2))
command=[sys.executable,'-u',str(code/'main.py'),
    '--source','/content/amazon_0-9_train_all.txt', '--target','/content/webcam_0-9_20-30_test.txt',
    '--data_dir','/content/osda-datasets/', '--log_dir',str(output)+'/', '--name','seed1',
    '--batch_size','64','--learning_rate','0.00005','--shared_classes','10',
    '--all_classes','28']
env=dict(os.environ,RTA_PILOT_ARM='proto18',RTA_CAPACITY_PRIOR='uniform_reference2',RTA_SEED='1',RTA_EPOCHS='6',
    RTA_MODEL_PATH='/content/osda-datasets/resnet50-19c8e357.pth',
    OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',PYTHONUNBUFFERED='1')
print('PILOT_START',json.dumps(audit),flush=True)
with (output/'console.log').open('x') as log:
    process=subprocess.Popen(command,cwd=code,env=env,stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,text=True)
    for line in process.stdout:
        print(line,end='',flush=True)
        log.write(line)
        log.flush()
    status=process.wait()
(output/'process-status.json').write_text(json.dumps(dict(exit_code=status)))
if status:
    raise RuntimeError('Pilot failed; preserve evidence and diagnose, no automatic retry')
history=[json.loads(row) for row in (output/'a2w_seed1/history.jsonl').read_text().splitlines()]
assert [row['epoch'] for row in history]==[5,6]
print('PILOT_ARM_DONE',arm,json.dumps(history[-1]),flush=True)
