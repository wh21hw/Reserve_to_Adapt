"""Four released-code warmup epochs on native L4; fixed-last, not oracle-best."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import torch
import torchvision
import faiss
import numpy as np

code = Path('/content/rta-l4-control-v1')
output = Path('/content/imp-runs/rta-space-warmup-l4-v1')
if output.exists():
    raise RuntimeError('Refusing overwrite')
if not torch.cuda.is_available() or 'L4' not in torch.cuda.get_device_name():
    raise RuntimeError('Expected L4')
output.mkdir(parents=True)
files = ['main.py','networks.py','utilities.py','centroid.py','data.py','domain_bus.py']
audit = dict(stage='released-code four warmup epochs in modern environment',
    upstream='8a4cd890b341a0dd4ed7c239533a1b6a9583fe8b', epochs=4, seed=1,
    selection='fixed last checkpoint; never use best.pt for feature inference',
    differences=['native framework/FAISS versions', 'trusted pinned legacy loader', 'epoch limit via env'],
    environment=dict(torch=torch.__version__, torchvision=torchvision.__version__,
                     faiss=faiss.__version__, numpy=np.__version__, gpu=torch.cuda.get_device_name()),
    code_sha256={name: hashlib.sha256((code/name).read_bytes()).hexdigest() for name in files})
(output/'audit.json').write_text(json.dumps(audit, indent=2))
command = [sys.executable, '-u', str(code/'main.py'), '--source', '/content/amazon_0-9_train_all.txt',
    '--target', '/content/webcam_0-9_20-30_test.txt', '--data_dir', '/content/osda-datasets/',
    '--log_dir', str(output)+'/', '--name', 'seed1', '--batch_size', '64',
    '--learning_rate', '0.00005', '--shared_classes', '10', '--all_classes', '12']
env = dict(os.environ, RTA_SEED='1', RTA_EPOCHS='4',
    RTA_MODEL_PATH='/content/osda-datasets/resnet50-19c8e357.pth',
    OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', PYTHONUNBUFFERED='1')
print('RTA_WARMUP_CONTROL', json.dumps(audit), flush=True)
with (output/'console.log').open('x') as log:
    process = subprocess.Popen(command, cwd=code, env=env, stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT, text=True)
    for line in process.stdout:
        print(line, end='', flush=True)
        log.write(line)
        log.flush()
    status = process.wait()
if status:
    raise RuntimeError('Warmup failed; preserve evidence')
history = [json.loads(row) for row in (output/'a2w_seed1/history.jsonl').read_text().splitlines()]
assert [row['epoch'] for row in history] == [1,2,3,4]
print('RTA_SPACE_WARMUP_DONE', str(output), flush=True)
