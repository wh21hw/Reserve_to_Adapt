"""Final predeclared baseline seed; no changes chosen from prior results."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import torch

root = Path('/content/rta_multitask_baseline_v1')
expected = {'main.py': '683c35e76a89ad1588f35dea83e3ecbc6f9a8af5bb3250e9e3de1cbfc3c7a148',
            'utilities.py': '6ba276bb2d0076a4d64cde811496d2dfb71b330c8e1660add02d8fbd42e636e7'}
for name, digest in expected.items():
    assert hashlib.sha256((root / name).read_bytes()).hexdigest() == digest
assert torch.cuda.is_available() and 'L4' in torch.cuda.get_device_name()
for seed in (1, 2):
    prior = json.loads(Path(f'/content/imp-runs/rta-multitask-baseline-v1/office31-a2w_seed{seed}/audit-summary.json').read_text())
    assert prior['epochs_verified'] == 70 and prior['seed'] == seed
protocol = json.loads(Path('/content/imp-runs/multitask-preflight-v1/office31-a2w_seed1/protocol.json').read_text())
for name, role in (('amazon_0-9_train_all.txt', 'source'), ('webcam_0-9_20-30_test.txt', 'target')):
    assert hashlib.sha256((Path('/content') / name).read_bytes()).hexdigest() == protocol[role]['list_sha256']
environment = dict(os.environ, RTA_SEED='3', RTA_EPOCHS='70',
                   RTA_MODEL_PATH='/content/osda-datasets/resnet50-19c8e357.pth',
                   OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2')
command = [sys.executable, '-u', str(root / 'main.py'), '--task', 'office31-a2w',
           '--source', '/content/amazon_0-9_train_all.txt',
           '--target', '/content/webcam_0-9_20-30_test.txt',
           '--data_dir', '/content/osda-datasets', '--virtual-clusters', '20',
           '--log_dir', '/content/imp-runs/rta-multitask-baseline-v1', '--name', 'seed3']
with Path('/content/rta-multitask-a2w-seed3-launch-v1.json').open('x') as stream:
    json.dump(dict(command=command, code_sha256=expected, seed=3, epochs=70,
                   loss_protocol='published code, not paper-equation reimplementation',
                   checkpoint_selection='fixed final; oracle-best only separate diagnostic',
                   storage='temporary /content; Drive full'), stream, indent=2)
with Path('/content/rta-multitask-a2w-seed3-console-v1.log').open('x') as stream:
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                               text=True, cwd=root, env=environment)
    print('BASELINE_WORKER_PID', process.pid, flush=True)
    for line in process.stdout:
        print(line, end='', flush=True)
        stream.write(line)
        stream.flush()
    status = process.wait()
if status:
    raise RuntimeError(f'Baseline failed ({status}); preserve evidence, do not rerun automatically')
print('BASELINE_SEED3_WORKER_COMPLETE', flush=True)
