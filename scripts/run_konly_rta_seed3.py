"""One arm per invocation; common source prior, original 70-epoch RTA."""
import json
import os
from pathlib import Path
import subprocess

arm = os.environ.get('KONLY_ARM', 'fixed2')
estimate = json.loads(Path('/content/imp-runs/konly-estimate-v1/seed3-mobile/estimate.json').read_text())
if arm not in ('fixed2', 'estimated'):
    raise ValueError('Unknown arm')
K = 2 if arm == 'fixed2' else int(estimate['K'])
if K <= 0:
    raise ValueError('No positive estimated unknown capacity')
output = Path('/content/imp-runs/konly-rta-v1') / arm
output.mkdir(parents=True, exist_ok=False)
prior = '/content/imp-runs/konly-source-prior-v2/seed3/source-final.pt'
command = ['/content/rta-py38/bin/python', '-u', '/content/train_konly_rta_entry.py',
           '--source', '/content/amazon_0-9_train_all.txt',
           '--target', '/content/webcam_0-9_20-30_test.txt',
           '--data_dir', '/content/osda-datasets/', '--log_dir', str(output)+'/',
           '--batch_size', '64', '--learning_rate', '0.00005',
           '--shared_classes', '10', '--all_classes', str(10+K), '--name', 'seed3']
environment = dict(os.environ, RTA_SEED='3', PYTHONUNBUFFERED='1',
                   KONLY_SOURCE_PRIOR=prior,
                   RTA_MODEL_PATH='/content/osda-datasets/resnet50-19c8e357.pth',
                   OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2')
(output/'launch.json').write_text(json.dumps(dict(command=command, seed=3, epochs=70,
    K=K, virtual_clusters=20, source_prior=prior, arm=arm,
    selection='Posthoc selected seed3; report final and target-label oracle best separately',
    change='Both arms share source initialization; only K differs'), indent=2))
with (output/'seed3-console.log').open('x') as stream:
    process = subprocess.Popen(command, cwd='/content/rta-legacy-l4-bridge-v1',
                               env=environment, stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT, text=True)
    print('KONLY_WORKER_PID', process.pid, 'arm', arm, 'K', K, flush=True)
    for line in process.stdout:
        print(line, end='', flush=True)
        stream.write(line)
        stream.flush()
    status = process.wait()
if status:
    raise RuntimeError('K-only arm failed; preserve the log')
print('KONLY_ARM_COMPLETE', arm, flush=True)
