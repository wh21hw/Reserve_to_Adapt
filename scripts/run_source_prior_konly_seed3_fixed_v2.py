"""Repair only the scheduler callback name; preserve v1 failed evidence."""
import os
from pathlib import Path
import subprocess

original = Path('/content/train_source_prior_konly.py').read_text()
old = 'schedule = lambda step, lr: inverseDecaySheduler(step, lr, gamma=10, power=.75, max_iter=10000)'
new = 'schedule = lambda step, initial_lr: inverseDecaySheduler(step, initial_lr, gamma=10, power=.75, max_iter=10000)'
if original.count(old) != 1:
    raise RuntimeError('Unexpected original callback; do not patch blindly')
worker = Path('/content/train_source_prior_konly_fixed_v2.py')
with worker.open('x') as stream:
    stream.write(original.replace(old, new))
output = Path('/content/imp-runs/konly-source-prior-v2/seed3')
if output.exists():
    raise RuntimeError('Existing source prior; do not overwrite')
command = ['/content/rta-py38/bin/python', '-u', str(worker),
           '--code-root', '/content/rta-legacy-l4-bridge-v1',
           '--source', '/content/amazon_0-9_train_all.txt',
           '--target', '/content/webcam_0-9_20-30_test.txt',
           '--data-root', '/content/osda-datasets',
           '--weights', '/content/osda-datasets/resnet50-19c8e357.pth',
           '--output', str(output), '--seed', '3', '--epochs', '3']
environment = dict(os.environ, OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2')
with Path('/content/konly-source-prior-seed3-console-v2.log').open('x') as log:
    process = subprocess.Popen(command, env=environment, stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT, text=True)
    print('SOURCE_PRIOR_WORKER_PID', process.pid, flush=True)
    for line in process.stdout:
        print(line, end='', flush=True)
        log.write(line)
        log.flush()
    status = process.wait()
if status:
    raise RuntimeError('Source prior failed; preserve the log')
