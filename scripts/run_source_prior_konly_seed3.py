"""Shared C-class source prior for the selected baseline seed3, ONE stage."""
import os
from pathlib import Path
import subprocess

output = Path('/content/imp-runs/konly-source-prior-v1/seed3')
if output.exists():
    raise RuntimeError('Existing source prior; do not overwrite or repeat training')
command = ['/content/rta-py38/bin/python', '-u', '/content/train_source_prior_konly.py',
           '--code-root', '/content/rta-legacy-l4-bridge-v1',
           '--source', '/content/amazon_0-9_train_all.txt',
           '--target', '/content/webcam_0-9_20-30_test.txt',
           '--data-root', '/content/osda-datasets',
           '--weights', '/content/osda-datasets/resnet50-19c8e357.pth',
           '--output', str(output), '--seed', '3', '--epochs', '3']
environment = dict(os.environ, OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2')
with Path('/content/konly-source-prior-seed3-console-v1.log').open('x') as log:
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
