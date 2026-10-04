"""Prepare the three declared C-only priors; no RTA training in this stage."""
import json
from pathlib import Path
import subprocess

root = Path('/content/imp-runs/fusion-imp-rta-v1')
root.mkdir(exist_ok=False)
for seed in (1, 2, 3):
    output = root / ('seed%d' % seed)
    output.mkdir()
    command = ['/content/rta-py38/bin/python', '-u', '/content/train_source_prior_konly.py',
               '--code-root', '/content/rta-legacy-l4-bridge-v1',
               '--source', '/content/amazon_0-9_train_all.txt',
               '--target', '/content/webcam_0-9_20-30_test.txt',
               '--data-root', '/content/osda-datasets',
               '--weights', '/content/osda-datasets/resnet50-19c8e357.pth',
               '--output', str(output/'source'), '--seed', str(seed), '--epochs', '3']
    with (output/'source-console.log').open('x') as stream:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for line in process.stdout:
            print(line, end='', flush=True)
            stream.write(line)
            stream.flush()
        if process.wait():
            raise RuntimeError('Source stage failed; stop and preserve evidence')
    print('FUSION_PRIOR_COMPLETE', seed, flush=True)
print('FUSION_ALL_PRIORS_COMPLETE', flush=True)
