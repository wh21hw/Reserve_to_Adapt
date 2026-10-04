"""One shared C25 source prior. Run only after scoped data restoration succeeds."""
import json
import os
from pathlib import Path
import subprocess

data = Path('/content/osda-officehome-pr2rw-v1')
if not (data/'restoration.json').is_file():
    raise RuntimeError('Restore inputs before launching the source stage')
output = Path('/content/imp-runs/source-precision-officehome-v1/seed1')
output.mkdir(parents=True, exist_ok=False)
command = ['/content/rta-py38/bin/python', '-u', '/content/train_source_prior_konly.py',
           '--code-root', '/content/rta-legacy-l4-bridge-v1',
           '--source', str(data/'product_0-24_train_all.txt'),
           '--target', str(data/'real_world_0-64_test.txt'), '--data-root', str(data),
           '--weights', '/content/osda-datasets/resnet50-19c8e357.pth',
           '--output', str(output/'source'), '--seed', '1', '--epochs', '3']
(output/'launch.json').write_text(json.dumps(dict(task='OfficeHome Pr->Rw', seed=1,
    source_epochs=3, C=25, command=command, source_only=True,
    target_labels_used=False, RTA_started=False, K_estimator='source-precision-capacity-v1',
    caveat='Source budget transferred from declared A->W design; not paper baseline replication'), indent=2))
with (output/'source-console.log').open('x') as stream:
    process = subprocess.Popen(command, cwd='/content',
        env=dict(os.environ, OPENBLAS_NUM_THREADS='2', OMP_NUM_THREADS='2'),
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for line in process.stdout:
        print(line, end='', flush=True)
        stream.write(line)
        stream.flush()
    if process.wait():
        raise RuntimeError('OfficeHome source stage failed; preserve evidence')
print('PRECISION_OFFICEHOME_SOURCE_COMPLETE', flush=True)
