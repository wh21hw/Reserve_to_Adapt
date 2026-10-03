"""One 70-epoch environment bridge, original T4 code/settings, diagnostic seed3."""
import json
import os
from pathlib import Path
import subprocess

root = Path('/content/rta-legacy-l4-bridge-v1')
output = Path('/content/imp-runs/rta-legacy-l4-bridge-v1')
compatibility = json.loads(Path('/content/rta-legacy-l4-compatibility-v1.json').read_text())
if not compatibility['finite_gradients'] or 'L4' not in compatibility['gpu']:
    raise RuntimeError('Legacy L4 compatibility check missing')
if output.exists():
    raise RuntimeError('Refusing to overwrite existing bridge experiment')
output.mkdir(parents=True)
command = ['/content/rta-py38/bin/python', '-u', str(root/'main.py'),
           '--source', '/content/amazon_0-9_train_all.txt',
           '--target', '/content/webcam_0-9_20-30_test.txt',
           '--data_dir', '/content/osda-datasets/', '--log_dir', str(output)+'/',
           '--batch_size', '64', '--learning_rate', '0.00005',
           '--shared_classes', '10', '--all_classes', '12', '--name', 'seed3']
environment = dict(os.environ, RTA_SEED='3', PYTHONUNBUFFERED='1',
                   RTA_MODEL_PATH='/content/osda-datasets/resnet50-19c8e357.pth',
                   OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2')
(output/'launch.json').write_text(json.dumps(dict(command=command, seed=3, epochs=70,
    K=2, virtual_clusters=20, environment=compatibility,
    purpose='Diagnose T4/L4 seed3 discrepancy with T4-era dependencies and T4 code; not a three-seed result',
    selection='final and target-label oracle best reported separately'), indent=2))
with (output/'seed3-console.log').open('x') as stream:
    process = subprocess.Popen(command, cwd=root, env=environment, stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT, text=True)
    print('LEGACY_BASELINE_PID', process.pid, flush=True)
    for line in process.stdout:
        print(line, end='', flush=True)
        stream.write(line)
        stream.flush()
    status = process.wait()
if status:
    raise RuntimeError('Legacy baseline failed; preserve log, no automatic training retry')
print('LEGACY_L4_SEED3_COMPLETE', flush=True)
