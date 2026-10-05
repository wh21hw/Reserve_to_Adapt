"""Prepare one K8 arm; explicit --run is required for new GPU training.

Uses the completed coefficient=1 arm as the common settings reference, but
does not reuse its K2 metrics as a K8 control. Both K8 arms must be run.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('arm', choices=['argmax', 'identity'])
mode = parser.add_mutually_exclusive_group()
mode.add_argument('--run', action='store_true')
mode.add_argument('--build-only', action='store_true', help='Compile the actual patched entry without a model')
args = parser.parse_args()
reference = Path('/content/imp-runs/a2w-unknown-ce-10e-v1')
base = json.loads((reference/'control/launch.json').read_text())
probe = reference/'identity-reconciliation-probe-v1'
settings = json.loads((probe/'settings.json').read_text())
artifact = probe/'clusters.npz'
if (base['C'], base['K'], base['Q'], base['seed'], base['epochs'], base['unknown_ce_weight']) != (10,2,20,3,10,1):
    raise ValueError('Unexpected common reference settings')
if settings['target_labels_used'] or settings['K'] != 8 or settings['occupied_clusters'] != 18:
    raise ValueError('Use only the fixed label-blind reconciliation artifact')
target_path = base['command'][base['command'].index('--target')+1]
paths = [line.rsplit(None,1)[0] for line in Path(target_path).read_text().splitlines() if line.strip()]
with np.load(artifact, allow_pickle=False) as data:
    if data['target_paths'].tolist() != paths or data['assignments'].shape != (len(paths),):
        raise ValueError('Artifact/input sample ordering differs')
    if not np.array_equal(np.unique(data['assignments'][data['assignments']>=10]), np.arange(10,18)):
        raise ValueError('Expected all eight candidate identities; never force K')
if not Path(base['source_prior']).is_file():
    raise FileNotFoundError('Restore the existing shared source3 checkpoint; do not retrain it')
root = Path('/content/imp-runs/a2w-reconciled-identity-10e-v1')
output = root/args.arm
command = list(base['command'])
command[command.index('/content/train_a2w_unknown_ce_entry.py')] = '/content/train_legacy_task_entry.py'
command[command.index('--all_classes')+1] = '18'
command[command.index('--log_dir')+1] = str(output)
environment = dict(RTA_SEED='3', RTA_EPOCHS='10', RTA_FREEZE_ENCODER_BN='1',
    KONLY_SOURCE_PRIOR=base['source_prior'], RTA_MODEL_PATH='/content/osda-datasets/resnet50-19c8e357.pth',
    OPENBLAS_NUM_THREADS='2', OMP_NUM_THREADS='2', PYTHONPATH='/content')
if args.arm == 'identity':
    environment['RTA_CLUSTER_LABELS'] = str(artifact)
manifest = dict(base, arm=args.arm, K=8, command=command, environment=environment,
    cluster_artifact=str(artifact), capacity_source=str(probe/'settings.json'),
    research_change='K8 argmax versus K8 fixed cluster identity labels on original selected candidates only',
    purpose='Separate capacity from identity transport; K2 comparison is background/capacity context',
    initialization='Same source3 seed3, original warm-end K-means; no IMP head weight initialization',
    selection='Previously selected seed3; fixed full ten epochs; oracle best is descriptive, not a search reward',
    caveat='Remaining clusters may contain shifted known samples; offline identities can become stale. Not semantic discovery proof or K-only method.')
env = dict(os.environ, **environment)
for key in ('RTA_CLUSTER_LABELS', 'RTA_UNKNOWN_CE_WEIGHT', 'LEGACY_TASK_BUILD_ONLY'):
    if key not in environment:
        env.pop(key, None)
if args.build_only:
    env['LEGACY_TASK_BUILD_ONLY'] = '1'
    subprocess.run(command, cwd='/content/rta-legacy-l4-bridge-v1', env=env, check=True)
    print('A2W_K8_BUILD_ONLY_COMPLETE', args.arm, flush=True)
    raise SystemExit(0)
if not args.run:
    print('A2W_K8_PREPARED_NOT_STARTED', json.dumps(manifest), flush=True)
    raise SystemExit(0)
output.mkdir(parents=True, exist_ok=False)
(output/'launch.json').write_text(json.dumps(manifest, indent=2))
with (output/'console.log').open('x') as log:
    process = subprocess.Popen(command, cwd='/content/rta-legacy-l4-bridge-v1', env=env,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for line in process.stdout:
        print(line,end='',flush=True)
        log.write(line); log.flush()
    status = process.wait()
(output/'process-status.json').write_text(json.dumps(dict(exit_code=status)))
if status:
    raise RuntimeError('Failed arm; preserve evidence, no automatic retry')
history = [json.loads(line) for line in (output/'office31-a2w_seed3/history.jsonl').read_text().splitlines()]
if [row['epoch'] for row in history] != list(range(1,11)):
    raise RuntimeError('Incomplete declared ten-epoch budget')
print('A2W_K8_ARM_COMPLETE', args.arm, flush=True)
