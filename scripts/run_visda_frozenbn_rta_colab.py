"""Separated VisDA ten-epoch K arms; no forced K or automatic duplicate arm."""
import argparse
import json
import os
from pathlib import Path
import subprocess

parser = argparse.ArgumentParser()
parser.add_argument('arm', choices=['fixed2', 'estimated'])
args = parser.parse_args()
root = Path('/content/imp-runs/visda-frozenbn-capacity-10e-v1')
data = Path('/content/osda-visda-syn2real-v1')
source = json.loads((root/'source/summary.json').read_text())
if not source['complete'] or source['epochs'] != 3:
    raise ValueError('Expected completed shared source3 prior')
if args.arm == 'estimated':
    estimate = json.loads((root/'capacity.json').read_text())
    if estimate['target_labels_used'] or not estimate['converged'] or estimate['K'] < 1:
        raise ValueError('No converged positive capacity; preserve failure, never force K1')
    if estimate['K'] == 2:
        raise ValueError('Estimated K equals fixed K2; run one equivalent arm only')
    k = int(estimate['K'])
else:
    # Fixed control needs only the completed shared prior, not an inferred K.
    k = 2
class_map = Path('/content/visda-rta-class-map-v1.json')
mapping = json.loads(class_map.read_text())
if [mapping[str(i)] for i in [1, 2, 3, 6, 10, 11]] != ['bicycle', 'bus', 'car', 'motorcycle', 'train', 'truck']:
    raise ValueError('Class map conflicts with source prior')
output = root/args.arm
output.mkdir(exist_ok=False)
prior = str(root/'source/source-final.pt')
command = ['/content/rta-py38/bin/python', '-u', '/content/train_visda_frozenbn_rta_entry.py',
    '--task', 'visda-synthetic2real', '--class-map', str(class_map), '--accept-unresolved-visda-resnet',
    '--shared_classes', '6', '--all_classes', str(6+k), '--virtual-clusters', '8',
    '--source', str(data/'source-known-6.txt'), '--target', str(data/'target-real-12.txt'),
    '--data_dir', str(data), '--log_dir', str(output), '--name', 'seed1',
    '--batch_size', '64', '--learning_rate', '0.00005']
env = dict(os.environ, OPENBLAS_NUM_THREADS='2', OMP_NUM_THREADS='2', RTA_SEED='1',
    RTA_EPOCHS='10', RTA_FREEZE_ENCODER_BN='1', KONLY_SOURCE_PRIOR=prior,
    RTA_MODEL_PATH='/content/osda-datasets/resnet50-19c8e357.pth')
env.pop('RTA_CLUSTER_LABELS', None)
manifest = dict(task='VisDA Synthetic->Real', arm=args.arm, C=6, K=k, Q=8, seed=1,
    epochs=10, source_epochs=3, source_prior=prior, command=command, backbone='ResNet50',
    RTA_FREEZE_ENCODER_BN=True, capacity_result=str(root/'capacity.json'),
    research_change='Only integer classifier K differs; original initialization, losses, selection and prediction',
    target_labels_for_structure=False, selection='Target-label oracle best epoch; short-course single seed',
    caveat='User-specified ResNet; original VisDA epoch budget and table/text backbone conflict unresolved; Q held8 rather than paper K coupling')
(output/'launch.json').write_text(json.dumps(manifest, indent=2))
with (output/'console.log').open('x') as log:
    p = subprocess.Popen(command, cwd='/content/rta-legacy-l4-bridge-v1', env=env,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for line in p.stdout:
        print(line, end='', flush=True)
        log.write(line)
        log.flush()
    if p.wait():
        raise RuntimeError('VisDA arm failed; preserve evidence, no automatic training retry')
history = [json.loads(line) for line in
    (output/'visda-synthetic2real_seed1/history.jsonl').read_text().splitlines()]
if [row['epoch'] for row in history] != list(range(1, 11)):
    raise RuntimeError('Incomplete declared ten-epoch budget')
print('VISDA_FROZENBN_RTA_ARM_COMPLETE', args.arm, flush=True)
