"""One explicitly selected ten-epoch arm, with a shared C-only source prior."""
import argparse
import json
import os
from pathlib import Path
import subprocess

parser = argparse.ArgumentParser()
parser.add_argument('arm', choices=['control', 'off'])
args = parser.parse_args()
root = Path('/content/imp-runs/a2w-unknown-ce-10e-v1')
prior = root / 'source/source-final.pt'
summary = json.loads((root / 'source/summary.json').read_text())
if not summary['complete'] or (summary['known_classes'], summary['epochs']) != (10, 3):
    raise ValueError('Require the completed shared C10 source3 prior')
output = root / args.arm
output.mkdir(exist_ok=False)
data = Path('/content/osda-office31-a2w-v1')
command = ['/content/rta-py38/bin/python', '-u', '/content/train_a2w_unknown_ce_entry.py',
    '--task', 'office31-a2w', '--shared_classes', '10', '--all_classes', '12',
    '--virtual-clusters', '20', '--source', str(data/'amazon_0-9_train_all.txt'),
    '--target', str(data/'webcam_0-9_20-30_test.txt'), '--data_dir', str(data),
    '--log_dir', str(output), '--name', 'seed3', '--batch_size', '64',
    '--learning_rate', '0.00005']
weight = 1 if args.arm == 'control' else 0
env = dict(os.environ, RTA_SEED='3', RTA_EPOCHS='10', RTA_FREEZE_ENCODER_BN='1',
    RTA_UNKNOWN_CE_WEIGHT=str(weight), KONLY_SOURCE_PRIOR=str(prior),
    RTA_MODEL_PATH='/content/osda-datasets/resnet50-19c8e357.pth',
    OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2')
env.pop('RTA_CLUSTER_LABELS', None)
env.pop('LEGACY_TASK_BUILD_ONLY', None)
manifest = dict(task='Office31 A->W', arm=args.arm, C=10, K=2, Q=20,
    seed=3, epochs=10, source_epochs=3, source_prior=str(prior), command=command,
    unknown_ce_weight=weight, backbone='ResNet50', encoder_bn='frozen running stats, affine trainable; head BN unchanged',
    research_change='Post-warmup unknown pseudo-label CE coefficient 1 versus 0 only',
    initialization='Same C-only source3 prior and original warm-end K-means',
    selection='Previously selected seed3; target-oracle best epoch descriptive; final/all epochs also reported',
    caveat='Mechanism ablation on A->W, not proof of VisDA failure cause, not K-only or full paper reproduction')
(output/'launch.json').write_text(json.dumps(manifest, indent=2))
print('A2W_CE_ARM_START', json.dumps(manifest), flush=True)
with (output/'console.log').open('x') as log:
    process = subprocess.Popen(command, cwd='/content/rta-legacy-l4-bridge-v1', env=env,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for line in process.stdout:
        print(line, end='', flush=True)
        log.write(line); log.flush()
    status = process.wait()
(output/'process-status.json').write_text(json.dumps(dict(exit_code=status)))
if status:
    raise RuntimeError('Failed arm; preserve evidence, no automatic retry')
rows = [json.loads(line) for line in (output/'office31-a2w_seed3/history.jsonl').read_text().splitlines()]
if [row['epoch'] for row in rows] != list(range(1, 11)):
    raise RuntimeError('Incomplete declared ten-epoch budget')
print('A2W_CE_ARM_COMPLETE', args.arm, flush=True)
