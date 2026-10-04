"""Declared serial seeds1/2/3, each 70 epochs, stop on failure."""
import json
import os
from pathlib import Path
import subprocess

root = Path('/content/imp-runs/fusion-imp-rta-v1')
for seed in (1, 2, 3):
    output = root / ('seed%d' % seed)
    prior = output/'source'
    summary = json.loads((prior/'summary.json').read_text())
    if not summary['complete'] or summary['epochs'] != 3:
        raise RuntimeError('Declared source stage is not complete')
    arm = output/'rta'
    arm.mkdir(exist_ok=False)
    environment = dict(os.environ, RTA_SEED=str(seed), FUSION_EPOCHS='70',
                       KONLY_SOURCE_PRIOR=str(prior/'source-final.pt'),
                       FUSION_FEATURES=str(prior/'features.npz'),
                       RTA_MODEL_PATH='/content/osda-datasets/resnet50-19c8e357.pth',
                       OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2')
    environment.pop('FUSION_FIXED_K', None)
    environment.pop('FUSION_BUILD_ONLY', None)
    command = ['/content/rta-py38/bin/python', '-u', '/content/train_fusion_imp_rta_entry.py',
               '--source', '/content/amazon_0-9_train_all.txt',
               '--target', '/content/webcam_0-9_20-30_test.txt',
               '--data_dir', '/content/osda-datasets/', '--log_dir', str(arm)+'/',
               '--batch_size', '64', '--learning_rate', '0.00005',
               '--shared_classes', '10', '--name', 'seed%d' % seed]
    (output/'launch.json').write_text(json.dumps(dict(seed=seed, epochs=70, source_epochs=3,
        design='fusion-v1', command=command, K='initial IMP estimate, fixed during RTA',
        V='IMP refreshed each epoch', head_initialization='original warm-end K-means',
        target_labels_for_structure=False, selection='Report all seeds, mean and sample SD; best is target-label oracle'), indent=2))
    print('FUSION_SEED_START', seed, flush=True)
    with (output/'rta-console.log').open('x') as stream:
        process = subprocess.Popen(command, env=environment, cwd='/content',
                                   stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for line in process.stdout:
            print(line, end='', flush=True)
            stream.write(line)
            stream.flush()
        if process.wait():
            raise RuntimeError('Fusion training failed; do not automatically retry')
    metrics = json.loads((arm/('a2w_seed%d' % seed)/'metrics.json').read_text())
    # Existing bridge reports completed epochs, not zero-based loop index.
    if metrics['epoch'] != 70:
        raise RuntimeError('Fusion run ended before 70 epochs')
    print('FUSION_SEED_COMPLETE', seed, flush=True)
print('FUSION_ALL_SEEDS_COMPLETE', flush=True)
