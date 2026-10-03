"""Start the separately declared alternating arm only after one-shot pair completes."""
import json
import os
from pathlib import Path
import subprocess

for arm in ('fixed2', 'estimated'):
    metrics = json.loads((Path('/content/imp-runs/konly-rta-v1')/arm/'a2w_seed3/metrics.json').read_text())
    if metrics['epoch'] != 70 or metrics['seed'] != 3:
        raise RuntimeError('Complete the one-shot comparison before alternating')
estimate = json.loads(Path('/content/imp-runs/konly-estimate-v1/seed3-mobile/estimate.json').read_text())
K = int(estimate['K'])
if K <= 0 or estimate['target_labels_used']:
    raise RuntimeError('Invalid initial unlabeled K estimate')
output = Path('/content/imp-runs/konly-alternating-v1/seed3')
output.mkdir(parents=True, exist_ok=False)
prior = '/content/imp-runs/konly-source-prior-v2/seed3/source-final.pt'
command = ['/content/rta-py38/bin/python', '-u', '/content/train_alternating_konly_entry.py',
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
    initial_K=K, virtual_clusters=20, refresh_completed_epochs=[20, 40, 60],
    source_prior=prior, losses='Original published-code RTA',
    fresh_features='Source and target recomputed at every refresh; no target labels',
    new_head_rows='Random directions norm-matched to current RTA weights',
    matching='Current predictions vs new responsibilities; retain matching SGD state',
    schedule='Original optimizer steps continue, no warmup/scheduler reset',
    selection='Selected seed3; final and target-label oracle-best separately'), indent=2))
with (output/'seed3-console.log').open('x') as stream:
    process = subprocess.Popen(command, cwd='/content/rta-legacy-l4-bridge-v1',
                               env=environment, stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT, text=True)
    print('ALTERNATING_WORKER_PID', process.pid, 'initial_K', K, flush=True)
    for line in process.stdout:
        print(line, end='', flush=True)
        stream.write(line)
        stream.flush()
    status = process.wait()
if status:
    raise RuntimeError('Alternating arm stopped; preserve evidence, no automatic tuning')
print('ALTERNATING_ARM_COMPLETE', flush=True)
