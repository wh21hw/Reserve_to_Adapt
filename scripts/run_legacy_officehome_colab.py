"""One OfficeHome published-code baseline; no silent retries or tuning."""
import json
import os
from pathlib import Path
import subprocess

seed = int(os.environ.get('OFFICEHOME_SEED', '3'))
if seed not in (1, 2, 3):
    raise ValueError('Use declared baseline seeds 1/2/3')
output = Path('/content/imp-runs/legacy-officehome-v1') / ('seed'+str(seed))
output.mkdir(parents=True, exist_ok=False)
command = ['/content/rta-py38/bin/python', '-u', '/content/train_legacy_task_entry.py',
           '--task', 'officehome-pr2rw', '--shared_classes', '25',
           '--all_classes', '29', '--virtual-clusters', '29',
           '--source', '/content/osda-officehome-pr2rw-v1/product_0-24_train_all.txt',
           '--target', '/content/osda-officehome-pr2rw-v1/real_world_0-64_test.txt',
           '--data_dir', '/content/osda-officehome-pr2rw-v1',
           '--log_dir', str(output), '--name', 'seed'+str(seed),
           '--batch_size', '64', '--learning_rate', '0.00005']
environment = dict(os.environ, RTA_SEED=str(seed), PYTHONUNBUFFERED='1',
                   RTA_MODEL_PATH='/content/osda-datasets/resnet50-19c8e357.pth',
                   OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2')
(output/'launch.json').write_text(json.dumps(dict(command=command, seed=seed,
    epochs=70, budget_status='Transferred release Office31 budget; not confirmed paper OfficeHome budget',
    K=4, virtual_clusters=29,
    virtual_count_status='Declared C+baselineK choice, not confirmed author configuration',
    backbone='ImageNet ResNet50', environment='Selected legacy Python3.8/torch1.7.1 on L4',
    changes='Task protocol only; published-code losses, warmup and optimizer retained',
    target_training_labels='Constant unknown sentinel, no semantic labels',
    selection='Report final and target-label HOS oracle-best; seed3 first, not proven best seed'), indent=2))
with (output/'console.log').open('x') as stream:
    process = subprocess.Popen(command, cwd='/content/rta-legacy-l4-bridge-v1',
        env=environment, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    print('OFFICEHOME_BASELINE_PID', process.pid, 'seed', seed, flush=True)
    for line in process.stdout:
        print(line, end='', flush=True)
        stream.write(line)
        stream.flush()
    status = process.wait()
if status:
    raise RuntimeError('OfficeHome baseline failed; preserve evidence, no automatic tuning')
print('OFFICEHOME_BASELINE_COMPLETE', seed, flush=True)
