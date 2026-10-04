"""Independent old-design experiment; never resumes/tunes automatically."""
import json
import os
from pathlib import Path
import subprocess

output = Path('/content/imp-runs/legacy-imp-virtual-v1/seed3')
output.mkdir(parents=True, exist_ok=False)
command = ['/content/rta-py38/bin/python', '-u', '/content/train_legacy_imp_virtual_entry.py',
    '--source', '/content/amazon_0-9_train_all.txt',
    '--target', '/content/webcam_0-9_20-30_test.txt',
    '--data_dir', '/content/osda-datasets/', '--log_dir', str(output),
    '--shared_classes', '10', '--all_classes', '12', '--batch_size', '64',
    '--learning_rate', '0.00005', '--lambda1', '.01', '--lambda2', '1',
    '--lambda3', '.3', '--name', 'seed3']
environment = dict(os.environ, RTA_SEED='3', PYTHONUNBUFFERED='1',
    RTA_MODEL_PATH='/content/osda-datasets/resnet50-19c8e357.pth',
    OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2')
(output/'launch.json').write_text(json.dumps(dict(command=command,
    task='Office31 A->W', seed=3, source_epochs=5, rta_epochs=70,
    K_out=2, alpha=.05, IMP_steps=5, known_centers_fixed=True,
    virtual_prototype_refresh='Every epoch, original historical implementation',
    comparison='Whole historical recipe vs selected baseline; not a pure IMP ablation',
    selection='Previously selected seed3; best epoch uses target HOS',
    numerical_algorithm='Original pasted IMP, no threshold/soft-assignment rewrite'), indent=2))
with (output/'console.log').open('x') as stream:
    process = subprocess.Popen(command, env=environment,
        cwd='/content/rta-legacy-l4-bridge-v1', stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, text=True)
    print('LEGACY_IMP_WORKER_PID', process.pid, flush=True)
    for line in process.stdout:
        print(line, end='', flush=True)
        stream.write(line)
        stream.flush()
    status = process.wait()
if status:
    raise RuntimeError('Historical experiment failed; preserve evidence, diagnose before any repair')
print('LEGACY_IMP_VIRTUAL_COMPLETE', flush=True)
