"""Run the audited upstream-code A->W baseline on the existing Colab runtime."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import time

root = Path('/content/rta-official-baseline-v2')
run_root = Path('/content/drive/MyDrive/OSDA/runs/rta-official-a2w-v2')
python = '/content/rta-py38/bin/python'
if not Path(python).is_file() or not run_root.parent.is_dir():
    raise RuntimeError('Old-PyTorch environment or Drive mount missing')
if root.exists():
    raise RuntimeError('Refusing to overwrite an existing baseline checkout')
root.mkdir()
with tarfile.open('/content/rta-baseline-code-v2.tar.gz') as archive:
    for member in archive.getmembers():
        if member.issym() or member.islnk() or not (root/member.name).resolve().is_relative_to(root.resolve()):
            raise RuntimeError('Unsafe code archive')
    archive.extractall(root)
run_root.mkdir(exist_ok=False)
report = dict(upstream_commit='8a4cd890b341a0dd4ed7c239533a1b6a9583fe8b',
              github_branch='codex/rta-baseline',
              baseline_commit='35fcfaa',
              paper_a2w=dict(OS_star=92.2, unknown=93.8, HOS=93.0),
              seeds=[1, 2, 3], files={})
for relative in ['main.py', 'networks.py', 'utilities.py', 'centroid.py', 'domain_bus.py',
                 'data.py', 'data/amazon_0-9_train_all.txt', 'data/webcam_0-9_20-30_test.txt']:
    report['files'][relative] = hashlib.sha256((root/relative).read_bytes()).hexdigest()
for name in ['amazon_0-9_train_all.txt', 'webcam_0-9_20-30_test.txt']:
    missing = []
    counts = {}
    for row in (root/'data'/name).read_text().splitlines():
        image, label = row.rsplit(' ', 1)
        if not (Path('/content/osda-datasets')/image).is_file():
            missing.append(image)
        counts[label] = counts.get(label, 0)+1
    if missing:
        raise RuntimeError(f'Missing images: {missing[:5]}')
    report[name] = counts
environment = subprocess.check_output([python, '-c',
    'import sys,torch,torchvision,faiss,numpy,scipy,sklearn; print(dict(python=sys.version,torch=torch.__version__,torchvision=torchvision.__version__,numpy=numpy.__version__,scipy=scipy.__version__,sklearn=sklearn.__version__,faiss=faiss.__version__,gpu=torch.cuda.get_device_name()))'], text=True)
report['environment'] = environment
(run_root/'audit.json').write_text(json.dumps(report, indent=2))
(run_root/'pip-freeze.txt').write_text(subprocess.check_output(['uv','pip','freeze','--python',python], text=True))
print('BASELINE_AUDIT', json.dumps(report), flush=True)
for seed in report['seeds']:
    env = dict(os.environ, RTA_SEED=str(seed),
               RTA_MODEL_PATH='/content/osda-datasets/resnet50-19c8e357.pth',
               PYTHONUNBUFFERED='1', OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2')
    argv = [python, '-u', 'main.py', '--source', 'data/amazon_0-9_train_all.txt',
            '--target', 'data/webcam_0-9_20-30_test.txt',
            '--data_dir', '/content/osda-datasets/', '--log_dir', str(run_root)+'/',
            '--batch_size', '64', '--learning_rate', '0.00005',
            '--shared_classes', '10', '--all_classes', '12', '--name', f'seed{seed}']
    print('START_SEED', seed, ' '.join(argv), flush=True)
    with (run_root/f'seed{seed}-console.log').open('w') as log:
        process = subprocess.Popen(argv, cwd=root, env=env, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True, bufsize=1)
        for line in process.stdout:
            log.write(line)
            log.flush()
            print(line, end='', flush=True)
        if process.wait():
            raise RuntimeError(f'Baseline seed {seed} failed; inspect its console log')
    metrics = json.loads((run_root/f'a2w_seed{seed}'/'metrics.json').read_text())
    if metrics['epoch'] != 70:
        raise RuntimeError('Incomplete baseline')
    print('COMPLETED_SEED', seed, json.dumps(metrics), flush=True)
print('RTA_BASELINE_COMPLETE', flush=True)
