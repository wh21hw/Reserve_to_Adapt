"""One declared alpha/source-epoch arm, fresh seed3 initialization, ten RTA epochs."""
import json
import os
from pathlib import Path
import subprocess
import zipfile

alpha = float(os.environ.get('LEGACY_IMP_ALPHA', '.01'))
epochs = int(os.environ.get('LEGACY_SOURCE_EPOCHS', '5'))
if (alpha, epochs) not in ((.01, 5), (.1, 5), (.05, 3), (.05, 7)):
    raise ValueError('Use declared one-factor settings; existing .05/5 arm is reused')
name = 'alpha%s-ft%d' % (format(alpha, '.2g').replace('.', 'p'), epochs)
root = Path('/content/imp-runs/legacy-imp-tuning-v1')
output = root/name
output.mkdir(parents=True, exist_ok=False)
command = ['/content/rta-py38/bin/python', '-u', '/content/train_legacy_imp_virtual_entry.py',
    '--source', '/content/amazon_0-9_train_all.txt',
    '--target', '/content/webcam_0-9_20-30_test.txt',
    '--data_dir', '/content/osda-datasets/', '--log_dir', str(output),
    '--shared_classes', '10', '--all_classes', '12', '--batch_size', '64',
    '--learning_rate', '0.00005', '--lambda1', '.01', '--lambda2', '1',
    '--lambda3', '.3', '--name', 'seed3']
environment = dict(os.environ, RTA_SEED='3', PYTHONUNBUFFERED='1',
    LEGACY_IMP_ALPHA=str(alpha), LEGACY_SOURCE_EPOCHS=str(epochs),
    LEGACY_RTA_EPOCHS='10',
    RTA_MODEL_PATH='/content/osda-datasets/resnet50-19c8e357.pth',
    OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2')
(output/'launch.json').write_text(json.dumps(dict(command=command, alpha=alpha,
    source_epochs=epochs, rta_epochs=10, seed=3, K_out=2,
    target_selection='Target-HOS-selected epoch/config is exploratory oracle tuning, not unbiased test performance',
    prior='Fixed source centers', numerical_algorithm='Original pasted IMP; no softmax rewrite'), indent=2))
with (output/'console.log').open('x') as stream:
    process = subprocess.Popen(command, env=environment,
        cwd='/content/rta-legacy-l4-bridge-v1', stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, text=True)
    print('LEGACY_TUNING_ARM_START', name, 'PID', process.pid, flush=True)
    for line in process.stdout:
        print(line, end='', flush=True)
        stream.write(line)
        stream.flush()
    status = process.wait()
if status:
    raise RuntimeError('Arm failed; retain evidence and stop, no automatic retries')
files = list(output.glob('*_seed3/metrics.json'))
if len(files) != 1:
    raise RuntimeError('Missing completed arm metrics')
metrics = json.loads(files[0].read_text())
history = [json.loads(row) for row in (files[0].parent/'history.jsonl').read_text().splitlines() if row]
if metrics['epoch'] != 10 or len(history) != 10:
    raise RuntimeError('Incomplete tuning arm')
summary = dict(alpha=alpha, source_epochs=epochs, metrics=metrics,
    virtual_count_range=[min(r['virtual_prototypes'] for r in history), max(r['virtual_prototypes'] for r in history)],
    selection='Exploratory single-seed sweep; target-label oracle-best')
(output/'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False))
with zipfile.ZipFile(root/(name+'-results.zip'), 'x', compression=zipfile.ZIP_DEFLATED) as archive:
    for path in sorted(output.rglob('*')):
        if path.is_file() and path.suffix in ('.json', '.jsonl', '.txt', '.log'):
            archive.write(path, path.relative_to(output).as_posix())
print('LEGACY_TUNING_ARM_COMPLETE', json.dumps(summary, allow_nan=False), flush=True)
