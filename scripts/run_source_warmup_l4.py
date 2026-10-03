"""One fixed source-only warmup stage after data preflight has succeeded."""
from pathlib import Path
import os
import shutil
import subprocess
import sys

drive_root = Path('/content/drive/MyDrive/OSDA/runs')
output_root = drive_root if drive_root.is_dir() else Path('/content/imp-runs')
output_root.mkdir(exist_ok=True)
run_name = os.environ.get('IMP_WARMUP_RUN', 'source-warmup-l4-seed1-v2')
if Path(run_name).name != run_name or not run_name:
    raise ValueError('IMP_WARMUP_RUN must be a single directory name')
output = output_root/run_name
if output.exists():
    raise RuntimeError('Result folder already exists')
print('RESULT_STORAGE', str(output), 'download/persist required if stored in /content', flush=True)
script = Path('/content/source_warmup_features.py')
command = [sys.executable, '-u', str(script), '--source-list', '/content/amazon_0-9_train_all.txt',
           '--target-list', '/content/webcam_0-9_20-30_test.txt', '--output', str(output),
           '--epochs', '3', '--seed', '1']
process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
log_path = Path('/content')/(run_name+'-console.log')
with log_path.open('x') as log:
    for line in process.stdout:
        print(line, end='', flush=True)
        log.write(line)
        log.flush()
code = process.wait()
if output.is_dir():
    shutil.copy2(log_path, output/'console.log')
    shutil.copy2(script, output/script.name)
if code:
    raise RuntimeError('Source warmup failed; inspect saved evidence')
print('SOURCE_WARMUP_STAGE_DONE', str(output), flush=True)
