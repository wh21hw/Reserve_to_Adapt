"""One frozen-feature structure diagnostic stage, no target labels or training."""
from pathlib import Path
import shutil
import os
import subprocess
import sys

root = Path('/content/imp-runs')
warmup_run = os.environ.get('IMP_WARMUP_RUN', 'source-warmup-l4-seed1-v2')
grid_run = os.environ.get('IMP_GRID_RUN', 'source-anchored-frozen-grid-v2')
if any(Path(name).name != name or not name for name in [warmup_run, grid_run]):
    raise ValueError('Run names must be single directory names')
output = root/grid_run
if output.exists():
    raise RuntimeError('Refusing overwrite')
process = subprocess.Popen([sys.executable, '-u', '/content/infer_frozen_source_structure.py',
    '--features', str(root/warmup_run/'features.npz'),
    '--module', '/content/source_anchored_imp.py', '--output', str(output)],
    stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
log_path = Path('/content')/(grid_run+'-console.log')
with log_path.open('x') as log:
    for line in process.stdout:
        print(line, end='', flush=True)
        log.write(line)
        log.flush()
code = process.wait()
if output.is_dir():
    shutil.copy2(log_path, output/'console.log')
    for name in ['source_anchored_imp.py', 'infer_frozen_source_structure.py']:
        shutil.copy2(Path('/content')/name, output/name)
if code:
    raise RuntimeError('Structure diagnostics failed; evidence preserved')
