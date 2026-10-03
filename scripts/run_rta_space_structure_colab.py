"""Reuse declared grid on fixed RTA-space features; not a virtual-loss ablation."""
import hashlib
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile

output = Path('/content/imp-runs/rta-space-structure-v1')
if output.exists():
    raise RuntimeError('Refusing overwrite')
subprocess.run([sys.executable, '-u', '/content/infer_frozen_source_structure.py',
    '--features', '/content/imp-runs/rta-space-warmup-l4-features-v1/features.npz',
    '--module', '/content/source_anchored_imp.py', '--output', str(output)], check=True)
for name in ['source_anchored_imp.py','infer_frozen_source_structure.py']:
    shutil.copy2(Path('/content')/name, output/name)
archive = Path('/content/rta-space-structure-v1.zip')
with zipfile.ZipFile(archive, 'x', zipfile.ZIP_DEFLATED) as bundle:
    for path in output.iterdir():
        bundle.write(path, path.name)
print('RTA_STRUCTURE_ARCHIVE', archive.stat().st_size,
      hashlib.sha256(archive.read_bytes()).hexdigest(), flush=True)
