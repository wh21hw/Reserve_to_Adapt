"""Run attribution outside notebook argv; preserve report and evaluator source."""
from pathlib import Path
import hashlib
import shutil
import subprocess
import sys
import zipfile

script = Path('/content/evaluate_frozen_candidates.py')
result = Path('/content/source-anchored-attribution-v1.json')
subprocess.run([sys.executable, str(script)], check=True)
archive = Path('/content/source-anchored-attribution-v1.zip')
with zipfile.ZipFile(archive, 'x', zipfile.ZIP_DEFLATED) as bundle:
    bundle.write(result, result.name)
    bundle.write(script, script.name)
print('ATTRIBUTION_ARCHIVE', hashlib.sha256(archive.read_bytes()).hexdigest(), flush=True)
