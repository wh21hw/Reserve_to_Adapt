"""Unpack scoped code and run preflight in fresh interpreter, avoiding kernel cache."""
from pathlib import Path
import subprocess
import sys
import zipfile

archive = Path('/content/rta-multitask-baseline-v1.zip')
with zipfile.ZipFile(archive) as package:
    assert not Path('/content/rta_multitask_baseline_v1').exists()
    assert all(not Path(name).is_absolute() and '..' not in Path(name).parts for name in package.namelist())
    package.extractall('/content')
with Path('/content/multitask-preflight-console-v1.log').open('x') as stream:
    process = subprocess.Popen([sys.executable, '-u', '/content/preflight_multitask_colab.py'],
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for line in process.stdout:
        print(line, end='', flush=True)
        stream.write(line)
        stream.flush()
    status = process.wait()
if status:
    raise RuntimeError(f'Preflight failed ({status}); no optimizer step or training launched')
