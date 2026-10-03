import hashlib
from pathlib import Path
import zipfile

archive=Path('/content/rta-hierarchical-pilot-v1.zip')
code=Path('/content/rta-hierarchical-pilot-v1')
if code.exists():
    raise RuntimeError('Refusing to overwrite experiment')
with zipfile.ZipFile(archive) as bundle:
    assert all('/' not in name and '\\' not in name for name in bundle.namelist())
    bundle.extractall(code)
print('HIERARCHICAL_CODE_UNPACKED',hashlib.sha256(archive.read_bytes()).hexdigest(),flush=True)
