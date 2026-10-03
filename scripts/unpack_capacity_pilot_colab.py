import hashlib
from pathlib import Path
import zipfile

archive=Path('/content/rta-capacity-pilot-v1.zip')
code=Path('/content/rta-capacity-pilot-v1')
if code.exists():
    raise RuntimeError('Refusing to overwrite pilot code')
with zipfile.ZipFile(archive) as bundle:
    assert all('/' not in name and '\\' not in name for name in bundle.namelist())
    bundle.extractall(code)
print('PILOT_CODE_UNPACKED',hashlib.sha256(archive.read_bytes()).hexdigest(),flush=True)
