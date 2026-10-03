import hashlib
from pathlib import Path
import zipfile

archive=Path('/content/rta-capacity-prior-pilot-v1.zip')
assert hashlib.sha256(archive.read_bytes()).hexdigest()=='bba3a5861e0541b4102db61f442162852bb39e955cd336fa33c7f43e5e8f261b'
code=Path('/content/rta-capacity-prior-pilot-v1')
if code.exists():
    raise RuntimeError('Refusing to overwrite pilot code')
with zipfile.ZipFile(archive) as bundle:
    assert all('/' not in name and '\\' not in name for name in bundle.namelist())
    bundle.extractall(code)
print('CAPACITY_PRIOR_CODE_UNPACKED',flush=True)
