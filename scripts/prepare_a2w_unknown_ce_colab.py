"""Reuse the durable Office31 archive; never download another dataset copy."""
from pathlib import Path
import shutil
import tarfile

archive = Path('/content/drive/MyDrive/OSDA/datasets/office31_images.tar')
local = Path('/content/office31_images.tar')
data = Path('/content/osda-office31-a2w-v1')
if not archive.is_file():
    raise FileNotFoundError('Mount Drive and restore the existing Office31 cache')
if data.exists():
    raise FileExistsError('Inspect existing data instead of overwriting it')
shutil.copyfile(archive, local)
data.mkdir()
with tarfile.open(local) as bundle:
    for item in bundle.getmembers():
        path = Path(item.name)
        if path.is_absolute() or '..' in path.parts or not (item.isfile() or item.isdir()):
            raise ValueError('Unsafe archive member: ' + item.name)
    bundle.extractall(data)
for name in ('amazon_0-9_train_all.txt', 'webcam_0-9_20-30_test.txt'):
    original = Path('/content/rta-legacy-l4-bridge-v1/data') / name
    shutil.copyfile(original, data / name)
    rows = [line.rsplit(None, 1) for line in original.read_text().splitlines() if line.strip()]
    if any(not (data / row[0]).is_file() for row in rows):
        raise FileNotFoundError('Image paths do not match the restored archive')
    print('RESTORED_LIST', name, len(rows), flush=True)
print('OFFICE31_DRIVE_CACHE_REUSED', flush=True)
