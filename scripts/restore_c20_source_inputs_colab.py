"""Restore only existing Product source images for the declared C20 proxy study."""
from pathlib import Path
import shutil
import zipfile

root = Path('/content/osda-officehome-pr2rw-v1')
root.mkdir(exist_ok=False)
with zipfile.ZipFile('/content/product_0-24_train_all.zip') as archive:
    for entry in archive.infolist():
        target = (root/entry.filename).resolve()
        if root.resolve() not in target.parents:
            raise ValueError('Unsafe archive path: '+entry.filename)
    archive.extractall(root)
rows = [line.rsplit(None, 1) for line in (root/'product_0-24_train_all.txt').read_text().splitlines() if line.strip()]
if len(rows) != 1785 or {int(row[1]) for row in rows} != set(range(25)):
    raise ValueError('Unexpected source protocol')
if any(not (root/row[0]).is_file() for row in rows):
    raise ValueError('Source image missing')
weights = Path('/content/osda-datasets')
weights.mkdir(exist_ok=True)
shutil.copyfile('/content/resnet50-19c8e357.pth', weights/'resnet50-19c8e357.pth')
print('C20_SOURCE_INPUTS_RESTORED; real target images not required for source proxy study', flush=True)
