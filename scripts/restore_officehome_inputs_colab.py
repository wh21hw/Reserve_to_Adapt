"""Restore scoped existing task archives; no hashes or repeated image audits."""
from pathlib import Path
import json
import shutil
import zipfile

root = Path('/content/osda-officehome-pr2rw-v1')
inputs = Path('/content')
target = inputs/'real_world_0-64_test.zip'
if not target.exists():
    parts = [inputs/('real_world_0-64_test.zip.part%03d' % i) for i in range(3)]
    if not all(part.is_file() for part in parts):
        raise RuntimeError('Target upload incomplete; do not extract or start training')
    with target.open('xb') as output:
        for part in parts:
            with part.open('rb') as stream:
                shutil.copyfileobj(stream, output, 8*1024**2)
root.mkdir(exist_ok=False)
for name in ('product_0-24_train_all.zip', 'real_world_0-64_test.zip'):
    with zipfile.ZipFile(inputs/name) as archive:
        for entry in archive.infolist():
            path = (root/entry.filename).resolve()
            if root.resolve() not in path.parents or path.exists():
                raise RuntimeError('Unsafe or overlapping archive member: '+entry.filename)
        archive.extractall(root)
report = {}
for name in ('product_0-24_train_all.txt', 'real_world_0-64_test.txt'):
    rows = [line.rsplit(None, 1) for line in (root/name).read_text().splitlines() if line.strip()]
    missing = [row[0] for row in rows if not (root/row[0]).is_file()]
    if missing:
        raise RuntimeError('Missing task images: '+str(missing[:5]))
    report[name] = dict(samples=len(rows), classes=len({int(row[1]) for row in rows}))
if report['product_0-24_train_all.txt']['classes'] != 25 or report['real_world_0-64_test.txt']['classes'] != 65:
    raise RuntimeError('Unexpected OfficeHome protocol')
(root/'restoration.json').write_text(json.dumps(report, indent=2))
print('OFFICEHOME_RESTORED', json.dumps(report), flush=True)
