"""Package all frozen-grid outputs, including failures and exact code snapshots."""
import hashlib
import json
from pathlib import Path
import zipfile

root = Path('/content/imp-runs/source-anchored-frozen-grid-v2')
report = json.loads((root/'report.json').read_text())
assert len(report['trials']) == 9
for trial in report['trials']:
    if 'error' not in trial:
        assert trial['rows_sum_to_one']
        assert (root/(trial['name']+'.npz')).is_file()
archive = Path('/content/source-anchored-frozen-grid-v2.zip')
with zipfile.ZipFile(archive, 'x', zipfile.ZIP_DEFLATED) as bundle:
    for path in root.iterdir():
        if path.is_file():
            bundle.write(path, path.name)
print('GRID_ARCHIVE', archive.name, archive.stat().st_size,
      hashlib.sha256(archive.read_bytes()).hexdigest(), flush=True)
