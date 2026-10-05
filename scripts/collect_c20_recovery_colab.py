"""Save completed C20 source model/features locally before capacity experiments."""
import json
from pathlib import Path
import zipfile

root = Path('/content/imp-runs/source-frozenbn-block0to4-v1')
source = root/'source-frozenbn/source'
summary = json.loads((source/'summary.json').read_text())
if not summary['complete'] or summary['source_shape'] != [1458, 256] or summary['target_shape'] != [1785, 256]:
    raise ValueError('C20 source feature stage incomplete')
archive = Path('/content/c20-frozenbn-recovery-20261005.zip')
with zipfile.ZipFile(archive, 'x', compression=zipfile.ZIP_STORED) as output:
    for path in root.rglob('*'):
        if path.is_file():
            output.write(path, str(path.relative_to(root)))
print('C20_RECOVERY_ARCHIVE', str(archive), archive.stat().st_size, flush=True)
