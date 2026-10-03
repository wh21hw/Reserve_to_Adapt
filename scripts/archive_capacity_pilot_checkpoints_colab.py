"""Separate <500MiB archives because Drive mount/API is unavailable."""
import hashlib
import json
from pathlib import Path
import zipfile

root=Path('/content/imp-runs/rta-capacity-pilot-v1')
expected={row['arm']:row['last_checkpoint_sha256'] for row in json.loads((root/'summary.json').read_text())['arms']}
for arm in ('original2','proto2','proto18'):
    path=root/arm/'a2w_seed1/last.pt'
    assert hashlib.sha256(path.read_bytes()).hexdigest()==expected[arm]
    archive=Path('/content')/f'rta-capacity-pilot-v1-{arm}-checkpoint.zip'
    with zipfile.ZipFile(archive,'x',zipfile.ZIP_STORED) as bundle:
        bundle.write(path,'last.pt')
        bundle.write(root/arm/'a2w_seed1/handoff.json','handoff.json')
        bundle.write(root/arm/'audit.json','audit.json')
    print('PILOT_CHECKPOINT_ARCHIVE',arm,archive.stat().st_size,hashlib.sha256(archive.read_bytes()).hexdigest(),flush=True)
