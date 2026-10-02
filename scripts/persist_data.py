"""Move CLI-uploaded inputs into durable Drive storage and verify hashes."""
import hashlib
import json
from pathlib import Path
import shutil

source_root = Path('/content/osda-upload')
drive_root = Path('/content/drive/MyDrive/OSDA')
manifest = json.loads((source_root / 'data-manifest.json').read_text(encoding='utf-8-sig'))
for item in manifest:
    source = source_root / item['file']
    folder = 'pretrained' if source.suffix == '.pth' else 'datasets'
    destination = drive_root / folder / source.name
    shutil.copy2(source, destination)
    digest = hashlib.sha256()
    with destination.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    if digest.hexdigest() != item['sha256']:
        raise RuntimeError(f'Drive checksum mismatch: {destination}')
    print('VERIFIED', destination, flush=True)
shutil.copy2(source_root / 'data-manifest.json', drive_root / 'datasets' / 'data-manifest.json')
print('DATA_PERSISTED', flush=True)
