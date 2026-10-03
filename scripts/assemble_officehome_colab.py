"""Verify all uploaded parts, then assemble once without overwriting."""
import hashlib
import json
from pathlib import Path
import shutil

def assemble(root):
    root = Path(root)
    manifest = json.loads((root / 'officehome-parts-v1.json').read_text(encoding='utf-8-sig'))
    assert manifest['file'] == 'real_world_0-64_test.zip'
    parts = manifest['parts']
    assert parts and [part['index'] for part in parts] == list(range(len(parts)))
    assert len({part['file'] for part in parts}) == len(parts)
    assert sum(part['bytes'] for part in parts) == manifest['bytes']
    for part in parts:
        assert Path(part['file']).name == part['file']
        path = root / part['file']
        assert path.stat().st_size == part['bytes'] and 0 < part['bytes'] <= 500 * 1024**2
        with path.open('rb') as stream:
            assert hashlib.file_digest(stream, 'sha256').hexdigest() == part['sha256']
        print('PART_VERIFIED', part['file'], flush=True)
    destination = root / manifest['file']
    with destination.open('xb') as output:
        for part in parts:
            with (root / part['file']).open('rb') as stream:
                shutil.copyfileobj(stream, output, 8 * 1024**2)
    assert destination.stat().st_size == manifest['bytes']
    with destination.open('rb') as stream:
        assert hashlib.file_digest(stream, 'sha256').hexdigest() == manifest['sha256']
    with (root / 'officehome-assembly-v1.json').open('x') as stream:
        json.dump(dict(file=manifest['file'], sha256=manifest['sha256'], bytes=manifest['bytes'],
                       verified_parts=len(parts), storage='temporary /content', training_started=False), stream, indent=2)
    print('OFFICEHOME_ASSEMBLY_VERIFIED', manifest['sha256'], flush=True)


if __name__ == '__main__':
    assemble('/content')
