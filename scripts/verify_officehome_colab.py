"""Verify the scoped Pr->Rw archive and every entry before/after extraction."""
import hashlib
import json
import os
from pathlib import Path
import zipfile
from PIL import Image

inputs = Path(os.environ.get('OFFICEHOME_INPUTS', '/content/drive/MyDrive/OSDA/datasets/officehome-pr2rw-v1'))
destination = Path('/content/osda-officehome-pr2rw-v1')
assert not destination.exists(), 'Refusing to overwrite an existing extracted dataset'
manifest = json.loads((inputs / ('officehome-manifest-v1.json' if inputs == Path('/content') else 'manifest.json')).read_text(encoding='utf-8-sig'))
report = []
for item in manifest:
    assert Path(item['file']).name == item['file']
    archive_path = inputs / item['file']
    with archive_path.open('rb') as stream:
        assert hashlib.file_digest(stream, 'sha256').hexdigest() == item['sha256']
    assert archive_path.stat().st_size == item['bytes']
    expected = {entry['path']: entry for entry in item['entries']}
    with zipfile.ZipFile(archive_path) as archive:
        assert len(archive.namelist()) == len(expected)
        assert set(archive.namelist()) == set(expected)
        for name, entry in expected.items():
            resolved = (destination / name).resolve()
            assert resolved.is_relative_to(destination.resolve())
            assert not resolved.exists(), 'Refusing to overwrite an extracted file'
            with archive.open(name) as stream:
                assert hashlib.file_digest(stream, 'sha256').hexdigest() == entry['sha256']
            assert archive.getinfo(name).file_size == entry['bytes']
        archive.extractall(destination)
    for name, entry in expected.items():
        path = destination / name
        with path.open('rb') as stream:
            assert hashlib.file_digest(stream, 'sha256').hexdigest() == entry['sha256']
        if name.lower().endswith(('.jpg', '.jpeg', '.png')):
            with Image.open(path) as image:
                image.verify()
            with Image.open(path) as image:
                image.load()  # decode pixels, not only header/structural validation
    report.append(dict(file=item['file'], sha256=item['sha256'], verified_entries=len(expected)))
    print('ARCHIVE_AND_IMAGES_VERIFIED', item['file'], len(expected), flush=True)
with Path('/content/officehome-input-verification-v1.json').open('x') as stream:
    json.dump(dict(archives=report, destination=str(destination), training_started=False), stream, indent=2)
print('OFFICEHOME_INPUT_VERIFIED', json.dumps(report), flush=True)
