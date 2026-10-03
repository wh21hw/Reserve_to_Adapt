"""Inspect official archive names and list metadata, no extraction or hashes."""
from collections import Counter
import json
from pathlib import Path, PurePosixPath
import tarfile

root = Path('/content/osda-visda-syn2real-v1')
counts, lists, first = Counter(), {}, []
with tarfile.open(root/'train.tar', 'r:') as archive:
    for member in archive:
        path = PurePosixPath(member.name)
        if path.is_absolute() or '..' in path.parts or member.issym() or member.islnk():
            raise ValueError('Unsafe archive member: ' + member.name)
        if len(first) < 12:
            first.append(member.name)
        if member.isfile() and path.suffix.lower() in ('.png', '.jpg', '.jpeg'):
            counts[path.parent.name] += 1
        if member.isfile() and path.name == 'image_list.txt':
            if member.size > 50 * 1024**2:
                raise ValueError('Unexpected large list metadata')
            rows = archive.extractfile(member).read().decode('utf-8').splitlines()
            lists[member.name] = dict(rows=len(rows), first_rows=rows[:5])
report = dict(stage='Archive names/list inspection only; images not extracted',
              archive_bytes=(root/'train.tar').stat().st_size,
              first_members=first, images_per_folder=dict(counts), image_lists=lists,
              source_filter='Only bicycle,bus,car,motorcycle,train,truck for RTA source')
(root/'train-archive-inspection.json').write_text(json.dumps(report, indent=2))
print('VISDA_TRAIN_ARCHIVE_INSPECTED', json.dumps(report), flush=True)
