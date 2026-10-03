"""Prepare official validation target; labels remain evaluation metadata only."""
from collections import Counter
import json
from pathlib import Path, PurePosixPath
import subprocess
import tarfile

root = Path('/content/osda-visda-syn2real-v1')
names = ['aeroplane', 'bicycle', 'bus', 'car', 'horse', 'knife',
         'motorcycle', 'person', 'plant', 'skateboard', 'train', 'truck']
if (root/'validation').exists() or (root/'target-real-12.txt').exists():
    raise RuntimeError('Preserve existing target preparation; do not overwrite')
counts = Counter()
with tarfile.open(root/'validation.tar', 'r:') as archive:
    for member in archive:
        path = PurePosixPath(member.name)
        if path.is_absolute() or '..' in path.parts or member.issym() or member.islnk():
            raise ValueError('Unsafe archive member: '+member.name)
        if not path.parts or path.parts[0] != 'validation':
            raise ValueError('Unexpected archive root: '+member.name)
        if member.isfile() and path.suffix.lower() in ('.png', '.jpg', '.jpeg'):
            counts[path.parent.name] += 1
if sum(counts.values()) != 55388 or set(counts) != set(names):
    raise RuntimeError('Unexpected official target counts/categories')
process = subprocess.Popen(['tar', '-xf', str(root/'validation.tar'), '-C', str(root)])
print('VISDA_TARGET_EXTRACT_PID', process.pid, flush=True)
if process.wait():
    raise RuntimeError('Extraction failed; preserve partial evidence')
rows = []
for line in (root/'validation/image_list.txt').read_text().splitlines():
    name, raw_label = line.rsplit(None, 1)
    label = int(raw_label)
    path = PurePosixPath(name)
    if path.is_absolute() or '..' in path.parts or not 0 <= label < 12:
        raise ValueError('Unsafe target list entry')
    if path.parts[0] != names[label]:
        raise ValueError('Target folder/label mapping differs from official IDs')
    relative = 'validation/'+name
    if not (root/relative).is_file():
        raise FileNotFoundError(relative)
    rows.append(relative+' '+str(label))
if len(rows) != 55388:
    raise RuntimeError('Unexpected target list count')
with (root/'target-real-12.txt').open('x') as stream:
    stream.write('\n'.join(rows)+'\n')
with (root/'target-unlabeled-paths.txt').open('x') as stream:
    stream.write('\n'.join(row.rsplit(None, 1)[0] for row in rows)+'\n')
report = dict(source_samples=79765, target_samples=len(rows),
              target_class_counts=dict(counts), image_root=str(root),
              source_list=str(root/'source-known-6.txt'),
              target_evaluation_list=str(root/'target-real-12.txt'),
              target_unlabeled_list=str(root/'target-unlabeled-paths.txt'),
              target_labels_used_for_training=False, hashes_used=False,
              stage='Data preparation only; backbone unresolved; no model/training')
(root/'data-preparation.json').write_text(json.dumps(report, indent=2))
print('VISDA_DATA_READY', json.dumps(report), flush=True)
