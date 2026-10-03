"""Extract only the six declared source categories from the inspected archive."""
from collections import Counter
import json
from pathlib import Path
import subprocess

root = Path('/content/osda-visda-syn2real-v1')
inspection = json.loads((root/'train-archive-inspection.json').read_text())
names = {1:'bicycle', 2:'bus', 3:'car', 6:'motorcycle', 10:'train', 11:'truck'}
if (root/'train').exists() or (root/'source-known-6.txt').exists():
    raise RuntimeError('Preserve an existing extraction; do not overwrite')
expected = sum(inspection['images_per_folder'][name] for name in names.values())
if expected != 79765:
    raise RuntimeError('Source archive counts differ from the inspected official data')
command = ['tar', '-xf', str(root/'train.tar'), '-C', str(root)]
command += ['train/'+name for name in names.values()] + ['train/image_list.txt']
process = subprocess.Popen(command)
print('VISDA_SOURCE_EXTRACT_PID', process.pid, 'expected_source_images', expected, flush=True)
if process.wait():
    raise RuntimeError('Extraction failed; preserve partial evidence')
rows, counts = [], Counter()
for line in (root/'train/image_list.txt').read_text().splitlines():
    name, raw_label = line.rsplit(None, 1)
    label = int(raw_label)
    if label not in names:
        continue
    path = Path(name)
    if path.is_absolute() or '..' in path.parts or path.parts[0] != names[label]:
        raise ValueError('Source folder/label mapping differs from official IDs')
    relative = 'train/'+name
    if not (root/relative).is_file():
        raise FileNotFoundError(relative)
    rows.append(relative+' '+str(label))
    counts[label] += 1
if len(rows) != expected:
    raise RuntimeError('Extracted source/list count differs')
with (root/'source-known-6.txt').open('x') as stream:
    stream.write('\n'.join(rows)+'\n')
report = dict(source_samples=len(rows), original_known_ids=list(names),
              class_counts=dict(counts), image_root=str(root),
              source_list=str(root/'source-known-6.txt'), target_ready=False,
              target_labels_used_for_training=False, hashes_used=False,
              stage='Known source extracted; no model/training')
(root/'source-preparation.json').write_text(json.dumps(report, indent=2))
print('VISDA_KNOWN_SOURCE_READY', json.dumps(report), flush=True)
