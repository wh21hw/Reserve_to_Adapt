"""Save ordinary logs/configuration for one completed arm, no hash audit."""
import json
import os
from pathlib import Path
import zipfile

arm = os.environ.get('KONLY_COLLECT_ARM', 'fixed2')
if arm not in ('fixed2', 'estimated'):
    raise ValueError('Unknown arm')
root = Path('/content/imp-runs/konly-rta-v1') / arm
run = root/'a2w_seed3'
metrics = json.loads((run/'metrics.json').read_text())
history = [json.loads(line) for line in (run/'history.jsonl').read_text().splitlines() if line]
if metrics['epoch'] != 70 or metrics['seed'] != 3 or len(history) != 70:
    raise RuntimeError('Do not collect an incomplete arm as a completed result')
archive = Path('/content') / ('konly-rta-v1-' + arm + '-results.zip')
with zipfile.ZipFile(archive, 'x', compression=zipfile.ZIP_DEFLATED) as stream:
    for parent, prefix in [(root, 'arm'),
            (Path('/content/imp-runs/konly-source-prior-v2/seed3'), 'source_prior'),
            (Path('/content/imp-runs/konly-estimate-v1/seed3-mobile'), 'estimate')]:
        for path in sorted(parent.rglob('*')):
            if path.is_file() and path.suffix in ('.json', '.jsonl', '.txt', '.log'):
                stream.write(path, prefix + '/' + path.relative_to(parent).as_posix())
print('KONLY_RESULTS_SAVED', str(archive), 'bytes', archive.stat().st_size,
      'best', metrics['best'], 'final', {k: metrics[k] for k in ('OS_star', 'unknown', 'HOS')}, flush=True)
