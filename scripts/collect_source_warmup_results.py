"""Audit warmup products and package exact features/checkpoint for persistence."""
import hashlib
import json
import os
from pathlib import Path
import zipfile

import numpy as np
import torch

run_name = os.environ.get('IMP_WARMUP_RUN', 'source-warmup-l4-seed1-v2')
if Path(run_name).name != run_name or not run_name:
    raise ValueError('IMP_WARMUP_RUN must be a single directory name')
root = Path('/content/imp-runs')/run_name
manifest = json.loads((root/'manifest.json').read_text())
for name, digest in manifest['files'].items():
    if hashlib.sha256((root/name).read_bytes()).hexdigest() != digest:
        raise RuntimeError('Artifact hash mismatch: ' + name)
history = [json.loads(row) for row in (root/'history.jsonl').read_text().splitlines()]
assert [row['epoch'] for row in history] == [1, 2, 3]
assert history[-1]['steps'] == 42
features = np.load(root/'features.npz', allow_pickle=False)
assert features['source'].shape == (958, 256) and features['target'].shape == (564, 256)
assert 'target_labels' not in features.files
assert np.isfinite(features['source']).all() and np.isfinite(features['target']).all()
checkpoint = torch.load(root/'source-final.pt', map_location='cpu', weights_only=False)
assert checkpoint['step'] == 42 and checkpoint['config']['epochs'] == 3
print('SOURCE_ARTIFACT_AUDIT_PASS', json.dumps(manifest), flush=True)
for name, with_weights in [(run_name+'-features.zip', False),
                           (run_name+'-complete.zip', True)]:
    archive = Path('/content')/name
    with zipfile.ZipFile(archive, 'x', zipfile.ZIP_DEFLATED) as bundle:
        for path in root.iterdir():
            if path.is_file() and (with_weights or path.name != 'source-final.pt'):
                bundle.write(path, path.name)
    print('ARCHIVE', name, 'BYTES', archive.stat().st_size,
          'SHA256', hashlib.sha256(archive.read_bytes()).hexdigest(), flush=True)
