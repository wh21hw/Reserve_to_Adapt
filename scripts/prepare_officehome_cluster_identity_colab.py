"""Recover missing unlabeled cluster assignments from the same completed prior."""
import json
from pathlib import Path
import sys
import numpy as np

sys.path.insert(0, '/content/visda-capacity-code-v1')
from source_precision_capacity import estimate_source_cost_capacity

prior = Path('/content/imp-runs/officehome-frozenbn-capacity-10e-v1/prior')
output = Path('/content/imp-runs/officehome-cluster-identity-10e-v1')
output.mkdir(exist_ok=False)
old = json.loads((prior/'capacity.json').read_text())
paths = [r.rsplit(None, 1)[0] for r in
         Path('/content/osda-officehome-pr2rw-v1/real_world_0-64_test.txt').read_text().splitlines()]
with np.load(prior/'source/features.npz') as data:
    source, labels, target = data['source'], data['source_labels'], data['target']
if source.shape != (1785, 256) or target.shape != (4357, 256) or len(paths) != 4357:
    raise ValueError('Expected existing full OfficeHome prior and target order')
print('CLUSTER_IDENTITY_RECOVERY_START', flush=True)
result, settings = estimate_source_cost_capacity(source, labels, target)
report = dict(K=result['K'], settings=settings, counts=result['counts'].tolist(),
              noise_count=result['noise_count'], target_labels_used=False,
              reason='Previous count-only artifact omitted cluster identities')
(output/'recovery.json').write_text(json.dumps(report, indent=2, allow_nan=False))
for key in ('lambda_radius', 'birth_cost', 'source_prior_counts', 'reference_samples'):
    if settings[key] != old['settings'][key]:
        raise RuntimeError('Recovered settings differ from existing estimate; do not train')
if result['K'] != 2 or report['counts'] != old['counts'] or report['noise_count'] != old['noise_count']:
    raise RuntimeError('Recovered clustering differs from existing K2 evidence; preserve and inspect')
np.savez_compressed(output/'clusters.npz', assignments=result['assignments'],
                    centers=result['centers'], target_paths=np.array(paths))
print('CLUSTER_IDENTITY_RECOVERY_COMPLETE', json.dumps(report), flush=True)
