"""One-factor source proxy: update anchors before proposing new components."""
import json
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0, '/content/visda-capacity-code-v1')
from robust_capacity import fit_robust_capacity

root = Path('/content/imp-runs/source-frozenbn-block0to4-v1')
output = root/'birth-order-probe-v1.json'
if output.exists():
    raise FileExistsError('Preserve completed source probe')
previous = json.loads((root/'prior-mass-probe-v1.json').read_text())
with np.load(root/'source-frozenbn/source/features.npz') as data:
    source, labels, all_source = data['source'].astype(np.float64), data['source_labels'], data['target'].astype(np.float64)
if source.shape != (1458,256) or all_source.shape != (1785,256):
    raise ValueError('Unexpected C20 features')
y = np.array([int(line.rsplit(None,1)[1]) for line in Path(
    '/content/osda-officehome-pr2rw-v1/product_0-24_train_all.txt').read_text().splitlines() if line.strip()])
rng = np.random.RandomState(2026)
train, known_test, hidden_test = [], [], []
for cls in range(20):
    ids = np.flatnonzero(labels == cls); rng.shuffle(ids)
    cut = max(1, int(.7*len(ids))); middle = cut+(len(ids)-cut)//2
    train.extend(ids[:cut]); known_test.extend(ids[middle:])
for cls in range(5):
    ids = np.flatnonzero(y == cls); rng.shuffle(ids)
    cut = max(1, int(.7*len(ids))); middle = cut+(len(ids)-cut)//2
    hidden_test.extend(ids[middle:])
train, known_test, hidden_test = map(np.asarray, (train, known_test, hidden_test))
anchors = np.stack([source[train][labels[train] == cls].mean(0) for cls in range(20)])
report = dict(source_only=True, real_target_used=False, factor='birth_order only',
    before_update=previous['arms']['source_counts'], after_update={},
    caveat=previous['caveat'], training_changed=False)
for name, x in [('known_only',source[known_test]),
                ('known_plus_hidden',np.vstack([source[known_test],all_source[hidden_test]]))]:
    settings = previous['arms']['source_counts'][name]['settings']
    result = fit_robust_capacity(x, anchors, settings['lambda_radius'],
        prior_strength=settings['source_prior_counts'], reference_samples=settings['reference_samples'],
        birth_penalty=settings['birth_cost'], birth_order='after_update', proposal_block_size=64)
    ids = result['assignments']
    row = dict(K=result['K'], converged=result['converged'], counts=result['counts'].tolist(),
        noise_count=result['noise_count'], history=result['history'],
        known_false_candidate_rate=float((ids[:len(known_test)] >= 20).mean()),
        known_identity_accuracy=float((ids[:len(known_test)] == labels[known_test]).mean()))
    if name == 'known_plus_hidden':
        row['hidden_candidate_recall'] = float((ids[len(known_test):] >= 20).mean())
        row['hidden_absorbed_by_known_rate'] = float(((ids[len(known_test):] >= 0) & (ids[len(known_test):] < 20)).mean())
    report['after_update'][name] = row
    print('BIRTH_ORDER_PROXY', name, json.dumps(row), flush=True)
output.write_text(json.dumps(report, indent=2, allow_nan=False))
print('BIRTH_ORDER_PROXY_COMPLETE', str(output), flush=True)
