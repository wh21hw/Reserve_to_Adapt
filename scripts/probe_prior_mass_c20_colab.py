"""Fixed source-only positive/negative probe, not target-label model selection."""
import json
from pathlib import Path
import numpy as np
from source_precision_capacity import estimate_source_cost_capacity

root = Path('/content/imp-runs/source-frozenbn-block0to4-v1')
output = root/'prior-mass-probe-v1.json'
if output.exists():
    raise FileExistsError('Preserve previous probe')
with np.load(root/'source-frozenbn/source/features.npz') as data:
    source, labels, all_source = data['source'], data['source_labels'], data['target']
if source.shape != (1458, 256) or all_source.shape != (1785, 256):
    raise ValueError('Expected the declared real C20 frozen-BN features')
y = np.array([int(row.rsplit(None, 1)[1]) for row in Path(
    '/content/osda-officehome-pr2rw-v1/product_0-24_train_all.txt').read_text().splitlines() if row.strip()])
rng = np.random.RandomState(2026)
known_test = []
for c in range(20):
    ids = np.flatnonzero(labels == c)
    rng.shuffle(ids)
    cut = max(1, int(.7*len(ids)))
    middle = cut+(len(ids)-cut)//2
    known_test.extend(ids[middle:])
hidden_test = []
for c in range(5):
    ids = np.flatnonzero(y == c)
    rng.shuffle(ids)
    cut = max(1, int(.7*len(ids)))
    middle = cut+(len(ids)-cut)//2
    hidden_test.extend(ids[middle:])
known_test, hidden_test = map(np.asarray, (known_test, hidden_test))
known_x, hidden_x = source[known_test], all_source[hidden_test]
report = dict(source_only=True, real_target_used=False, seed=1, source_epochs=3,
    C=20, hidden_original_source_ids=list(range(5)), split_seed=2026,
    known_test_count=len(known_test), hidden_test_count=len(hidden_test), arms={},
    caveat='Proxy labels are source labels only. Known probe images participated in encoder supervision; no out-of-sample target guarantee.')
for mode in ('source_counts', 'domain_balanced'):
    arm = {}
    for name, x in [('known_only', known_x), ('known_plus_hidden', np.vstack([known_x, hidden_x]))]:
        result, settings = estimate_source_cost_capacity(source, labels, x,
            proposal_block_size=64, prior_mass_mode=mode)
        ids = result['assignments']
        row = dict(settings=settings, K=result['K'], converged=result['converged'],
            counts=result['counts'].tolist(), noise_count=result['noise_count'],
            history=result['history'], known_false_candidate_rate=float((ids[:len(known_x)] >= 20).mean()),
            known_noise_rate=float((ids[:len(known_x)] < 0).mean()),
            known_identity_accuracy=float((ids[:len(known_x)] == labels[known_test]).mean()))
        if name == 'known_plus_hidden':
            hidden_ids = ids[len(known_x):]
            row.update(hidden_candidate_recall=float((hidden_ids >= 20).mean()),
                hidden_absorbed_by_known_rate=float(((hidden_ids >= 0) & (hidden_ids < 20)).mean()),
                hidden_noise_rate=float((hidden_ids < 0).mean()),
                per_hidden_class_recall={str(c): float((hidden_ids[y[hidden_test] == c] >= 20).mean()) for c in range(5)})
        arm[name] = row
        print('PRIOR_MASS_PROBE', mode, name, json.dumps(row), flush=True)
    report['arms'][mode] = arm
output.write_text(json.dumps(report, indent=2, allow_nan=False))
print('PRIOR_MASS_C20_COMPLETE', str(output), flush=True)
