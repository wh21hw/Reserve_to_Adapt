"""Declared source-only common-translation stress test; no real target use."""
import json
from pathlib import Path
import numpy as np
from robust_capacity import fit_robust_capacity

root = Path('/content/imp-runs/source-frozenbn-block0to4-v1')
output = root/'shared-shift-probe-v1.json'
if output.exists():
    raise FileExistsError('Preserve source experiment')
previous = json.loads((root/'prior-mass-probe-v1.json').read_text())
with np.load(root/'source-frozenbn/source/features.npz') as f:
    source, labels, all_source = f['source'].astype(np.float64), f['source_labels'], f['target'].astype(np.float64)
y = np.array([int(line.rsplit(None, 1)[1]) for line in Path('/content/osda-officehome-pr2rw-v1/product_0-24_train_all.txt').read_text().splitlines() if line.strip()])
rng = np.random.RandomState(2026)
train, test, hidden = [], [], []
for cls in range(20):
    rows = np.flatnonzero(labels == cls); rng.shuffle(rows)
    cut = max(1, int(.7*len(rows))); middle = cut+(len(rows)-cut)//2
    train.extend(rows[:cut]); test.extend(rows[middle:])
for cls in range(5):
    rows = np.flatnonzero(y == cls); rng.shuffle(rows)
    cut = max(1, int(.7*len(rows))); middle = cut+(len(rows)-cut)//2
    hidden.extend(rows[middle:])
train, test, hidden = map(np.asarray, (train, test, hidden))
anchors = np.stack([source[train][labels[train] == c].mean(0) for c in range(20)])
settings = previous['arms']['source_counts']['known_only']['settings']
centered = source[train]-source[train].mean(0)
_, directions = np.linalg.eigh(centered.T.dot(centered))
direction = directions[:, -1]
direction *= 1 if direction[np.argmax(np.abs(direction))] >= 0 else -1
translation = .5*np.sqrt(settings['lambda_radius'])*direction

# One targeted interface check, not a recheck of stable training/checkpoints.
toy = np.array([[.4], [.45], [.5]])
check = fit_robust_capacity(toy, np.array([[0.]]), 10., prior_strength=100.,
    birth_penalty=100., shared_shift_precision=1.)
if not check['converged'] or check['K'] != 0 or abs(check['shared_shift'][0]) < .1:
    raise RuntimeError('Shared-shift interface failed')
report = dict(source_only=True, real_target_used=False, training_changed=False,
    translation_rule='Half square-root source radius along first source-anchor-training PCA direction; no renormalization',
    translation=translation.tolist(), shared_shift_precision=settings['reference_samples'], arms={},
    caveat='Known proxy images were seen by encoder; artificial common translation is not real-domain validation. Hidden raw0..4 source classes remain untrained by encoder. No parameter grid or target labels.')
for scenario, offset in [('unshifted',np.zeros_like(translation)), ('translated',translation)]:
    for pool, x in [('known_only',source[test]), ('mixed',np.vstack([source[test],all_source[hidden]]))]:
        for method, precision in [('original',None), ('shared_shift',settings['reference_samples'])]:
            r = fit_robust_capacity(x+offset, anchors, settings['lambda_radius'],
                prior_strength=settings['source_prior_counts'], reference_samples=settings['reference_samples'],
                birth_penalty=settings['birth_cost'], birth_order='before_update',
                proposal_block_size=64, shared_shift_precision=precision)
            ids = r['assignments']
            row = dict(K=r['K'], converged=r['converged'], noise_count=r['noise_count'],
                known_false_candidate_rate=float((ids[:len(test)] >= 20).mean()),
                known_identity_accuracy=float((ids[:len(test)] == labels[test]).mean()),
                shift_norm=float(np.linalg.norm(r['shared_shift'])),
                shift_recovery_error=float(np.linalg.norm(np.array(r['shared_shift'])-offset)),
                history=r['history'])
            if pool == 'mixed':
                row['hidden_candidate_recall'] = float((ids[len(test):] >= 20).mean())
            key = '/'.join([scenario,pool,method])
            report['arms'][key] = row
            print('SHARED_SHIFT_ARM',key,json.dumps({k:v for k,v in row.items() if k != 'history'}),flush=True)
output.write_text(json.dumps(report,indent=2,allow_nan=False))
print('SHARED_SHIFT_COMPLETE',str(output),flush=True)
