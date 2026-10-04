"""Source-derived reference mass; no target labels, no trained-network changes."""
import json
import os
from pathlib import Path
import numpy as np
from robust_capacity import fit_robust_capacity

birth_order = os.environ.get('ROBUST_BIRTH_ORDER', 'after_update')

toy = np.array([[0.], [.1], [10.], [10.1], [100.]])
one = fit_robust_capacity(toy, np.array([[0.]]), 1., reference_samples=5., birth_order=birth_order)
two = fit_robust_capacity(np.repeat(toy, 2, axis=0), np.array([[0.]]), 1., reference_samples=5., birth_order=birth_order)
assert one['K'] == two['K'] == 1
assert np.allclose(one['centers'], two['centers'])
assert np.isclose(one['history'][-1]['objective'], two['history'][-1]['objective'])

def summary(result):
    return dict(K=result['K'], counts=result['counts'].tolist(),
                noise_count=result['noise_count'], converged=result['converged'],
                reference_samples=result['reference_samples'],
                objective=result['history'][-1]['objective'])

reports = []
for seed in (1, 2, 3):
    path = Path('/content/imp-runs/fusion-imp-rta-v1')/('seed%d/source/features.npz' % seed)
    with np.load(path) as data:
        source, labels, target = data['source'].astype(np.float64), data['source_labels'], data['target'].astype(np.float64)
    train, probe = [], []
    rng = np.random.RandomState(2026)
    for c in range(10):
        indices = np.flatnonzero(labels == c)
        rng.shuffle(indices)
        cut = max(1, int(.7*len(indices)))
        train.extend(indices[:cut]); probe.extend(indices[cut:])
    train, probe = np.asarray(train), np.asarray(probe)
    anchors = np.stack([source[train][labels[train] == c].mean(0) for c in range(10)])
    residual = source[train]-anchors[labels[train]]
    penalty = max(float(np.quantile((residual**2).sum(1), .99)), 1e-8)
    reference = len(train)/10.
    def fit(x, centers=anchors, lam=penalty, ref=reference):
        return fit_robust_capacity(x, centers, lam, reference_samples=ref, birth_order=birth_order)
    full = fit(target)
    matched = fit(target[np.random.RandomState(2027).permutation(len(target))[:len(probe)]])
    negative = fit(source[probe])
    retained = train[labels[train] != 9]
    residual9 = source[retained]-anchors[labels[retained]]
    penalty9 = max(float(np.quantile((residual9**2).sum(1), .99)), 1e-8)
    positive = fit(source[probe], anchors[:9], penalty9, len(retained)/9.)
    hidden = labels[probe] == 9
    positive_report = summary(positive)
    positive_report['hidden_candidate_recall'] = float((positive['assignments'][hidden] >= 9).mean())
    positive_report['retained_known_false_candidate'] = float((positive['assignments'][~hidden] >= 9).mean())
    report = dict(seed=seed, penalty=penalty, birth_order=birth_order, target_full=summary(full),
                  target_matched=summary(matched), source_negative=summary(negative),
                  source_missing_anchor9=positive_report, target_labels_used=False,
                  reference_policy='mean class size in source calibration split; proposed modeling choice, not inferred DP concentration')
    if seed == 1:
        duplicate = fit(np.repeat(target, 2, axis=0))
        same = duplicate['K'] == full['K'] and np.allclose(duplicate['centers'], full['centers'])
        if not same:
            raise RuntimeError('Real-feature duplication invariance failed')
        report['duplicate_target_check'] = dict(passed=True, K=duplicate['K'], rows=2*len(target))
    reports.append(report)
    print(json.dumps(report), flush=True)
destination = Path('/content/normalized-capacity-v1.json' if birth_order == 'after_update'
                   else '/content/normalized-capacity-birth-first-v1.json')
destination.write_text(json.dumps(reports, indent=2, allow_nan=False))
print('NORMALIZED_CAPACITY_RESULTS', destination, flush=True)
