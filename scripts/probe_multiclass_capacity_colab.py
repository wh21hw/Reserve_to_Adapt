"""Missing five source anchors: structure check, not unseen-class generalization."""
import json
from pathlib import Path
import numpy as np
from robust_capacity import fit_robust_capacity

reports = []
for seed in (1, 2, 3):
    path = Path('/content/imp-runs/fusion-imp-rta-v1')/('seed%d/source/features.npz' % seed)
    with np.load(path) as data:
        source, labels = data['source'].astype(np.float64), data['source_labels']
    train, probe = [], []
    rng = np.random.RandomState(2026)
    for c in range(10):
        indices = np.flatnonzero(labels == c)
        rng.shuffle(indices)
        cut = max(1, int(.7*len(indices)))
        train.extend(indices[:cut]); probe.extend(indices[cut:])
    train, probe = np.asarray(train), np.asarray(probe)
    retained = train[labels[train] < 5]
    anchors = np.stack([source[retained][labels[retained] == c].mean(0) for c in range(5)])
    residual = source[retained]-anchors[labels[retained]]
    penalty = max(float(np.quantile((residual**2).sum(1), .99)), 1e-8)
    result = fit_robust_capacity(source[probe], anchors, penalty,
                                 reference_samples=len(retained)/5., birth_order='before_update')
    ids, y = result['assignments'], labels[probe]
    hidden = y >= 5
    # Full table preserves known absorption and noise; scoring labels never enter fit.
    table = {str(c): {str(int(slot)): int((ids[y == c] == slot).sum())
                     for slot in np.unique(ids[y == c])} for c in range(10)}
    per_class = {str(c): float((ids[y == c] >= 5).mean()) for c in range(5, 10)}
    report = dict(seed=seed, known_anchors=5, removed_classes=[5,6,7,8,9],
                  K=result['K'], counts=result['counts'].tolist(),
                  noise_count=result['noise_count'], converged=result['converged'],
                  penalty=penalty, reference_samples=len(retained)/5.,
                  hidden_candidate_recall=float((ids[hidden] >= 5).mean()),
                  per_hidden_class_recall=per_class,
                  retained_known_false_candidate=float((ids[~hidden] >= 5).mean()),
                  class_slot_table=table, target_features_used=False, target_labels_used=False,
                  caveat='All classes seen by source-trained representation; source labels only score probe after fit')
    reports.append(report)
    print(json.dumps(report), flush=True)
destination = Path('/content/multiclass-capacity-control-v1.json')
destination.write_text(json.dumps(reports, indent=2, allow_nan=False))
print('MULTICLASS_CAPACITY_RESULTS', destination, flush=True)
