"""Source-count anchored mobile prototypes; no target-label tuning."""
import json
from pathlib import Path
import numpy as np
from robust_capacity import fit_robust_capacity

# Focused vector-precision check: scalar and constant vector represent same objective.
toy = np.array([[0.], [.2], [8.], [8.1]])
left = fit_robust_capacity(toy, np.array([[0.]]), 1., prior_strength=5.)
right = fit_robust_capacity(toy, np.array([[0.]]), 1., prior_strength=[5.])
assert np.allclose(left['centers'], right['centers']) and left['K'] == right['K']

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
    counts = np.asarray([(labels[train] == c).sum() for c in range(10)])
    def fit(x, c):
        retained = train[labels[train] < c]
        residual = source[retained]-anchors[labels[retained]]
        penalty = max(float(np.quantile((residual**2).sum(1), .99)), 1e-8)
        result = fit_robust_capacity(x, anchors[:c], penalty, prior_strength=counts[:c],
                                     reference_samples=len(retained)/float(c), birth_order='before_update')
        report = dict(K=result['K'], counts=result['counts'].tolist(),
                      prior_strength=result['prior_strength'], penalty=penalty,
                      noise_count=result['noise_count'], converged=result['converged'],
                      known_drift_norm=np.linalg.norm(result['centers'][:c]-anchors[:c], axis=1).tolist())
        return result, report
    _, full = fit(target, 10)
    _, negative = fit(source[probe], 10)
    positive, positive_report = fit(source[probe], 5)
    ids, y = positive['assignments'], labels[probe]
    hidden = y >= 5
    positive_report.update(hidden_candidate_recall=float((ids[hidden] >= 5).mean()),
                           retained_known_false_candidate=float((ids[~hidden] >= 5).mean()),
                           per_hidden_class_recall={str(c): float((ids[y == c] >= 5).mean()) for c in range(5,10)})
    report = dict(seed=seed, target_full=full, source_negative=negative,
                  source_missing_five=positive_report, target_labels_used=False,
                  policy='kappa_c=source calibration class count; R=mean source class count; mobile known centers',
                  caveat='power-weighted empirical objective, not full DP posterior or target semantic-count guarantee')
    reports.append(report)
    print(json.dumps(report), flush=True)
destination = Path('/content/source-precision-capacity-v1.json')
destination.write_text(json.dumps(reports, indent=2, allow_nan=False))
print('SOURCE_PRECISION_RESULTS', destination, flush=True)
