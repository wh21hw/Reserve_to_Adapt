"""Label-free target geometry and exact birth-gain diagnostic; no tuning."""
import json
from pathlib import Path
import numpy as np
from source_precision_capacity import estimate_capacity

root = Path('/content/imp-runs/source-precision-officehome-v1/seed1')
output = root/'geometry-diagnostic.json'
if output.exists():
    raise FileExistsError('Preserve previous diagnostic')
f = np.load(root/'source/features.npz')
s, labels, x = f['source'].astype(float), f['source_labels'], f['target'].astype(float)
r, settings = estimate_capacity(s, labels, x)
rng = np.random.RandomState(2026)
cal, valid = [], []
for c in range(settings['C']):
    ids = np.flatnonzero(labels == c)
    rng.shuffle(ids)
    cut = max(1, int(.7*len(ids)))
    cal.extend(ids[:cut]); valid.extend(ids[cut:])
cal, valid = np.asarray(cal), np.asarray(valid)
a = np.stack([s[cal][labels[cal] == c].mean(0) for c in range(settings['C'])])
def distance(left, right):
    return np.maximum((left*left).sum(1)[:, None]+(right*right).sum(1)[None]-2*left.dot(right.T), 0.)
lam, weight = settings['penalty'], settings['reference_samples']/len(x)
radii = np.array([np.quantile(((s[cal][labels[cal] == c]-a[c])**2).sum(1), .99)
                  for c in range(len(a))])
report = dict(settings=settings, observation_weight=weight,
    minimum_support_necessary_even_at_maximum_gain=1/weight,
    source_class_radii=radii.tolist(), source_global_penalty=lam,
    target_labels_used=False, refitted_settings=False)
for name, mu in [('initial', a), ('final', r['centers'])]:
    d = distance(x, mu)
    residual = np.minimum(d.min(1), lam)
    gains, support = [], []
    for start in range(0, len(x), 256):
        improvement = np.maximum(residual[:, None]-distance(x, x[start:start+256]), 0.)
        gains.extend((weight*improvement.sum(0)-lam).tolist())
        support.extend((improvement > 1e-12).sum(0).tolist())
    best = int(np.argmax(gains))
    report[name] = dict(best_gain=gains[best], best_gain_over_penalty=gains[best]/lam,
        best_proposal_support=support[best],
        nearest_distance_over_penalty_quantiles=np.quantile(d.min(1)/lam, [.5,.9,.95,.99]).tolist(),
        outside_global_fraction=float((d.min(1) >= lam).mean()),
        outside_all_source_class_radii_fraction=float((d >= radii[None]).all(1).mean()))
own = ((s[valid]-a[labels[valid]])**2).sum(1)
report['source_holdout'] = dict(samples=len(valid), outside_global=float((own >= lam).mean()),
    outside_own_class_radius=float((own >= radii[labels[valid]]).mean()))
output.write_text(json.dumps(report, indent=2))
print('OFFICEHOME_GEOMETRY', json.dumps(report), flush=True)
