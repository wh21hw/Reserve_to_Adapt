"""Read frozen geometry after K fitting; target labels are diagnosis-only."""
import json
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score

root = Path('/content/imp-runs/visda-frozenbn-capacity-10e-v1')
data = Path('/content/osda-visda-syn2real-v1')
output = root/'capacity-geometry-diagnostic.json'
if output.exists():
    raise FileExistsError('Preserve diagnostic')
capacity = json.loads((root/'capacity.json').read_text())
config = json.loads((root/'source/config.json').read_text())
with np.load(root/'source/features.npz') as f:
    source, labels, target = f['source'].astype(np.float64), f['source_labels'], f['target'].astype(np.float64)
rows = [line.rsplit(None, 1) for line in (data/'target-real-12.txt').read_text().splitlines()]
if [r[0] for r in rows] != (data/'target-unlabeled-paths.txt').read_text().splitlines():
    raise ValueError('Target feature order differs')
raw = np.array([int(r[1]) for r in rows])
if len(raw) != len(target):
    raise ValueError('Target length differs')
c = capacity['C']
rng = np.random.RandomState(2026)
anchor_ids, calibration_ids = [], []
for cls in range(c):
    ids = np.flatnonzero(labels == cls)
    rng.shuffle(ids)
    cut = max(1, int(.7*len(ids)))
    middle = cut+(len(ids)-cut)//2
    anchor_ids.extend(ids[:cut])
    calibration_ids.extend(ids[cut:middle])
anchor_ids = np.array(anchor_ids)
anchors = np.stack([source[anchor_ids][labels[anchor_ids] == cls].mean(0) for cls in range(c)])
def nearest(x, centers):
    dist = np.maximum((x*x).sum(1)[:, None]+(centers*centers).sum(1)[None]-2*x.dot(centers.T), 0)
    return dist.min(1), dist.argmin(1)
radius = capacity['settings']['lambda_radius']
distance, predicted = nearest(target, anchors)
known_ids = config['original_known_ids']
known = np.isin(raw, known_ids)
def stats(x):
    return dict(samples=len(x), mean=float(x.mean()), quantiles={str(q): float(np.quantile(x, q)) for q in [.1,.5,.9,.99]},
                outside_source_radius=float((x > radius).mean()))
with np.load(root/'capacity.npz') as f:
    final_known = f['centers'][:c]
mapped = np.array([known_ids.index(int(v)) if v in known_ids else -1 for v in raw])
report = dict(task=capacity['task'], K=capacity['K'], source_radius=radius,
    source_calibration=stats(nearest(source[calibration_ids], anchors)[0]),
    target_known=stats(distance[known]), target_unknown=stats(distance[~known]),
    target_known_nearest_source_identity_accuracy=float((predicted[known] == mapped[known]).mean()),
    posthoc_unknown_distance_auroc=float(roc_auc_score(~known, distance)),
    known_center_squared_displacement=((final_known-anchors)**2).sum(1).tolist(),
    target_labels_used_for_diagnostic=True, target_labels_used_for_training_or_capacity=False,
    caveat='No new thresholds, refitting, forward pass, or training changes; source radius was frozen before target evaluation')
output.write_text(json.dumps(report, indent=2, allow_nan=False))
print('VISDA_GEOMETRY_DIAGNOSTIC', json.dumps(report), flush=True)
