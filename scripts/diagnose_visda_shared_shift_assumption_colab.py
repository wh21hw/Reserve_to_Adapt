"""Post-hoc labeled mean geometry; never feed diagnostic shifts into inference."""
import json
from pathlib import Path
import numpy as np

root = Path('/content/imp-runs/visda-frozenbn-capacity-10e-v1')
output = root/'shared-shift-assumption-diagnostic.json'
if output.exists():
    raise FileExistsError('Preserve assumption diagnostic')
config = json.loads((root/'source/config.json').read_text())
with np.load(root/'source/features.npz') as f:
    source, labels, target = f['source'].astype(np.float64), f['source_labels'], f['target'].astype(np.float64)
data = Path('/content/osda-visda-syn2real-v1')
rows = [line.rsplit(None,1) for line in (data/'target-real-12.txt').read_text().splitlines()]
if [row[0] for row in rows] != (data/'target-unlabeled-paths.txt').read_text().splitlines():
    raise ValueError('Target order differs')
raw = np.array([int(row[1]) for row in rows])
known_ids = config['original_known_ids']
rng = np.random.RandomState(2026)
anchors = []
for cls in range(6):
    ids = np.flatnonzero(labels == cls); rng.shuffle(ids)
    anchors.append(source[ids[:max(1,int(.7*len(ids)))]].mean(0))
anchors = np.stack(anchors)
target_means = np.stack([target[raw == cls].mean(0) for cls in known_ids])
displacements = target_means-anchors
common = displacements.mean(0)
before = float((displacements**2).sum())
after = float(((displacements-common)**2).sum())
norms = np.linalg.norm(displacements,axis=1)
cosines = displacements.dot(displacements.T)/np.maximum(norms[:,None]*norms[None],1e-12)
report = dict(task='VisDA Synthetic->Real', original_known_ids=known_ids,
    target_labels_used_for_diagnostic=True, target_labels_used_for_inference_or_training=False,
    common_shift_norm=float(np.linalg.norm(common)), per_class_shift_norm=norms.tolist(),
    class_balanced_mean_displacement_sse=before,
    residual_mean_displacement_sse_after_common_shift=after,
    fraction_mean_displacement_sse_explained=1-after/before if before else None,
    shift_direction_cosines=cosines.tolist(),
    caveat='Oracle known-class means assess representational plausibility only. This diagnostic common shift is not an estimator, capacity input, parameter calibration, or classification score. It does not measure sample-level novelty separation.')
output.write_text(json.dumps(report,indent=2,allow_nan=False))
print('SHARED_SHIFT_ASSUMPTION',json.dumps(report),flush=True)
