"""Fixed-assignment merge costs; diagnosis only, not capacity refitting."""
import json
from pathlib import Path
import numpy as np

root = Path('/content/imp-runs/visda-frozenbn-capacity-10e-v1')
output = root/'capacity-anchor-competition-v2.json'
if output.exists():
    raise FileExistsError('Preserve previous diagnostic')
capacity = json.loads((root/'capacity.json').read_text())
settings = capacity['settings']
c = capacity['C']
with np.load(root/'source/features.npz') as f:
    source, labels, target = f['source'].astype(np.float64), f['source_labels'], f['target'].astype(np.float64)
with np.load(root/'capacity.npz') as f:
    centers, ids = f['centers'], f['assignments']
rng = np.random.RandomState(2026)
anchors = []
for cls in range(c):
    rows = np.flatnonzero(labels == cls)
    rng.shuffle(rows)
    anchors.append(source[rows[:max(1, int(.7*len(rows)))]].mean(0))
anchors = np.stack(anchors)
prior = np.asarray(settings['source_prior_precision'], dtype=np.float64)
weight = settings['reference_samples']/len(target)
beta = settings['birth_cost']
radius = settings['lambda_radius']
def full_objective(mu):
    distances = np.maximum((target*target).sum(1)[:, None]+(mu*mu).sum(1)[None]-2*target.dot(mu.T), 0.)
    return float(weight*np.minimum(distances.min(1), radius).sum()+beta*(len(mu)-c)
        +(prior*((mu[:c]-anchors)**2).sum(1)).sum())
base_objective = full_objective(centers)
rows = []
for j in range(c, len(centers)):
    new = target[ids == j]
    comparisons = []
    for known in range(c):
        old = target[ids == known]
        merged = np.concatenate([old, new])
        mu = (weight*merged.sum(0)+prior[known]*anchors[known])/(weight*len(merged)+prior[known])
        before = weight*(((old-centers[known])**2).sum()+((new-centers[j])**2).sum())
        before += prior[known]*((centers[known]-anchors[known])**2).sum()+beta
        after = weight*((merged-mu)**2).sum()+prior[known]*((mu-anchors[known])**2).sum()
        trial = np.delete(centers, j, axis=0).copy()
        trial[known] = mu
        comparisons.append(dict(known_slot=known,
            delta_objective=full_objective(trial)-base_objective,
            delta_fixed_membership_quadratic=float(after-before),
            merged_center_squared_displacement=float(((mu-anchors[known])**2).sum())))
    best = min(comparisons, key=lambda row: row['delta_objective'])
    rows.append(dict(new_slot=j, members=len(new), best_known_merge=best,
        all_known_merges=comparisons))
report = dict(K=capacity['K'], observation_weight=weight, birth_cost=beta, base_objective=base_objective,
    target_labels_used=False, training_changed=False, clusters=rows,
    caveat='Proposed known center uses fixed member sets. delta_objective evaluates full capped nearest-center objective including reassignment/noise; no iterative refitting or proof of semantic novelty. v1 quadratic-only report is not a full objective cost.')
output.write_text(json.dumps(report, indent=2, allow_nan=False))
print('ANCHOR_COMPETITION', json.dumps(report), flush=True)
