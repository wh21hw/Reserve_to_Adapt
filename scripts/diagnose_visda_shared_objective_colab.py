"""Evaluate saved feasible solutions under the same shared objective; no fit."""
import json
from pathlib import Path
import numpy as np
root = Path('/content/imp-runs/visda-frozenbn-capacity-10e-v1')
out = root/'shared-shift-feasible-objective-v1.json'
if out.exists(): raise FileExistsError('Preserve diagnostic')
old = json.loads((root/'capacity.json').read_text())
new = json.loads((root/'shared-domain-shift-v1/capacity.json').read_text())
settings = old['settings']; c = old['C']
with np.load(root/'source/features.npz') as f:
    source, labels, target = f['source'].astype(np.float64), f['source_labels'], f['target'].astype(np.float64)
rng = np.random.RandomState(2026); anchors = []
for cls in range(c):
    rows = np.flatnonzero(labels == cls); rng.shuffle(rows)
    anchors.append(source[rows[:max(1,int(.7*len(rows)))]].mean(0))
anchors = np.stack(anchors)
prior = np.array(settings['source_prior_precision']); tau = new['settings']['shared_shift_precision']
w = settings['reference_samples']/len(target)
radius, beta = settings['lambda_radius'], settings['birth_cost']
def objective(mu,shift):
    d = np.maximum((target*target).sum(1)[:,None]+(mu*mu).sum(1)[None]-2*target.dot(mu.T),0.)
    return float(w*np.minimum(d.min(1),radius).sum()+beta*(len(mu)-c)
        +(prior*((mu[:c]-anchors-shift)**2).sum(1)).sum()+tau*shift.dot(shift))
with np.load(root/'capacity.npz') as f: original = f['centers']
with np.load(root/'shared-domain-shift-v1/capacity.npz') as f: shared = f['centers']
# Conditional MAP of delta with centers held fixed; no target assignment fitting.
conditional_shift = (prior[:,None]*(original[:c]-anchors)).sum(0)/(tau+prior.sum())
report = dict(target_labels_used=False, RTA_changed=False, capacity_refitted=False,
    original_K=old['K'], shared_K=new['K'],
    original_feasible_zero_shift=objective(original,np.zeros_like(conditional_shift)),
    original_feasible_profiled_shift=objective(original,conditional_shift),
    saved_shared_objective=objective(shared,np.array(new['settings']['shared_shift'])),
    conditional_shift_norm=float(np.linalg.norm(conditional_shift)),
    caveat='Objective values measure optimization, not semantic quality. Original K8 is a feasible point of the more flexible model; lower objective does not certify correct unknown count or justify target-label tuning.')
report['shared_minus_feasible_objective'] = report['saved_shared_objective']-report['original_feasible_profiled_shift']
out.write_text(json.dumps(report,indent=2,allow_nan=False))
print('SHARED_FEASIBLE_OBJECTIVE',json.dumps(report),flush=True)
