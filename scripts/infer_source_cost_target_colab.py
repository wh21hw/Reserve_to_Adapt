"""Exploratory target capacity from the already tested source-only cost rule."""
import json
from pathlib import Path
import numpy as np
from robust_capacity import fit_robust_capacity

root=Path('/content/imp-runs/source-precision-officehome-v1/seed1')
out=root/'source-cost-target-capacity.json'
if out.exists():
    raise FileExistsError('Preserve prior inference')
calibration=json.loads((root/'source-birth-calibration.json').read_text())['reports'][0]
f=np.load(root/'source/features.npz')
s,y,x=f['source'].astype(float),f['source_labels'],f['target'].astype(float)
rng=np.random.RandomState(2026); train=[]
for c in range(25):
    ids=np.flatnonzero(y==c); rng.shuffle(ids); train.extend(ids[:max(1,int(.7*len(ids)))])
train=np.asarray(train)
a=np.stack([s[train][y[train]==c].mean(0) for c in range(25)])
counts=np.array([(y[train]==c).sum() for c in range(25)])
r=fit_robust_capacity(x,a,calibration['lambda_radius'],prior_strength=counts,
    reference_samples=calibration['reference_samples'],birth_order='before_update',
    birth_penalty=calibration['birth_cost'])
report=dict(task='OfficeHome Pr->Rw',seed=1,C=25,K=r['K'],converged=r['converged'],
    lambda_radius=calibration['lambda_radius'],birth_cost=calibration['birth_cost'],
    counts=r['counts'].tolist(),noise_count=r['noise_count'],history=r['history'],
    target_labels_used=False,semantic_unknown_count=None,
    caveat='exploratory capacity rule; imperfect source controls, not certified semantic class count',
    reference_samples=calibration['reference_samples'],source_prior_counts=counts.tolist())
out.write_text(json.dumps(report,indent=2))
print('SOURCE_COST_TARGET_CAPACITY',json.dumps(report),flush=True)
