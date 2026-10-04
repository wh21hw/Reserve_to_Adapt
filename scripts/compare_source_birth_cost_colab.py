"""Matched old-cost controls, reusing new-cost probe results without rerunning them."""
import json
from pathlib import Path
import numpy as np
from robust_capacity import fit_robust_capacity

root=Path('/content/imp-runs/source-precision-officehome-v1/seed1')
out=root/'source-birth-cost-comparison.json'
if out.exists():
    raise FileExistsError('Preserve previous control')
new=json.loads((root/'source-birth-calibration.json').read_text())
f=np.load(root/'source/features.npz')
s,y=f['source'].astype(float),f['source_labels']
rng=np.random.RandomState(2026)
train,test=[],[]
for c in range(25):
    ids=np.flatnonzero(y==c); rng.shuffle(ids)
    cut=max(1,int(.7*len(ids)))
    middle=cut+(len(ids)-cut)//2
    train.extend(ids[:cut]); test.extend(ids[middle:])
train,test=np.asarray(train),np.asarray(test)
old=[]
for previous in new['reports']:
    known=previous['known']; tr=train[y[train]<known]
    anchors=np.stack([s[tr][y[tr]==c].mean(0) for c in range(known)])
    counts=np.array([(y[tr]==c).sum() for c in range(known)])
    r=fit_robust_capacity(s[test],anchors,previous['lambda_radius'],prior_strength=counts,
        reference_samples=previous['reference_samples'],birth_order='before_update')
    hidden=y[test]>=known; unknown=r['assignments']>=known
    old.append(dict(known=known,K=r['K'],converged=r['converged'],
        known_false_candidate_fraction=float(unknown[~hidden].mean()),
        hidden_candidate_recall=float(unknown[hidden].mean()) if hidden.any() else None,
        noise_count=r['noise_count'],history=r['history']))
report=dict(new_cost=new,old_cost=old,only_changed='birth penalty; shared partitions and features',target_used=False)
out.write_text(json.dumps(report,indent=2))
print('MATCHED_SOURCE_COST',json.dumps(report),flush=True)
