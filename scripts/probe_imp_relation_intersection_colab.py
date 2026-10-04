"""Restore initial IMP assignments and probe intersection with cached RTA gate."""
import json
from pathlib import Path
import numpy as np
from robust_capacity import fit_robust_capacity

prior=Path('/content/imp-runs/source-precision-officehome-v1/seed1')
root=Path('/content/imp-runs/officehome-capacity-10e-v1')
output=root/'imp-relation-intersection-probe.json'
if output.exists():
    raise FileExistsError('Preserve previous probe')
calibration=json.loads((prior/'source-birth-calibration.json').read_text())['reports'][0]
f=np.load(prior/'source/features.npz')
s,y,x=f['source'].astype(float),f['source_labels'],f['target'].astype(float)
rng=np.random.RandomState(2026);train=[]
for c in range(25):
    ids=np.flatnonzero(y==c);rng.shuffle(ids);train.extend(ids[:max(1,int(.7*len(ids)))])
train=np.asarray(train)
anchors=np.stack([s[train][y[train]==c].mean(0) for c in range(25)])
counts=np.array([(y[train]==c).sum() for c in range(25)])
result=fit_robust_capacity(x,anchors,calibration['lambda_radius'],prior_strength=counts,
    reference_samples=calibration['reference_samples'],birth_order='before_update',birth_penalty=calibration['birth_cost'])
previous=json.loads((prior/'source-cost-target-capacity.json').read_text())
if not result['converged'] or result['K']!=previous['K']:
    raise ValueError('Existing initial inference not restored; do not change K')
assignment=result['assignments']
np.savez_compressed(root/'initial-imp-assignments.npz',assignments=assignment,centers=result['centers'])
report=dict(rule='Keep RTA candidates only if initial IMP assigns to unknown prototype; noise excluded',
    calibration='Same existing source-only settings, no new threshold or target-label tuning',
    scope='Source-pretrained geometry crossed with final reconstructed RTA gate; not contemporaneous training or causal evidence',
    K=result['K'],noise=int((assignment<0).sum()),arms={})
# Both extractions use target list order, shuffle=False; cached truth is only
# read after geometry and the fixed intersection rule have been computed.
for arm in ('fixed4','estimated'):
    cached=np.load(root/(arm+'-final-relation-snapshot.npz'))
    original=cached['selected']
    if len(original)!=len(assignment):
        raise ValueError('Target row mismatch')
    kept=original&(assignment>=25);removed=original&~kept
    truth=cached['truth']
    report['arms'][arm]=dict(original=int(original.sum()),kept=int(kept.sum()),
        kept_known=int((kept&(truth<25)).sum()),kept_unknown=int((kept&(truth>=25)).sum()),
        removed_known=int((removed&(truth<25)).sum()),removed_unknown=int((removed&(truth>=25)).sum()),
        kept_known_fraction=float((truth[kept]<25).mean()) if kept.any() else None,
        unknown_candidate_retention=float((kept&(truth>=25)).sum()/(original&(truth>=25)).sum()))
output.write_text(json.dumps(report,indent=2,allow_nan=False))
print('IMP_RELATION_INTERSECTION_PROBE',json.dumps(report),flush=True)
