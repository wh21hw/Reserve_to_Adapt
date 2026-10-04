"""Four remaining source class blocks; fixed calibration, no target access."""
import json
from pathlib import Path
import numpy as np
from scipy.optimize import linear_sum_assignment
from robust_capacity import fit_robust_capacity

root=Path('/content/imp-runs/source-precision-officehome-v1/seed1')
out=root/'source-birth-groups.json'
if out.exists():
    raise FileExistsError('Preserve previous experiment')
f=np.load(root/'source/features.npz')
s,y=f['source'].astype(float),f['source_labels']
rng=np.random.RandomState(2026)
train,cost_rows,test=[],[],[]
for c in range(25):
    ids=np.flatnonzero(y==c); rng.shuffle(ids)
    cut=max(1,int(.7*len(ids))); middle=cut+(len(ids)-cut)//2
    train.extend(ids[:cut]); cost_rows.extend(ids[cut:middle]); test.extend(ids[middle:])
train,cost_rows,test=map(np.asarray,(train,cost_rows,test))
def d(left,right):
    return np.maximum((left*left).sum(1)[:,None]+(right*right).sum(1)[None]-2*left.dot(right.T),0.)
report=dict(protocol='four remaining five-class blocks; unchanged source cost rule',target_used=False,
    caveat='network supervised all25 classes; geometric held-anchor controls only',groups=[])
for start in (0,5,10,15):
    hidden_ids=list(range(start,start+5)); known=[c for c in range(25) if c not in hidden_ids]
    tr=train[np.isin(y[train],known)]; cr=cost_rows[np.isin(y[cost_rows],known)]
    a=np.stack([s[tr][y[tr]==c].mean(0) for c in known])
    counts=np.array([(y[tr]==c).sum() for c in known])
    remap={c:i for i,c in enumerate(known)}
    radius=max(float(np.quantile(((s[tr]-a[[remap[int(c)] for c in y[tr]]])**2).sum(1),.99)),1e-8)
    reference=float(counts.mean())
    residual=np.minimum(d(s[cr],a).min(1),radius)
    beta=max(float((reference/len(cr)*np.maximum(residual[:,None]-d(s[cr],s[cr]),0.).sum(0)).max())*(1+1e-6),1e-8)
    group=dict(hidden_ids=hidden_ids,beta=beta,lambda_radius=radius,arms=[])
    hidden=np.isin(y[test],hidden_ids)
    for name,cost in [('original',radius),('calibrated',beta)]:
        r=fit_robust_capacity(s[test],a,radius,prior_strength=counts,reference_samples=reference,
                             birth_order='before_update',birth_penalty=cost)
        assignment=r['assignments']; unknown=assignment>=20
        matrix=np.array([[(assignment[(y[test]==c)]==20+j).sum() for j in range(r['K'])]
                         for c in hidden_ids],dtype=int).reshape(5,r['K'])
        matched=0
        if r['K']:
            rows,columns=linear_sum_assignment(-matrix)
            matched=int(matrix[rows,columns].sum())
        arm=dict(name=name,K=r['K'],converged=r['converged'],
            hidden_candidate_recall=float(unknown[hidden].mean()),
            known_false_candidate_fraction=float(unknown[~hidden].mean()),
            one_to_one_hidden_accuracy=matched/int(hidden.sum()),
            hidden_class_by_candidate_counts=matrix.tolist(),
            hidden_absorbed_known=int((hidden & (assignment>=0) & (assignment<20)).sum()),
            hidden_noise=int((hidden & (assignment==-1)).sum()))
        group['arms'].append(arm)
    report['groups'].append(group)
    print('GROUP_DONE',json.dumps(group),flush=True)
out.write_text(json.dumps(report,indent=2))
print('SOURCE_GROUPS_COMPLETE',flush=True)
