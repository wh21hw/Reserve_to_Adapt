"""Frozen source leave-class diagnostic; labels only from the source-domain list."""
import json
import argparse
from pathlib import Path
import numpy as np
from scipy.optimize import linear_sum_assignment
from robust_capacity import fit_robust_capacity

parser=argparse.ArgumentParser()
parser.add_argument('--root',default='/content/imp-runs/source-leaveclass-officehome-v1/seed1')
parser.add_argument('--features',default='source/features.npz')
parser.add_argument('--output',default='capacity-controls.json')
parser.add_argument('--metric',choices=['euclidean','source-shrinkage'],default='euclidean')
args=parser.parse_args()
root=Path(args.root)
feature_path=(root/args.features).resolve()
out=(root/args.output).resolve()
if root.resolve() not in feature_path.parents or root.resolve() not in out.parents:
    raise ValueError('Feature and output paths must stay inside this experiment')
if out.exists():
    raise FileExistsError('Preserve previous inference')
f=np.load(feature_path)
x=f['target'].astype(float) # This artifact contains all SOURCE-domain filenames.
rows=[line.rsplit(None,1) for line in Path('/content/osda-officehome-pr2rw-v1/product_0-24_train_all.txt').read_text().splitlines() if line.strip()]
y=np.asarray([int(row[1]) for row in rows])
if len(x)!=len(y):
    raise ValueError('Source extraction row count mismatch')
hidden_ids=list(range(10,15)); known=[c for c in range(25) if c not in hidden_ids]
rng=np.random.RandomState(2026); train,cost_rows,test=[],[],[]
for c in range(25):
    ids=np.flatnonzero(y==c); rng.shuffle(ids)
    cut=max(1,int(.7*len(ids))); middle=cut+(len(ids)-cut)//2
    train.extend(ids[:cut]); cost_rows.extend(ids[cut:middle]); test.extend(ids[middle:])
train,cost_rows,test=map(np.asarray,(train,cost_rows,test))
tr=train[np.isin(y[train],known)]; cr=cost_rows[np.isin(y[cost_rows],known)]
a=np.stack([x[tr][y[tr]==c].mean(0) for c in known])
counts=np.array([(y[tr]==c).sum() for c in known]); remap={c:i for i,c in enumerate(known)}
metric_info=dict(name=args.metric)
if args.metric=='source-shrinkage':
    from sklearn.covariance import ledoit_wolf
    residuals=x[tr]-a[[remap[int(c)] for c in y[tr]]]
    covariance,shrinkage=ledoit_wolf(residuals,assume_centered=True)
    eigenvalues,vectors=np.linalg.eigh(covariance)
    floor=max(float(eigenvalues.max())*1e-12,1e-12)
    transform=vectors/np.sqrt(np.maximum(eigenvalues,floor))[None,:]
    x=x.dot(transform); a=a.dot(transform)
    metric_info.update(shrinkage=float(shrinkage),condition_number=float(eigenvalues.max()/max(eigenvalues.min(),floor)),
                       source_calibration_rows=len(tr),eigenvalue_floor=floor,renormalized=False)
radius=max(float(np.quantile(((x[tr]-a[[remap[int(c)] for c in y[tr]]])**2).sum(1),.99)),1e-8)
reference=float(counts.mean())
def d(left,right):
    return np.maximum((left*left).sum(1)[:,None]+(right*right).sum(1)[None]-2*left.dot(right.T),0.)
residual=np.minimum(d(x[cr],a).min(1),radius)
beta=max(float((reference/len(cr)*np.maximum(residual[:,None]-d(x[cr],x[cr]),0.).sum(0)).max())*(1+1e-6),1e-8)
report=dict(hidden_ids=hidden_ids,beta=beta,lambda_radius=radius,real_target_used=False,
    feature_file=args.features,feature_dimension=x.shape[1],
    metric=metric_info,
    hidden_supervision=False,caveat='posthoc chosen block; known test images participated in known supervision',arms=[])
for name,ids,cost in [('known-negative',test[np.isin(y[test],known)],beta),
                     ('original-cost',test,radius),('calibrated-cost',test,beta)]:
    r=fit_robust_capacity(x[ids],a,radius,prior_strength=counts,reference_samples=reference,
        birth_order='before_update',birth_penalty=cost)
    assignment=r['assignments']; hidden=np.isin(y[ids],hidden_ids); unknown=assignment>=20
    matrix=np.array([[(assignment[y[ids]==c]==20+j).sum() for j in range(r['K'])]
                     for c in hidden_ids],dtype=int).reshape(5,r['K'])
    matched=0
    if r['K']:
        rr,cc=linear_sum_assignment(-matrix); matched=int(matrix[rr,cc].sum())
    report['arms'].append(dict(name=name,K=r['K'],converged=r['converged'],
        known_false_candidate_fraction=float(unknown[~hidden].mean()),
        hidden_candidate_recall=float(unknown[hidden].mean()) if hidden.any() else None,
        one_to_one_hidden_accuracy=matched/int(hidden.sum()) if hidden.any() else None,
        hidden_class_by_candidate_counts=matrix.tolist(),noise_count=r['noise_count']))
out.write_text(json.dumps(report,indent=2))
print('LEAVECLASS_CAPACITY',json.dumps(report),flush=True)
