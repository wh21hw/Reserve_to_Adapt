"""Saved-geometry known identity takeover costs, no fitting or target labels."""
import json
from pathlib import Path
import numpy as np
root = Path('/content/imp-runs/visda-frozenbn-capacity-10e-v1')
out = root/'identity-takeover-cost-v1.json'
if out.exists(): raise FileExistsError('Preserve evidence')
cap=json.loads((root/'capacity.json').read_text()); s=cap['settings']; c=cap['C']
with np.load(root/'source/features.npz') as f:
    source, labels, x=f['source'].astype(np.float64),f['source_labels'],f['target'].astype(np.float64)
with np.load(root/'capacity.npz') as f: centers=f['centers']
rng=np.random.RandomState(2026); anchors=[]
for cls in range(c):
    ids=np.flatnonzero(labels==cls);rng.shuffle(ids)
    anchors.append(source[ids[:max(1,int(.7*len(ids)))]].mean(0))
a=np.stack(anchors); prior=np.array(s['source_prior_precision'])
w=s['reference_samples']/len(x); radius=s['lambda_radius']; beta=s['birth_cost']
def terms(mu):
    d=np.maximum((x*x).sum(1)[:,None]+(mu*mu).sum(1)[None]-2*x.dot(mu.T),0.)
    return dict(data=float(w*np.minimum(d.min(1),radius).sum()),
        birth=float(beta*(len(mu)-c)),prior=float((prior*((mu[:c]-a)**2).sum(1)).sum()))
base=terms(centers); rows=[]
for j in range(c,len(centers)):
    proposals=[]
    for k in range(c):
        mu=np.delete(centers,j,axis=0).copy();mu[k]=centers[j]
        t=terms(mu); delta={key:t[key]-base[key] for key in base}
        proposals.append(dict(known_slot=k,delta=delta,total_delta=sum(delta.values()),
            source_squared_displacement=float(((centers[j]-a[k])**2).sum())))
    rows.append(dict(candidate_slot=j,best=min(proposals,key=lambda q:q['total_delta']),all=proposals))
report=dict(target_labels_used=False,capacity_refitted=False,RTA_changed=False,
    base_terms=base,candidates=rows,
    caveat='Replace a known center by a saved candidate center, delete that candidate slot, retain all other geometry. Costs use capped reassigned data objective. This is a feasible identity takeover, not optimized merge or semantic proof; no target labels choose identity.')
out.write_text(json.dumps(report,indent=2,allow_nan=False))
print('IDENTITY_TAKEOVER',json.dumps(dict(base_terms=base,best=[r['best'] for r in rows])),flush=True)
