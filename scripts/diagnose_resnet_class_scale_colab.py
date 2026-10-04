"""Source-only heteroskedastic prototype distance; no target or network update."""
import json
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score

root=Path('/content/imp-runs/source-leaveclass-officehome-v1/seed1')
output=root/'class-scale-diagnosis-v1.json'
if output.exists():
    raise FileExistsError('Preserve existing result')
x=np.load(root/'source/features.npz')['target'].astype(np.float64)  # All Product SOURCE rows.
y=np.array([int(r.rsplit(None,1)[1]) for r in Path('/content/osda-officehome-pr2rw-v1/product_0-24_train_all.txt').read_text().splitlines() if r.strip()])
if len(x)!=len(y) or not np.isfinite(x).all():
    raise ValueError('Invalid source features')
known=[c for c in range(25) if c not in range(10,15)]
rng=np.random.RandomState(2026)
train,calibration,test=[],[],[]
for c in range(25):
    ids=np.flatnonzero(y==c);rng.shuffle(ids)
    cut=max(1,int(.7*len(ids)));middle=cut+(len(ids)-cut)//2
    train.extend(ids[:cut]);calibration.extend(ids[cut:middle]);test.extend(ids[middle:])
train,calibration,test=map(np.asarray,(train,calibration,test))
tr=train[np.isin(y[train],known)];cr=calibration[np.isin(y[calibration],known)]
anchors=np.stack([x[tr][y[tr]==c].mean(0) for c in known])
# Source training residual mean is a per-class scale, not a target-tuned radius.
scales=np.array([max(float(((x[tr][y[tr]==c]-anchors[i])**2).sum(1).mean()),1e-8) for i,c in enumerate(known)])
def distances(ids):
    return ((x[ids,None,:]-anchors[None,:,:])**2).sum(2)
dtest=distances(test);dcal=distances(cr)
hidden=np.isin(y[test],range(10,15))
report=dict(real_target_used=False,network_updated=False,formal_RTA_changed=False,
    rule='s_c = source training mean squared within-class radius; score = min_c squared_distance/s_c; threshold = known calibration score99. No threshold/scale search.',
    caveat='Posthoc held-class stress block, known test images used in backbone supervision; ratio distance is not a normalized Bayesian likelihood or posterior.',
    train_known_count=len(tr),calibration_known_count=len(cr),test_known_count=int((~hidden).sum()),test_hidden_count=int(hidden.sum()),
    source_class_scales={str(c):float(scales[i]) for i,c in enumerate(known)},arms={})
for name,denom in [('global_euclidean',np.ones(len(known))),('source_class_scaled',scales)]:
    dt=dtest/denom;dc=dcal/denom
    score=dt.min(1);nearest=np.array(known)[dt.argmin(1)]
    threshold=float(np.quantile(dc.min(1),.99));selected=score>threshold
    accepted_hidden=hidden&~selected
    report['arms'][name]=dict(AUROC=float(roc_auc_score(hidden,score)),average_precision=float(average_precision_score(hidden,score)),
        threshold99=threshold,known_false_candidate_rate=float(selected[~hidden].mean()),hidden_recall=float(selected[hidden].mean()),
        rejected_known_count=int(selected[~hidden].sum()),selected_hidden_count=int(selected[hidden].sum()),
        per_hidden_class_recall={str(c):float(selected[y[test]==c].mean()) for c in range(10,15)},
        per_known_class_false_candidate_rate={str(c):float(selected[y[test]==c].mean()) for c in known},
        missed_hidden_nearest_known_counts={str(c):int((accepted_hidden&(nearest==c)).sum()) for c in known},
        hidden_class_to_accepted_known_counts={str(h):{str(c):int(((y[test]==h)&~selected&(nearest==c)).sum()) for c in known} for h in range(10,15)})
output.write_text(json.dumps(report,indent=2,allow_nan=False))
print('SOURCE_CLASS_SCALE_DIAGNOSIS',json.dumps(report),flush=True)
