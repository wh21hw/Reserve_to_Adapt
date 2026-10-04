"""Final-only source held-class diagnosis for the retention experiment."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from sklearn.metrics import roc_auc_score,average_precision_score

torch.set_num_threads(2)
parser=argparse.ArgumentParser()
parser.add_argument('--root',default='/content/imp-runs/source-retention-officehome-v1/seed1')
parser.add_argument('--variant-name',default='source_retention')
args=parser.parse_args()
root=Path(args.root)
old=Path('/content/imp-runs/source-leaveclass-officehome-v1/seed1')
output=root/'comparison.json'
if output.exists():raise FileExistsError('Preserve existing diagnosis')
y=np.array([int(r.rsplit(None,1)[1]) for r in Path('/content/osda-officehome-pr2rw-v1/product_0-24_train_all.txt').read_text().splitlines() if r.strip()])
known=[c for c in range(25) if c not in range(10,15)];mapping={c:i for i,c in enumerate(known)}
rng=np.random.RandomState(2026);train,calibration,test=[],[],[]
for c in range(25):
    ids=np.flatnonzero(y==c);rng.shuffle(ids);cut=max(1,int(.7*len(ids)));middle=cut+(len(ids)-cut)//2
    train.extend(ids[:cut]);calibration.extend(ids[cut:middle]);test.extend(ids[middle:])
train,calibration,test=map(np.asarray,(train,calibration,test));tr=train[np.isin(y[train],known)];cr=calibration[np.isin(y[calibration],known)]
hidden=np.isin(y[test],range(10,15));kt=test[~hidden]
report=dict(source_only=True,real_target_used=False,final_epoch=3,hidden_classes=list(range(10,15)),
    selection='Predeclared source variant, final3; no target/hidden label epoch, coefficient or model selection',
    caveat='Single seed, posthoc stress block; known test images participated in source network supervision. Not RTA performance.',arms={})
for name,base in [('source_ce',old),(args.variant_name,root)]:
    checkpoint=torch.load(str(base/'source/source-final.pt'),map_location='cpu')
    if checkpoint['config']['epochs']!=3 or len(checkpoint['history'])!=3:
        raise RuntimeError('Training budget mismatch')
    state=checkpoint['model'];bottle=np.load(base/'source/features.npz')['target']
    backbone_path=base/'backbone-features.npz' if name=='source_ce' else base/'source/backbone-features.npz'
    backbone=np.load(backbone_path)['target']
    with torch.no_grad():
        v=torch.nn.functional.batch_norm(torch.from_numpy(bottle[kt]),state['1.main.1.0.running_mean'],
            state['1.main.1.0.running_var'],state['1.main.1.0.weight'],state['1.main.1.0.bias'],training=False,eps=1e-5)
        pred=torch.nn.functional.linear(torch.nn.functional.leaky_relu(v,.2),state['1.fc.weight']).argmax(1).numpy()
    arm=dict(retention_weight=checkpoint['config'].get('retention_weight',0),
        freeze_backbone_bn=checkpoint['config'].get('freeze_backbone_bn',False),
        known_classifier_accuracy=float((pred==np.array([mapping[int(c)] for c in y[kt]])).mean()),
        training_history=checkpoint['history'],scores={})
    for feature_name,x in [('bottleneck',bottle),('backbone',backbone)]:
        x=x.astype(np.float64)
        if len(x)!=len(y) or not np.isfinite(x).all():raise ValueError('Source features mismatch')
        anchors=np.stack([x[tr][y[tr]==c].mean(0) for c in known])
        def distance(ids):return ((x[ids,None,:]-anchors[None,:,:])**2).sum(2).min(1)
        score=distance(test);threshold=float(np.quantile(distance(cr),.99));selected=score>threshold
        arm['scores'][feature_name]=dict(AUROC=float(roc_auc_score(hidden,score)),AP=float(average_precision_score(hidden,score)),
            threshold99=threshold,known_false_candidate_rate=float(selected[~hidden].mean()),hidden_recall=float(selected[hidden].mean()),
            rejected_known_count=int(selected[~hidden].sum()),selected_hidden_count=int(selected[hidden].sum()),
            per_hidden_class_recall={str(c):float(selected[y[test]==c].mean()) for c in range(10,15)})
    report['arms'][name]=arm
output.write_text(json.dumps(report,indent=2,allow_nan=False))
print('SOURCE_RETENTION_COMPARISON',json.dumps(report),flush=True)
