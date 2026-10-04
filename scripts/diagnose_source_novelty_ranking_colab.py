"""Source held-class score ranking, not a threshold optimization procedure."""
import json
from pathlib import Path
import numpy as np
import torch
from sklearn.metrics import roc_auc_score, average_precision_score
from relation_gate import source_relation_scores

torch.set_num_threads(2)
root = Path('/content/imp-runs/source-leaveclass-officehome-v1/seed1')
output = root/'novelty-ranking-v1.json'
if output.exists():
    raise FileExistsError('Preserve previous diagnosis')
x = np.load(root/'source/features.npz')['target'].astype(np.float32)
y = np.array([int(r.rsplit(None,1)[1]) for r in Path('/content/osda-officehome-pr2rw-v1/product_0-24_train_all.txt').read_text().splitlines() if r.strip()])
if len(x) != len(y):
    raise ValueError('Source list mismatch')
hidden_ids = list(range(10,15))
known = [c for c in range(25) if c not in hidden_ids]
mapping = {c:i for i,c in enumerate(known)}
rng = np.random.RandomState(2026)
train, calibration, test = [], [], []
for c in range(25):
    ids = np.flatnonzero(y==c); rng.shuffle(ids)
    cut=max(1,int(.7*len(ids))); middle=cut+(len(ids)-cut)//2
    train.extend(ids[:cut]); calibration.extend(ids[cut:middle]); test.extend(ids[middle:])
train, calibration, test = np.asarray(train),np.asarray(calibration),np.asarray(test)
tr = train[np.isin(y[train],known)]
cr = calibration[np.isin(y[calibration],known)]
labels = np.array([mapping[int(c)] for c in y[tr]])
anchors = np.stack([x[tr][y[tr]==c].mean(0) for c in known])
state = torch.load(str(root/'source/source-final.pt'),map_location='cpu')['model']
@torch.no_grad()
def logits(features):
    v=torch.nn.functional.batch_norm(torch.from_numpy(features),state['1.main.1.0.running_mean'],
        state['1.main.1.0.running_var'],state['1.main.1.0.weight'],state['1.main.1.0.bias'],training=False,eps=1e-5)
    return torch.nn.functional.linear(torch.nn.functional.leaky_relu(v,.2),state['1.fc.weight'])
def relation(ids):
    return source_relation_scores(logits(x[tr]),torch.from_numpy(labels).long(),logits(x[ids]),20)['scores'].numpy()
def distance(ids):
    return ((x[ids,None,:].astype(float)-anchors[None,:,:].astype(float))**2).sum(2).min(1)
hidden=np.isin(y[test],hidden_ids)
report=dict(hidden_ids=hidden_ids,source_only=True,hidden_supervision=False,samples=len(test),
    hidden_count=int(hidden.sum()),known_count=int((~hidden).sum()),scores={},
    caveat='Posthoc pressure block; known images previously supervised; no target labels, optimal threshold search or training')
for name,score,cal in [('relation_kl',relation(test),relation(cr)),('nearest_known_squared_distance',distance(test),distance(cr))]:
    threshold=float(np.quantile(cal,.99)); selected=score>threshold
    report['scores'][name]=dict(AUROC=float(roc_auc_score(hidden,score)),
        average_precision=float(average_precision_score(hidden,score)),source99_threshold=threshold,
        known_false_candidate_rate=float(selected[~hidden].mean()),hidden_recall=float(selected[hidden].mean()),
        known_quantiles=np.quantile(score[~hidden],[.1,.5,.9,.99]).tolist(),
        hidden_quantiles=np.quantile(score[hidden],[.1,.5,.9,.99]).tolist(),
        per_hidden_class_recall={str(c):float(selected[y[test]==c].mean()) for c in hidden_ids})
output.write_text(json.dumps(report,indent=2,allow_nan=False))
print('NOVELTY_RANKING',json.dumps(report),flush=True)
