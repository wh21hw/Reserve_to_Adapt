"""Class-held-out source proxy novelty learner; no real target inputs."""
import json
from pathlib import Path
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score
from relation_gate import source_relation_scores

torch.set_num_threads(2)
root = Path('/content/imp-runs/source-leaveclass-officehome-v1/seed1')
output = root/'proxy-novelty-classfolds-v1.json'
if output.exists():
    raise FileExistsError('Preserve previous probe')
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
# Features are defined before novelty labels are supplied to the learner.
relation=source_relation_scores(logits(x[tr]),torch.from_numpy(labels).long(),logits(x),20)['scores'].numpy()
distance=((x[:,None,:].astype(float)-anchors[None,:,:].astype(float))**2).sum(2).min(1)
z=np.column_stack([relation,distance])
report=dict(source_only=True,hidden_ids=hidden_ids,representation_hidden_supervision=False,
    detector_uses_source_proxy_labels=True,features=['relation_kl','nearest_known_squared_distance'],
    model=dict(type='StandardScaler + LogisticRegression',C=1.,class_weight='balanced',solver='liblinear',random_state=2026),
    threshold_rule='99% quantile of known-source calibration scores, same rule for all methods',
    caveat='Five class folds, not five seeds; posthoc pressure block; known images previously supervised; no target access or tuning',folds=[])
for held in hidden_ids:
    positive_classes=[c for c in hidden_ids if c!=held]
    fit_ids=train[(np.isin(y[train],known)) | np.isin(y[train],positive_classes)]
    fit_labels=np.isin(y[fit_ids],positive_classes).astype(int)
    eval_ids=test[np.isin(y[test],known) | (y[test]==held)]
    truth=y[eval_ids]==held
    if held in set(y[fit_ids]) or np.intersect1d(fit_ids,eval_ids).size or np.intersect1d(fit_ids,cr).size:
        raise ValueError('Class or row leakage')
    scaler=StandardScaler().fit(z[fit_ids])
    model=LogisticRegression(C=1.,class_weight='balanced',solver='liblinear',random_state=2026,max_iter=1000)
    model.fit(scaler.transform(z[fit_ids]),fit_labels)
    learned=model.decision_function(scaler.transform(z[eval_ids]))
    cal_learned=model.decision_function(scaler.transform(z[cr]))
    fold=dict(held_class=held,train_proxy_classes=positive_classes,train_samples=len(fit_ids),
              test_hidden=int(truth.sum()),test_known=int((~truth).sum()),
              standardized_coefficients=model.coef_[0].tolist(),methods={})
    for name,score,cal in [('relation',relation[eval_ids],relation[cr]),('distance',distance[eval_ids],distance[cr]),
                           ('learned_joint',learned,cal_learned)]:
        threshold=float(np.quantile(cal,.99)); selected=score>threshold
        fold['methods'][name]=dict(AUROC=float(roc_auc_score(truth,score)),AP=float(average_precision_score(truth,score)),
            threshold=threshold,known_false_candidate_rate=float(selected[~truth].mean()),
            hidden_recall=float(selected[truth].mean()))
    report['folds'].append(fold)
report['macro_classfold_mean']={name:{metric:float(np.mean([f['methods'][name][metric] for f in report['folds']]))
    for metric in ['AUROC','AP','known_false_candidate_rate','hidden_recall']} for name in ['relation','distance','learned_joint']}
output.write_text(json.dumps(report,indent=2,allow_nan=False))
print('SOURCE_PROXY_NOVELTY',json.dumps(report),flush=True)
