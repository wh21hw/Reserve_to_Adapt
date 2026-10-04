"""Source-only leave-class control for fixed gate -> candidate DP-means.

Uses an existing C20 model with classes10..14 absent from its supervision.
No target data, training, threshold search or oracle pool in the estimator.
"""
import json
import argparse
from pathlib import Path
import numpy as np
import torch
from sklearn.mixture import BayesianGaussianMixture
from scipy.optimize import linear_sum_assignment
from relation_gate import source_relation_scores
from conditional_dpmeans import fit_candidate_dpmeans

torch.set_num_threads(2)
root = Path('/content/imp-runs/source-leaveclass-officehome-v1/seed1')
parser = argparse.ArgumentParser()
parser.add_argument('--gate',choices=['mixture','source99'],default='mixture')
args = parser.parse_args()
output = root/('conditional-dpmeans-controls-v1.json' if args.gate=='mixture' else 'conditional-dpmeans-source99-controls-v1.json')
if output.exists():
    raise FileExistsError('Preserve prior controls')
x = np.load(root/'source/features.npz')['target'] # SOURCE filenames only.
rows = [r.rsplit(None,1) for r in Path('/content/osda-officehome-pr2rw-v1/product_0-24_train_all.txt').read_text().splitlines() if r.strip()]
y = np.array([int(r[1]) for r in rows])
if len(x) != len(y):
    raise ValueError('Source rows mismatch')
hidden_ids = list(range(10,15))
known = [c for c in range(25) if c not in hidden_ids]
mapping = {c:i for i,c in enumerate(known)}
rng = np.random.RandomState(2026)
train, calibration, test = [], [], []
for c in range(25):
    ids = np.flatnonzero(y==c); rng.shuffle(ids)
    cut = max(1,int(.7*len(ids))); middle = cut+(len(ids)-cut)//2
    train.extend(ids[:cut]); calibration.extend(ids[cut:middle]); test.extend(ids[middle:])
train, calibration, test = np.asarray(train), np.asarray(calibration), np.asarray(test)
tr = train[np.isin(y[train],known)]
cr = calibration[np.isin(y[calibration],known)]
train_labels = np.array([mapping[int(c)] for c in y[tr]])
anchors = np.stack([x[tr][y[tr]==c].mean(0) for c in known])
penalty = max(float(np.quantile(((x[tr]-anchors[train_labels])**2).sum(1),.99)),1e-8)
state = torch.load(str(root/'source/source-final.pt'),map_location='cpu')['model']
@torch.no_grad()
def logits(features):
    v = torch.nn.functional.batch_norm(torch.from_numpy(features),
        state['1.main.1.0.running_mean'],state['1.main.1.0.running_var'],
        state['1.main.1.0.weight'],state['1.main.1.0.bias'],training=False,eps=1e-5)
    return torch.nn.functional.linear(torch.nn.functional.leaky_relu(v,.2),state['1.fc.weight'])
source_score = source_relation_scores(logits(x[tr]),torch.from_numpy(train_labels).long(),logits(x[cr]),20)['scores'].numpy()
gate_threshold = float(np.quantile(source_score,.99))
report = dict(hidden_ids=hidden_ids,real_target_used=False,hidden_supervision=False,
              penalty=penalty,gate=args.gate,source_score_threshold99=gate_threshold,
              rule='Only gate changes; DP-means penalty remains source99% squared radius',
              caveat='Posthoc chosen source stress block; known test images participated in model supervision',arms=[])
for name, ids in [('known-negative',test[np.isin(y[test],known)]),('mixed-heldclass',test)]:
    score = source_relation_scores(logits(x[tr]),torch.from_numpy(train_labels).long(),logits(x[ids]),20)['scores'].numpy()
    means = None
    if args.gate == 'mixture':
        mixture = BayesianGaussianMixture(n_components=4,max_iter=800,random_state=2026).fit(score[:,None])
        if not mixture.converged_:
            raise RuntimeError('Unconverged gate; no rule change')
        known_component = int(mixture.means_.argmin())
        pool = mixture.predict_proba(score[:,None])[:,known_component] < .5
        means = mixture.means_.ravel().tolist()
    else:
        pool = score > gate_threshold
    fit = fit_candidate_dpmeans(x[ids][pool],penalty) if pool.any() else None
    if fit is not None and not fit['converged']:
        raise RuntimeError('Unconverged clustering')
    hidden = np.isin(y[ids],hidden_ids)
    k = fit['K'] if fit is not None else 0
    matrix = np.zeros((5,k),dtype=int)
    if fit is not None:
        for ci,c in enumerate(hidden_ids):
            for j in range(k):
                matrix[ci,j] = int(((y[ids][pool]==c)&(fit['assignments']==j)).sum())
    matched = 0
    if k:
        rr,cc = linear_sum_assignment(-matrix); matched = int(matrix[rr,cc].sum())
    report['arms'].append(dict(name=name,K=k,pool_size=int(pool.sum()),
        known_false_candidate_rate=float(pool[~hidden].mean()),
        hidden_candidate_recall=float(pool[hidden].mean()) if hidden.any() else None,
        one_to_one_hidden_accuracy=matched/int(hidden.sum()) if hidden.any() else None,
        counts=fit['counts'].tolist() if fit else [],hidden_class_by_candidate_counts=matrix.tolist(),
        means=means,converged=True))
output.write_text(json.dumps(report,indent=2,allow_nan=False))
print('CONDITIONAL_SOURCE_CONTROL',json.dumps(report),flush=True)
