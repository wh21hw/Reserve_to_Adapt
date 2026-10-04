"""Exploratory source-only cost calibration; not a DP posterior or semantic proof."""
import json
from pathlib import Path
import numpy as np
from robust_capacity import fit_robust_capacity

root = Path('/content/imp-runs/source-precision-officehome-v1/seed1')
out = root/'source-birth-calibration.json'
if out.exists():
    raise FileExistsError('Preserve previous probe')
f = np.load(root/'source/features.npz')
s, y = f['source'].astype(float), f['source_labels']
rng = np.random.RandomState(2026)
train, cost_rows, test = [], [], []
for c in range(25):
    ids = np.flatnonzero(y == c); rng.shuffle(ids)
    cut = max(1, int(.7*len(ids)))
    middle = cut+(len(ids)-cut)//2
    train.extend(ids[:cut]); cost_rows.extend(ids[cut:middle]); test.extend(ids[middle:])
train, cost_rows, test = map(np.asarray, (train, cost_rows, test))
def d(left, right):
    return np.maximum((left*left).sum(1)[:,None]+(right*right).sum(1)[None]-2*left.dot(right.T),0.)
reports = []
for known in (25,20):
    tr = train[y[train] < known]
    cr = cost_rows[y[cost_rows] < known]
    anchors = np.stack([s[tr][y[tr] == c].mean(0) for c in range(known)])
    counts = np.array([(y[tr] == c).sum() for c in range(known)])
    radius = max(float(np.quantile(((s[tr]-anchors[y[tr]])**2).sum(1),.99)),1e-8)
    reference = float(counts.mean())
    residual = np.minimum(d(s[cr],anchors).min(1),radius)
    raw_gain = reference/len(cr)*np.maximum(residual[:,None]-d(s[cr],s[cr]),0.).sum(0)
    cost = max(float(raw_gain.max())*(1+1e-6),1e-8)
    result = fit_robust_capacity(s[test],anchors,radius,prior_strength=counts,
        reference_samples=reference,birth_order='before_update',birth_penalty=cost)
    unknown = result['assignments'] >= known
    hidden = y[test] >= known
    reports.append(dict(known=known, held_out_classes=list(range(known,25)),
        lambda_radius=radius,birth_cost=cost,reference_samples=reference,
        calibrated_on=len(cr),tested_on=len(test),K=result['K'],converged=result['converged'],
        known_false_candidate_fraction=float(unknown[~hidden].mean()),
        hidden_candidate_recall=float(unknown[hidden].mean()) if hidden.any() else None,
        noise_count=result['noise_count'],history=result['history']))
report = dict(version='source-birth-calibration-probe-v1',target_used=False,
    calibration='70% anchors; half remaining known rows calibrate maximum initial birth gain; remainder tests',
    caveat='Network supervised on all 25 source classes; held-out geometry is not unseen-class generalization',
    reports=reports)
out.write_text(json.dumps(report,indent=2))
print('SOURCE_BIRTH_CALIBRATION',json.dumps(report),flush=True)
