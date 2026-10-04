"""Unlabeled cluster compatibility probe using saved features and source head only.

No image/checkpoint forward, target semantic labels, pruning, or new training.
Source quantile below is descriptive, not a production rejection threshold.
"""
import json
from pathlib import Path
import numpy as np
import torch

root = Path('/content/imp-runs/officehome-cluster-identity-10e-v1')
prior = Path('/content/imp-runs/officehome-frozenbn-capacity-10e-v1/prior/source')
output = root/'unlabeled-known-compatibility.json'
if output.exists():
    raise FileExistsError('Preserve diagnostic')
with np.load(prior/'features.npz') as data:
    source, labels, target = data['source'], data['source_labels'], data['target']
with np.load(root/'clusters.npz') as data:
    assignments = data['assignments']
weights = torch.load(str(prior/'source-final.pt'), map_location='cpu')['model']['1.fc.weight'].numpy()
if weights.shape != (25, 256) or target.shape != (4357, 256) or len(assignments) != len(target):
    raise ValueError('Expected saved C25 feature/head interface')
# Released CLS uses normalized saved bottle features, bias-free fc, temperature1.
def probabilities(features):
    z = features.astype(np.float64).dot(weights.astype(np.float64).T)
    z -= z.max(1, keepdims=True)
    p = np.exp(z)
    return p/p.sum(1, keepdims=True)
ps, pt = probabilities(source), probabilities(target)
q = np.stack([ps[labels == c].mean(0) for c in range(25)])
def score(p):
    predicted = p.argmax(1)
    reference = q[predicted]
    return predicted, (reference*(np.log(np.maximum(reference, 1e-12))
                      -np.log(np.maximum(p, 1e-12)))).sum(1)
_, ks = score(ps)
predicted, kt = score(pt)
cut = float(np.quantile(ks, .99))
report = dict(purpose='Unlabeled compatibility diagnosis only; no K/assignment/threshold change',
    target_labels_used=False, source_KL_quantile99=cut,
    caveat='Source in-sample descriptive quantile; not validated target coverage or semantic novelty test', clusters=[])
for cluster in (25, 26):
    mask = assignments == cluster
    counts = np.bincount(predicted[mask], minlength=25)
    report['clusters'].append(dict(cluster=cluster, rows=int(mask.sum()),
        dominant_predicted_known_class=int(counts.argmax()),
        dominant_known_agreement=float(counts.max()/mask.sum()),
        mean_max_known_probability=float(pt[mask].max(1).mean()),
        mean_KL=float(kt[mask].mean()), median_KL=float(np.median(kt[mask])),
        fraction_within_source_descriptive_KL99=float((kt[mask] <= cut).mean())))
output.write_text(json.dumps(report, indent=2, allow_nan=False))
print('UNLABELED_KNOWN_COMPATIBILITY', json.dumps(report), flush=True)
