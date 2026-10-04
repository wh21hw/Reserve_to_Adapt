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
output = root/'unlabeled-known-compatibility-head-v2.json'
if output.exists():
    raise FileExistsError('Preserve diagnostic')
with np.load(prior/'features.npz') as data:
    source, labels, target = data['source'], data['source_labels'], data['target']
with np.load(root/'clusters.npz') as data:
    assignments = data['assignments']
state = torch.load(str(prior/'source-final.pt'), map_location='cpu')['model']
weights = state['1.fc.weight'].numpy()
if weights.shape != (25, 256) or target.shape != (4357, 256) or len(assignments) != len(target):
    raise ValueError('Expected saved C25 feature/head interface')
# Released CLS applies BN and LeakyReLU to saved normalized bottle features,
# then the bias-free fc (temperature1). Omitting these yields wrong scores.
mean = state['1.main.1.0.running_mean'].numpy().astype(np.float64)
variance = state['1.main.1.0.running_var'].numpy().astype(np.float64)
scale = state['1.main.1.0.weight'].numpy().astype(np.float64)
shift = state['1.main.1.0.bias'].numpy().astype(np.float64)
def probabilities(features):
    hidden = (features.astype(np.float64)-mean)/np.sqrt(variance+1e-5)*scale+shift
    hidden = np.where(hidden >= 0, hidden, .2*hidden)
    z = hidden.dot(weights.astype(np.float64).T)
    z -= z.max(1, keepdims=True)
    p = np.exp(z)
    return p/p.sum(1, keepdims=True)
# One focused equivalence check on saved features only, no backbone/images.
head = torch.nn.Sequential(torch.nn.BatchNorm1d(256), torch.nn.LeakyReLU(.2),
                           torch.nn.Linear(256, 25, bias=False)).eval()
head.load_state_dict({k: state['1.main.1.'+k] for k in head.state_dict()})
with torch.no_grad():
    reference = torch.softmax(head(torch.from_numpy(source[:8])), dim=1).numpy()
if not np.allclose(reference, probabilities(source[:8]), rtol=1e-5, atol=1e-7):
    raise RuntimeError('Saved-feature head calculation conflicts with released head')
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
    implementation='Corrected BN -> LeakyReLU -> fc -> softmax, eval statistics',
    supersedes='unlabeled-known-compatibility.json omitted head BN/activation; its scores are invalid',
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
