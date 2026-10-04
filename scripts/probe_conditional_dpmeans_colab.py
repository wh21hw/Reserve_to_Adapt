"""One predeclared initial A2W gate -> candidate-only DP-means probe."""
import json
from pathlib import Path
import numpy as np
import torch
from sklearn.mixture import BayesianGaussianMixture
from relation_gate import source_relation_scores
from conditional_dpmeans import fit_candidate_dpmeans

torch.set_num_threads(2)
prior = Path('/content/imp-runs/fusion-imp-rta-v1/seed1/source')
output = Path('/content/conditional-dpmeans-a2w-v1.json')
if output.exists():
    raise FileExistsError('Preserve previous probe')
data = np.load(prior/'features.npz')
s, y, x = data['source'], data['source_labels'], data['target']
anchors = np.stack([s[y == c].mean(0) for c in range(10)])
penalty = max(float(np.quantile(((s-anchors[y])**2).sum(1), .99)), 1e-8)
# Existing cached feature is the normalized bottleneck, before CLS BatchNorm.
# Apply the saved eval head only; no image inference, backbone or SGD needed.
state = torch.load(str(prior/'source-final.pt'), map_location='cpu')['model']
@torch.no_grad()
def logits(features):
    values = torch.nn.functional.batch_norm(torch.from_numpy(features),
        state['1.main.1.0.running_mean'], state['1.main.1.0.running_var'],
        state['1.main.1.0.weight'], state['1.main.1.0.bias'], training=False, eps=1e-5)
    values = torch.nn.functional.leaky_relu(values, negative_slope=.2)
    return torch.nn.functional.linear(values, state['1.fc.weight'])
scores = source_relation_scores(logits(s), torch.from_numpy(y).long(), logits(x), 10)['scores'].numpy()
mixture = BayesianGaussianMixture(n_components=4, max_iter=800, random_state=2026).fit(scores[:,None])
known_component = int(mixture.means_.argmin())
known_probability = mixture.predict_proba(scores[:,None])[:,known_component]
pool = known_probability < .5
if not pool.any() or not mixture.converged_:
    raise RuntimeError('Empty pool or unconverged gate; do not alter the rule')
fit = fit_candidate_dpmeans(x[pool], penalty)
if not fit['converged']:
    raise RuntimeError('DP-means not converged; do not report final K')
# All fit decisions precede label loading; diagnostics do not pick thresholds.
truth = np.array([int(r.rsplit(None,1)[1]) for r in Path('/content/webcam_0-9_20-30_test.txt').read_text().splitlines() if r.strip()])
if len(truth) != len(x):
    raise ValueError('Target row mismatch')
unknown = truth >= 20
components = []
for j in range(fit['K']):
    labels = truth[pool][fit['assignments'] == j]
    values, counts = np.unique(labels, return_counts=True)
    components.append(dict(slot=j, support=len(labels), known=int((labels<10).sum()),
                           unknown=int((labels>=20).sum()), composition={str(int(v)):int(n) for v,n in zip(values,counts)}))
report = dict(task='Office31 A->W', seed=1, source_epochs=3, K=fit['K'], penalty=penalty,
              rule='Known mixture probability < 0.5; DP-means only inside candidate pool',
              gate=dict(components=4, random_state=2026, known_component=known_component,
                        means=mixture.means_.ravel().tolist(), converged=bool(mixture.converged_)),
              pool_size=int(pool.sum()), pool_unknown_recall=float(pool[unknown].mean()),
              pool_precision=float(unknown[pool].mean()), known_false_candidate_rate=float(pool[~unknown].mean()),
              counts=fit['counts'].tolist(), components=components, history=fit['history'],
              target_labels_used_for_fit=False,
              caveat='No RTA training; gating may mistake domain shift for novelty; neither count nor purity proves semantic discovery')
output.write_text(json.dumps(report, indent=2, allow_nan=False))
np.savez_compressed('/content/conditional-dpmeans-a2w-v1.npz',pool=pool, assignments=fit['assignments'], centers=fit['centers'])
print('CONDITIONAL_DPMEANS', json.dumps(report), flush=True)
