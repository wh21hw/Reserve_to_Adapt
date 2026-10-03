"""Fixed relation/mixture audit of source-only logits; not official RTA training."""
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import warnings
import zipfile

import numpy as np
import sklearn
from sklearn.mixture import BayesianGaussianMixture
import torch

module_path = Path('/content/relation_gate.py')
spec = importlib.util.spec_from_file_location('relation_gate', module_path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
# KL semantics and extreme-logit safety independent of real target labels.
simple_logits = torch.tensor([[2., 0.], [0., 2.]])
simple_labels = torch.tensor([0, 1])
simple = module.source_relation_scores(simple_logits, simple_labels, simple_logits, 2)
assert torch.allclose(simple['scores'], torch.zeros(2, dtype=torch.float64), atol=1e-10)
extreme = module.source_relation_scores(simple_logits, simple_labels, torch.tensor([[10000., -10000.]]), 2)
assert torch.isfinite(extreme['scores']).all() and extreme['scores'].item() > 0
print('RELATION_UNIT_PASS', flush=True)
feature_path = Path('/content/imp-runs/source-warmup-l4-seed1-v2/features.npz')
data = np.load(feature_path, allow_pickle=False)
assert 'target_labels' not in data.files
result = module.source_relation_scores(torch.from_numpy(data['source_logits']),
    torch.from_numpy(data['source_labels']), torch.from_numpy(data['target_logits']), 10)
scores = result['scores'].numpy()
source_result = module.source_relation_scores(torch.from_numpy(data['source_logits']),
    torch.from_numpy(data['source_labels']), torch.from_numpy(data['source_logits']), 10)
output = Path('/content/imp-runs/relation-gate-frozen-v1')
if output.exists():
    raise RuntimeError('Refusing overwrite')
output.mkdir(parents=True)
np.savez_compressed(output/'relation-scores.npz', scores=scores,
    source_scores=source_result['scores'].numpy(), pseudo_class=result['pseudo_class'].numpy(),
    source_soft_prototypes=result['source_soft_prototypes'].numpy())
report = dict(feature_sha256=hashlib.sha256(feature_path.read_bytes()).hexdigest(),
    module_sha256=hashlib.sha256(module_path.read_bytes()).hexdigest(), sklearn=sklearn.__version__,
    meaning='Fixed full-data BGMM audit, not official minibatch/epoch protocol; no semantic class count',
    source_score_quantiles=np.quantile(source_result['scores'].numpy(), [0,.25,.5,.75,1]).tolist(),
    target_score_quantiles=np.quantile(scores, [0,.25,.5,.75,1]).tolist(), trials=[])
for seed in [1, 2, 3]:
    model = BayesianGaussianMixture(n_components=4, max_iter=800, random_state=seed)
    with warnings.catch_warnings(record=True) as messages:
        warnings.simplefilter('always')
        model.fit(scores[:, None])
    posterior = model.predict_proba(scores[:, None])
    known_index, unknown_index = int(model.means_.argmin()), int(model.means_.argmax())
    hard = posterior.argmax(1)
    np.savez_compressed(output/('seed%d.npz' % seed), posterior=posterior,
        known_probability=posterior[:, known_index], highest_kl_probability=posterior[:, unknown_index],
        highest_kl_flag=hard == unknown_index, nonlowest_kl_flag=hard != known_index)
    row = dict(seed=seed, converged=bool(model.converged_), iterations=int(model.n_iter_),
        means=model.means_.ravel().tolist(), weights=model.weights_.tolist(),
        known_component=known_index, highest_kl_component=unknown_index,
        hard_counts=np.bincount(hard, minlength=4).tolist(),
        highest_kl_samples=int((hard == unknown_index).sum()),
        nonlowest_kl_samples=int((hard != known_index).sum()),
        warnings=[str(message.message) for message in messages])
    report['trials'].append(row)
    print('RELATION_MIXTURE', json.dumps(row), flush=True)
(output/'report.json').write_text(json.dumps(report, indent=2, allow_nan=False))
shutil.copy2(module_path, output/module_path.name)
archive = Path('/content/relation-gate-frozen-v1.zip')
with zipfile.ZipFile(archive, 'x', zipfile.ZIP_DEFLATED) as bundle:
    for path in output.iterdir():
        bundle.write(path, path.name)
print('RELATION_AUDIT_COMPLETE', archive.stat().st_size,
      hashlib.sha256(archive.read_bytes()).hexdigest(), flush=True)
