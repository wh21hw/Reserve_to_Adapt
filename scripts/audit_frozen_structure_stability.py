"""Fixed-rule order/subsampling audit. No target evaluation labels are read."""
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import zipfile

import numpy as np
import torch

root = Path('/content/imp-runs')
feature_path = root/'source-warmup-l4-seed1-v2/features.npz'
module_path = Path('/content/source_anchored_imp.py')
output = root/'source-anchored-stability-v1'
if output.exists():
    raise RuntimeError('Refusing overwrite')
data = np.load(feature_path, allow_pickle=False)
assert 'target_labels' not in data.files
source = torch.from_numpy(data['source']).float()
target = torch.from_numpy(data['target']).float()
labels = torch.from_numpy(data['source_labels'])
assert torch.equal(torch.unique(labels), torch.arange(10))
centers = torch.stack([source[labels == i].mean(0) for i in range(10)])
residual = source-centers[labels]
threshold = max(float(np.quantile(residual.square().sum(1).numpy(), .99)), 1e-8)
variance = max(float(residual.square().mean()), 1e-8)
spec = importlib.util.spec_from_file_location('prototype_module', module_path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
torch.set_num_threads(2)
output.mkdir(parents=True)
report = dict(feature_sha256=hashlib.sha256(feature_path.read_bytes()).hexdigest(),
              module_sha256=hashlib.sha256(module_path.read_bytes()).hexdigest(),
              threshold=threshold, observation_variance=variance,
              prior_strength=5, steps=5, max_prototypes=100,
              candidate_semantics='Not identified by this audit', trials=[])
orders = [('original', np.arange(len(target))),
          ('reverse', np.arange(len(target)-1, -1, -1))]
for seed in range(1, 6):
    order = np.random.default_rng(seed).permutation(len(target))
    orders.extend([('shuffle%d' % seed, order),
                   ('subset80-seed%d' % seed, order[:int(.8*len(target))])])
baseline_probability = None
for name, order in orders:
    row = dict(name=name, fitting_samples=len(order))
    model = module.SourceAnchoredIMP(threshold, variance, prior_strength=5,
                                    steps=5, max_prototypes=100)
    result = model.fit(target[order.copy()], centers)
    # Every trial predicts the same complete target set for comparable diagnostics.
    prototypes = torch.cat([result['known_prototypes'], result['candidate_prototypes']])
    responsibilities = model._assign(target, prototypes)
    probability = responsibilities[:, 10:].sum(1).numpy()
    candidate_flag = responsibilities.argmax(1).numpy() >= 10
    if baseline_probability is None:
        baseline_probability = probability.copy()
        baseline_flag = candidate_flag.copy()
    union = (baseline_flag | candidate_flag).sum()
    row.update(candidate_count=result['candidate_count'],
               fit_candidate_mass=result['effective_counts'][10:].tolist(),
               full_target_candidate_fraction=float(candidate_flag.mean()),
               probability_mae_vs_original=float(np.abs(probability-baseline_probability).mean()),
               candidate_flag_jaccard_vs_original=float((baseline_flag & candidate_flag).sum()/union) if union else 1.,
               all_finite=bool(torch.isfinite(responsibilities).all()),
               rows_sum_to_one=bool(torch.allclose(responsibilities.sum(1), torch.ones(len(target)), atol=1e-5)))
    assert row['all_finite'] and row['rows_sum_to_one']
    np.savez_compressed(output/(name+'.npz'), responsibilities=responsibilities.numpy(),
                        known=result['known_prototypes'].numpy(),
                        candidates=result['candidate_prototypes'].numpy(), fit_indices=order)
    report['trials'].append(row)
    (output/'report.json').write_text(json.dumps(report, indent=2, allow_nan=False))
    print('STABILITY_TRIAL', json.dumps(row), flush=True)
shutil.copy2(module_path, output/module_path.name)
archive = Path('/content/source-anchored-stability-v1.zip')
with zipfile.ZipFile(archive, 'x', zipfile.ZIP_DEFLATED) as bundle:
    for path in output.iterdir():
        bundle.write(path, path.name)
print('STABILITY_COMPLETE', archive.stat().st_size,
      hashlib.sha256(archive.read_bytes()).hexdigest(), flush=True)
