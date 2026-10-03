"""CPU-only, label-free candidate audit on new fixed warm features; no SGD."""
import hashlib
import importlib.util
import json
from pathlib import Path
import warnings
import zipfile
import numpy as np
import torch
from sklearn.mixture import BayesianGaussianMixture

def load(name, filename, digest):
    path = Path('/content') / filename
    assert hashlib.sha256(path.read_bytes()).hexdigest() == digest
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

imp = load('matched_imp', 'matched_source_anchored_imp.py',
           'ec38a017ae2b91755499e136d309fc2af10eff802aa7a114b12fa2cd51943a4f')
relation = load('matched_relation', 'matched_relation_gate.py',
                '1556c60c5536b508e7f631fdd0dcac434a43749d99e08850eba2292bb26de3da')
feature_path = Path('/content/imp-runs/multitask-a2w-seed1-warm-features-v1/features.npz')
assert hashlib.sha256(feature_path.read_bytes()).hexdigest() == 'a9d1a40aeb863b72aafefaed965d22d8eb3b3099516aee117f1897bdb4fb5459'
data = np.load(feature_path, allow_pickle=False)
assert set(data.files) == {'source', 'target', 'source_labels', 'source_logits', 'target_logits'}
source, target = torch.from_numpy(data['source']), torch.from_numpy(data['target'])
labels = torch.from_numpy(data['source_labels'])
assert torch.equal(torch.unique(labels), torch.arange(10))
centers = torch.stack([source[labels == c].mean(0) for c in range(10)])
residual = source - centers[labels]
threshold = max(float(np.quantile(residual.square().sum(1).numpy(), .99)), 1e-8)
variance = max(float(residual.square().mean()), 1e-8)
scores = relation.source_relation_scores(torch.from_numpy(data['source_logits']), labels,
                                         torch.from_numpy(data['target_logits']), 10)['scores'].numpy()
torch.set_num_threads(2)
mixture = BayesianGaussianMixture(n_components=4, max_iter=800, random_state=1)
with warnings.catch_warnings(record=True) as messages:
    warnings.simplefilter('always')
    mixture.fit(scores[:, None])
known = int(mixture.means_.argmin())
compatibility = torch.from_numpy(mixture.predict_proba(scores[:, None])[:, known]).to(target.dtype)
output = Path('/content/imp-runs/matched-warm-candidates-v1')
assert not output.exists(), 'Refusing overwrite'
output.mkdir(parents=True)
report = dict(feature_sha256=hashlib.sha256(feature_path.read_bytes()).hexdigest(),
              threshold=threshold, variance=variance, prior_strength=5, steps=5,
              capacity=100, birth_unknown_min=.5, support_min=5, gate_seed=1,
              gate_converged=bool(mixture.converged_), gate_means=mixture.means_.ravel().tolist(),
              warnings=[str(message.message) for message in messages],
              semantic_unknown_count=None, optimizer_steps=0,
              caveat='In-sample source calibration; full-target gate inherited for subsets; not a DP posterior or semantic count',
              trials=[])
orders = [('original', np.arange(len(target))),
          ('reverse', np.arange(len(target) - 1, -1, -1)),
          ('shuffle1', np.random.default_rng(1).permutation(len(target)))]
orders += [(f'subset80-seed{s}', np.random.default_rng(s).permutation(len(target))[:int(.8 * len(target))])
           for s in (1, 2, 3)]
original = None
for name, order in orders:
    model = imp.SourceAnchoredIMP(threshold, variance, prior_strength=5, steps=5, max_prototypes=100)
    result = model.fit(target[order.copy()], centers, compatibility[order.copy()], birth_strategy='farthest')
    prototypes = torch.cat([result['known_prototypes'], result['candidate_prototypes']])
    responsibility = model._assign(target, prototypes, 10, compatibility)
    mass = responsibility.sum(0)[10:]
    equal = original is None or torch.equal(original, responsibility)
    if original is None:
        original = responsibility.clone()
    row = dict(name=name, fitting_samples=len(order), candidate_count=result['candidate_count'],
               supported_components=int((mass >= 5).sum()), masses=mass.tolist(),
               permutation_equal=equal, finite=bool(torch.isfinite(responsibility).all()),
               normalized=bool(torch.allclose(responsibility.sum(1), torch.ones(len(target)), atol=1e-5)))
    report['trials'].append(row)
    with (output / 'report.json').open('w') as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
    assert row['finite'] and row['normalized']
    if len(order) == len(target):
        assert equal, 'Permutation changed candidate inference'
    np.savez_compressed(output / f'{name}.npz', candidates=result['candidate_prototypes'].numpy(),
                        known=result['known_prototypes'].numpy(), responsibilities=responsibility.numpy(),
                        known_compatibility=compatibility.numpy(), fit_indices=order)
    print('MATCHED_CANDIDATES', json.dumps(row), flush=True)
archive_path = Path('/content/matched-warm-candidates-v1.zip')
with zipfile.ZipFile(archive_path, 'x', zipfile.ZIP_DEFLATED) as archive:
    for path in output.iterdir():
        archive.write(path, path.name)
    for path in (Path(__file__), Path('/content/matched_source_anchored_imp.py'), Path('/content/matched_relation_gate.py')):
        archive.write(path, 'code/' + path.name)
print('MATCHED_CANDIDATES_PASS', archive_path.stat().st_size,
      hashlib.sha256(archive_path.read_bytes()).hexdigest(), flush=True)
