"""Predeclared relation-prior comparison on fixed RTA-space features; label-free."""
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import warnings
import zipfile
import numpy as np
from sklearn.mixture import BayesianGaussianMixture
import torch

def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

module_path = Path('/content/source_anchored_imp_relation_v1.py')
module = load('constrained', module_path)
relation = load('relation', '/content/relation_gate.py')
features_path = Path('/content/imp-runs/rta-space-warmup-l4-features-v1/features.npz')
output = Path('/content/imp-runs/relation-constrained-frozen-v1')
if output.exists():
    raise RuntimeError('Refusing overwrite')
data = np.load(features_path, allow_pickle=False)
assert 'target_labels' not in data.files
source, target = torch.from_numpy(data['source']), torch.from_numpy(data['target'])
labels = torch.from_numpy(data['source_labels'])
centers = torch.stack([source[labels == c].mean(0) for c in range(10)])
residual = source-centers[labels]
threshold = max(float(np.quantile(residual.square().sum(1).numpy(), .99)), 1e-8)
variance = max(float(residual.square().mean()), 1e-8)
score = relation.source_relation_scores(torch.from_numpy(data['source_logits']), labels,
    torch.from_numpy(data['target_logits']), 10)['scores'].numpy()
torch.set_num_threads(2)
output.mkdir(parents=True)
report = dict(feature_sha256=hashlib.sha256(features_path.read_bytes()).hexdigest(),
    module_sha256=hashlib.sha256(module_path.read_bytes()).hexdigest(),
    threshold=threshold, observation_variance=variance, prior_strength=5, birth_unknown_min=.5,
    meaning='Candidate structures, not semantic class count; gate is plug-in compatibility', mixtures=[], trials=[])
orders = [('original', np.arange(len(target))), ('reverse', np.arange(len(target)-1,-1,-1))]
gates = [('ungated', None)]
for seed in [1,2,3]:
    mixture = BayesianGaussianMixture(n_components=4, max_iter=800, random_state=seed)
    with warnings.catch_warnings(record=True) as messages:
        warnings.simplefilter('always')
        mixture.fit(score[:,None])
    known = int(mixture.means_.argmin())
    compatibility = torch.from_numpy(mixture.predict_proba(score[:,None])[:,known]).to(target.dtype)
    gates.append(('gate-seed%d' % seed, compatibility))
    row = dict(seed=seed, converged=bool(mixture.converged_), means=mixture.means_.ravel().tolist(),
        weights=mixture.weights_.tolist(), known_component=known,
        birth_eligible_samples=int((1-compatibility >= .5).sum()),
        warnings=[str(message.message) for message in messages])
    report['mixtures'].append(row)
    print('GATE_MIXTURE', json.dumps(row), flush=True)
for gate_name, compatibility in gates:
    for order_name, order in orders:
        name = gate_name+'-'+order_name
        row = dict(name=name)
        try:
            model = module.SourceAnchoredIMP(threshold, variance, prior_strength=5, steps=5, max_prototypes=100)
            gate = compatibility[order.copy()] if compatibility is not None else None
            result = model.fit(target[order.copy()], centers, gate)
            prototypes = torch.cat([result['known_prototypes'],result['candidate_prototypes']])
            responsibilities = model._assign(target, prototypes, 10, compatibility)
            row.update(candidate_count=result['candidate_count'],
                candidate_mass=responsibilities.sum(0)[10:].tolist(),
                candidate_mass_at_least_5=int((responsibilities.sum(0)[10:] >= 5).sum()),
                mean_candidate_probability=float(responsibilities[:,10:].sum(1).mean()),
                hard_candidate_samples=int((responsibilities.argmax(1) >= 10).sum()),
                known_center_shift=(result['known_prototypes']-centers).norm(dim=1).tolist(),
                all_finite=bool(torch.isfinite(responsibilities).all()),
                rows_sum_to_one=bool(torch.allclose(responsibilities.sum(1),torch.ones(len(target)),atol=1e-5)))
            assert row['all_finite'] and row['rows_sum_to_one']
            np.savez_compressed(output/(name+'.npz'), responsibilities=responsibilities.numpy(),
                known=result['known_prototypes'].numpy(), candidates=result['candidate_prototypes'].numpy(),
                known_compatibility=compatibility.numpy() if compatibility is not None else np.array([]))
        except RuntimeError as error:
            row.update(candidate_count=None, error=str(error))
        report['trials'].append(row)
        (output/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False))
        print('CONSTRAINED_TRIAL',json.dumps(row),flush=True)
shutil.copy2(module_path,output/module_path.name)
shutil.copy2('/content/relation_gate.py',output/'relation_gate.py')
archive = Path('/content/relation-constrained-frozen-v1.zip')
with zipfile.ZipFile(archive,'x',zipfile.ZIP_DEFLATED) as bundle:
    for path in output.iterdir():
        bundle.write(path,path.name)
    bundle.write('/content/relation-constrained-unit-v1.json','unit-test.json')
print('CONSTRAINED_FROZEN_COMPLETE',archive.stat().st_size,
      hashlib.sha256(archive.read_bytes()).hexdigest(),flush=True)
