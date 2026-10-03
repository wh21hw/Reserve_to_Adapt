"""Farthest birth real-feature audit with fixed gate and source calibration."""
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import zipfile
import numpy as np
import torch

path=Path('/content/source_anchored_imp_farthest_v1.py')
spec=importlib.util.spec_from_file_location('farthest',path)
module=importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
data_path=Path('/content/imp-runs/rta-space-warmup-l4-features-v1/features.npz')
data=np.load(data_path,allow_pickle=False)
assert 'target_labels' not in data.files
source,target=torch.from_numpy(data['source']),torch.from_numpy(data['target'])
labels=torch.from_numpy(data['source_labels'])
centers=torch.stack([source[labels==c].mean(0) for c in range(10)])
residual=source-centers[labels]
threshold=max(float(np.quantile(residual.square().sum(1).numpy(),.99)),1e-8)
variance=max(float(residual.square().mean()),1e-8)
output=Path('/content/imp-runs/farthest-constrained-frozen-v1')
if output.exists():
    raise RuntimeError('Refusing overwrite')
output.mkdir(parents=True)
torch.set_num_threads(2)
report=dict(feature_sha256=hashlib.sha256(data_path.read_bytes()).hexdigest(),
    module_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),threshold=threshold,
    variance=variance,prior_strength=5,birth_unknown_min=.5,steps=5,capacity=100,
    interpretation='Deterministic geometric candidates; no semantic unknown count',
    subset_caveat='Mixture compatibility inherited from full-target fit, not refitted on subsets',trials=[])
orders=[('original',np.arange(len(target))),('reverse',np.arange(len(target)-1,-1,-1)),
        ('shuffle1',np.random.default_rng(1).permutation(len(target)))]
for seed in range(1,4):
    orders.append(('subset80-seed%d'%seed,np.random.default_rng(seed).permutation(len(target))[:int(.8*len(target))]))
gates=[('ungated',None)]
for seed in [1,2,3]:
    gate_path=Path('/content/imp-runs/relation-constrained-frozen-v1')/('gate-seed%d-original.npz'%seed)
    compatibility=torch.from_numpy(np.load(gate_path,allow_pickle=False)['known_compatibility'])
    gates.append(('gate-seed%d'%seed,compatibility))
for gate_name,compatibility in gates:
    original=None
    for order_name,order in orders:
        name=gate_name+'-'+order_name
        model=module.SourceAnchoredIMP(threshold,variance,prior_strength=5,steps=5,max_prototypes=100)
        gate=compatibility[order.copy()] if compatibility is not None else None
        result=model.fit(target[order.copy()],centers,gate,birth_strategy='farthest')
        prototypes=torch.cat([result['known_prototypes'],result['candidate_prototypes']])
        responsibilities=model._assign(target,prototypes,10,compatibility)
        if original is None:
            original=responsibilities.clone()
        equal=bool(torch.equal(original,responsibilities))
        if len(order)==len(target):
            assert equal, 'Permutation changed fixed farthest result'
        row=dict(name=name,fitting_samples=len(order),candidate_count=result['candidate_count'],
            candidate_mass_at_least_5=int((responsibilities.sum(0)[10:]>=5).sum()),
            mean_candidate_probability=float(responsibilities[:,10:].sum(1).mean()),
            exactly_equal_to_original=equal,all_finite=bool(torch.isfinite(responsibilities).all()),
            rows_sum_to_one=bool(torch.allclose(responsibilities.sum(1),torch.ones(len(target)),atol=1e-5)))
        assert row['all_finite'] and row['rows_sum_to_one']
        np.savez_compressed(output/(name+'.npz'),responsibilities=responsibilities.numpy(),
            known=result['known_prototypes'].numpy(),candidates=result['candidate_prototypes'].numpy(),
            fit_indices=order)
        report['trials'].append(row)
        (output/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False))
        print('FARTHEST_TRIAL',json.dumps(row),flush=True)
shutil.copy2(path,output/path.name)
archive=Path('/content/farthest-constrained-frozen-v1.zip')
with zipfile.ZipFile(archive,'x',zipfile.ZIP_DEFLATED) as bundle:
    for item in output.iterdir():
        bundle.write(item,item.name)
    bundle.write('/content/farthest-birth-unit-v1.json','unit-test.json')
print('FARTHEST_FROZEN_COMPLETE',archive.stat().st_size,
      hashlib.sha256(archive.read_bytes()).hexdigest(),flush=True)
