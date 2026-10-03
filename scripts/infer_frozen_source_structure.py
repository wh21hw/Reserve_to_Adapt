"""Source-calibrated prototype diagnostics, with no target-label file access.

Declared sensitivity grid, not target-test tuning. Candidate count is not a
semantic unknown class estimate. Inputs come only from a source-only checkpoint.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import time

import numpy as np
import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--features', required=True)
    parser.add_argument('--module', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise RuntimeError('Refusing overwrite')
    feature_path, module_path = Path(args.features), Path(args.module)
    data = np.load(feature_path, allow_pickle=False)
    if 'target_labels' in data.files:
        raise ValueError('Target evaluation labels must not enter structure inference')
    source = torch.from_numpy(data['source']).float()
    target = torch.from_numpy(data['target']).float()
    source_labels = torch.from_numpy(data['source_labels'])
    classes = torch.unique(source_labels, sorted=True)
    if not torch.equal(classes, torch.arange(10)):
        raise ValueError('Expected known source labels 0-9')
    centers = torch.stack([source[source_labels == label].mean(dim=0) for label in classes])
    residual = source - centers[source_labels]
    squared_radius = residual.square().sum(dim=1)
    threshold = max(float(np.quantile(squared_radius.numpy(), .99)), 1e-8)
    variance = max(float(residual.square().mean()), 1e-8)
    spec = importlib.util.spec_from_file_location('prototype_model', str(module_path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    torch.set_num_threads(2)
    output.mkdir(parents=True)
    report = dict(stage='source-calibrated frozen structure diagnostics',
                  feature_sha256=hashlib.sha256(feature_path.read_bytes()).hexdigest(),
                  module_sha256=hashlib.sha256(module_path.read_bytes()).hexdigest(),
                  source_samples=len(source), target_samples=len(target),
                  calibration=dict(source_radius_quantile=.99, threshold=threshold,
                                   isotropic_observation_variance=variance,
                                   caveat='In-sample source calibration, not held-out calibration'),
                  meaning='Candidate components, not automatically unknown semantic classes',
                  trials=[])
    for factor in [.5, 1., 2.]:
        for strength in [0., 5., 20.]:
            name = 'factor%s-prior%s' % (factor, strength)
            start = time.time()
            row = dict(name=name, threshold_factor=factor, prior_strength=strength,
                       max_prototypes=100, steps=5)
            try:
                model = module.SourceAnchoredIMP(threshold=threshold*factor,
                    observation_variance=variance, prior_strength=strength,
                    max_prototypes=100, steps=5)
                result = model.fit(target, centers)
                responsibilities = result['responsibilities']
                candidate_probability = responsibilities[:, 10:].sum(dim=1)
                candidate_mass = result['effective_counts'][10:]
                row.update(candidate_count=result['candidate_count'],
                    candidate_mass=candidate_mass.tolist(),
                    candidate_mass_at_least_5=int((candidate_mass >= 5).sum()),
                    mean_candidate_probability=float(candidate_probability.mean()),
                    rows_sum_to_one=bool(torch.allclose(responsibilities.sum(dim=1),
                                                       torch.ones(len(target)), atol=1e-5)),
                    known_center_shift=(result['known_prototypes']-centers).norm(dim=1).tolist(),
                    history=result['history'])
                np.savez_compressed(output/(name+'.npz'),
                    known=result['known_prototypes'].numpy(),
                    candidates=result['candidate_prototypes'].numpy(),
                    responsibilities=responsibilities.numpy(),
                    effective_counts=result['effective_counts'].numpy())
            except RuntimeError as error:
                row['error'] = str(error)
                row['candidate_count'] = None
            row['seconds'] = time.time()-start
            report['trials'].append(row)
            (output/'report.json').write_text(json.dumps(report, indent=2, allow_nan=False))
            print('STRUCTURE_TRIAL', json.dumps(row), flush=True)
    print('FROZEN_STRUCTURE_DIAGNOSTICS_COMPLETE', str(output), flush=True)


if __name__ == '__main__':
    main()
