"""One frozen-feature K estimate: source anchors, no relation gate or test labels.

DP-means-inspired birth rule plus soft Gaussian updates; source distance
quantile calibrates the threshold. This is not the full learned-variance IMP.
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch
sys.path.insert(0, '/content')
from source_anchored_imp import SourceAnchoredIMP


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source-run', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--known-centers-fixed', action='store_true')
    parser.add_argument('--source-quantile', type=float, default=.99)
    parser.add_argument('--prior-strength', type=float, default=5.)
    parser.add_argument('--steps', type=int, default=5)
    args = parser.parse_args()
    root, output = Path(args.source_run), Path(args.output)
    if output.exists():
        raise RuntimeError('Do not overwrite a previous estimate')
    if not 0 < args.source_quantile < 1:
        raise ValueError('source quantile must be between 0 and 1')
    config = json.loads((root/'config.json').read_text())
    data = np.load(root/'features.npz', allow_pickle=False)
    if set(data.files) != {'source', 'target', 'source_labels', 'source_centers'}:
        raise ValueError('Use the C-output source prior, without target labels')
    source, target = torch.from_numpy(data['source']), torch.from_numpy(data['target'])
    labels, centers = torch.from_numpy(data['source_labels']), torch.from_numpy(data['source_centers'])
    if not all(np.isfinite(data[key]).all() for key in data.files):
        raise ValueError('Nonfinite input')
    residual = source - centers[labels]
    threshold = max(float(np.quantile(residual.square().sum(1).numpy(), args.source_quantile)), 1e-8)
    variance = max(float(residual.square().mean()), 1e-8)
    torch.set_num_threads(2)
    model = SourceAnchoredIMP(threshold, variance, prior_strength=args.prior_strength,
        steps=args.steps, max_prototypes=100, known_centers_fixed=args.known_centers_fixed)
    result = model.fit(target, centers, known_compatibility=None, birth_strategy='farthest')
    K = result['candidate_count']
    report = dict(seed=config['seed'], known_classes=len(centers), K=K,
        classifier_dimensions=len(centers)+K, source_run=str(root),
        threshold=threshold, source_quantile=args.source_quantile, variance=variance,
        prior_strength=args.prior_strength, steps=args.steps, known_centers_fixed=args.known_centers_fixed,
        history=result['history'], effective_counts=result['effective_counts'].tolist(),
        target_labels_used=False, optimizer_updates=0, relation_gate=False,
        small_cluster_filter=False, downstream_change='Only integer K passed to original RTA',
        interpretation='New components treated as unknown classes under the assumed model, not semantic discovery proof',
        implementation='Source-calibrated DP-means-inspired birth and soft updates; not full IMP variance learning',
        can_start_rta=K > 0)
    output.mkdir(parents=True)
    np.savez_compressed(output/'prototypes.npz', known=result['known_prototypes'].numpy(),
                        unknown=result['candidate_prototypes'].numpy(),
                        responsibilities=result['responsibilities'].numpy())
    (output/'estimate.json').write_text(json.dumps(report, indent=2, allow_nan=False))
    print('K_ESTIMATE_COMPLETE', json.dumps(report, allow_nan=False), flush=True)
    if K == 0:
        print('No unknown component inferred; do not silently force K=1', flush=True)


if __name__ == '__main__':
    main()
