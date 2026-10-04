"""Inference-only probe: source labels allowed; no target labels are loaded."""
import json
from pathlib import Path
import numpy as np
from robust_capacity import fit_robust_capacity

# One targeted check: isolated novel point stays noise; a coherent group can birth.
toy = fit_robust_capacity(np.array([[0.], [.1], [10.], [10.1], [100.]]),
                         np.array([[0.]]), penalty=1.)
assert toy['K'] == 1 and toy['assignments'][-1] == -1
assert all(b['objective'] <= a['objective']+1e-8
           for a, b in zip(toy['history'], toy['history'][1:]))

reports = []
for seed in (1, 2, 3):
    path = Path('/content/imp-runs/fusion-imp-rta-v1')/('seed%d/source/features.npz' % seed)
    with np.load(path) as data:
        source, labels, target = data['source'], data['source_labels'], data['target']
    anchors = np.stack([source[labels == j].mean(0) for j in range(10)])
    residuals = ((source-anchors[labels])**2).sum(1)
    penalty = max(float(np.quantile(residuals, .99)), 1e-8)
    result = fit_robust_capacity(target, anchors, penalty, prior_strength=5.)
    report = dict(seed=seed, target_rows=len(target), K=result['K'],
                  counts=result['counts'].tolist(), noise_count=result['noise_count'],
                  converged=result['converged'], history=result['history'],
                  penalty=penalty, prior_strength=5., target_labels_used=False,
                  semantic_unknown_count=None, stage='inference only; no RTA training')
    reports.append(report)
    print(json.dumps(report), flush=True)
destination = Path('/content/robust-capacity-inference-v1.json')
destination.write_text(json.dumps(reports, indent=2, allow_nan=False))
print('ROBUST_CAPACITY_PROBE', destination, flush=True)
