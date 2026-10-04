"""Known-only source negative control; no target features or labels are used."""
import json
from pathlib import Path
import numpy as np
import torch
from robust_capacity import fit_robust_capacity
from source_anchored_imp import SourceAnchoredIMP

torch.set_num_threads(2)
reports = []
for seed in (1, 2, 3):
    file = Path('/content/imp-runs/fusion-imp-rta-v1')/('seed%d/source/features.npz' % seed)
    with np.load(file) as data:
        source = data['source'].astype(np.float64)
        labels = data['source_labels']
    rng = np.random.RandomState(2026)
    train, probe = [], []
    for c in range(10):
        indices = np.flatnonzero(labels == c)
        rng.shuffle(indices)
        cut = max(1, int(.7*len(indices)))
        train.extend(indices[:cut])
        probe.extend(indices[cut:])
    train, probe = np.asarray(train), np.asarray(probe)
    anchors = np.stack([source[train][labels[train] == c].mean(0) for c in range(10)])
    residual = source[train]-anchors[labels[train]]
    penalty = max(float(np.quantile((residual**2).sum(1), .99)), 1e-8)
    variance = max(float((residual**2).mean()), 1e-8)
    x = source[probe]
    robust = fit_robust_capacity(x, anchors, penalty)
    imp = SourceAnchoredIMP(penalty, variance, prior_strength=5., steps=5).fit(
        torch.from_numpy(x), torch.from_numpy(anchors), birth_strategy='farthest')
    imp_ids = imp['responsibilities'].argmax(1).numpy()
    robust_ids = robust['assignments']
    report = dict(seed=seed, source_train_rows=len(train), source_probe_rows=len(probe),
                  true_additional_classes=0, penalty=penalty,
                  robust=dict(K=robust['K'], counts=robust['counts'].tolist(),
                              noise_count=robust['noise_count'], converged=robust['converged'],
                              candidate_assignment_fraction=float((robust_ids >= 10).mean()),
                              known_identity_accuracy=float((robust_ids == labels[probe]).mean())),
                  imp=dict(K=imp['candidate_count'],
                           candidate_effective_counts=imp['effective_counts'][10:].tolist(),
                           candidate_assignment_fraction=float((imp_ids >= 10).mean()),
                           known_identity_accuracy=float((imp_ids == labels[probe]).mean())),
                  target_features_used=False, target_labels_used=False,
                  caveat='feature extractor was trained on full source; only centers/scales use 70% split; not independent generalization test')
    reports.append(report)
    print(json.dumps(report), flush=True)
destination = Path('/content/source-negative-capacity-v1.json')
destination.write_text(json.dumps(reports, indent=2, allow_nan=False))
print('SOURCE_NEGATIVE_RESULTS', destination, flush=True)
