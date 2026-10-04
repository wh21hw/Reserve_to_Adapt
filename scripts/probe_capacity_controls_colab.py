"""Matched-size and missing-anchor geometry controls, without target labels."""
import json
from pathlib import Path
import numpy as np
import torch
from robust_capacity import fit_robust_capacity
from source_anchored_imp import SourceAnchoredIMP

torch.set_num_threads(2)
reports = []

def infer(x, anchors, penalty, variance, evaluation_labels=None, hidden_class=None):
    c = len(anchors)
    robust = fit_robust_capacity(x, anchors, penalty)
    imp = SourceAnchoredIMP(penalty, variance, prior_strength=5., steps=5).fit(
        torch.from_numpy(x), torch.from_numpy(anchors), birth_strategy='farthest')
    output = {}
    for name, result, ids, k in (
            ('robust', robust, robust['assignments'], robust['K']),
            ('imp', imp, imp['responsibilities'].argmax(1).numpy(), imp['candidate_count'])):
        row = dict(K=k, candidate_assignment_fraction=float((ids >= c).mean()))
        if name == 'robust':
            row.update(noise_count=result['noise_count'], converged=result['converged'])
        if hidden_class is not None:
            hidden = evaluation_labels == hidden_class
            row.update(hidden_class_rows=int(hidden.sum()),
                       hidden_candidate_recall=float((ids[hidden] >= c).mean()),
                       retained_known_false_candidate=float((ids[~hidden] >= c).mean()))
        output[name] = row
    return output

for seed in (1, 2, 3):
    file = Path('/content/imp-runs/fusion-imp-rta-v1')/('seed%d/source/features.npz' % seed)
    with np.load(file) as data:
        source, labels = data['source'].astype(np.float64), data['source_labels']
        target = data['target'].astype(np.float64)
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
    indices = np.random.RandomState(2027).permutation(len(target))[:len(probe)]
    stages = dict(
        target_matched_291=infer(target[indices], anchors, penalty, variance),
        target_full_564=infer(target, anchors, penalty, variance))
    # Class9 chosen in advance, not for easiest recovery. Source-trained embedding
    # has already seen it; this tests anchor removal, not unseen-class generalization.
    retained = train[labels[train] != 9]
    residual9 = source[retained]-anchors[labels[retained]]
    penalty9 = max(float(np.quantile((residual9**2).sum(1), .99)), 1e-8)
    variance9 = max(float((residual9**2).mean()), 1e-8)
    stages['source_missing_anchor9'] = infer(source[probe], anchors[:9], penalty9,
                                            variance9, labels[probe], 9)
    report = dict(seed=seed, penalty=penalty, missing_anchor_penalty=penalty9,
                  stages=stages, target_labels_used=False,
                  source_centers_scale_rows=len(train), source_probe_rows=len(probe),
                  caveat='Full-source trained features; one deterministic subsample; geometry controls not proof of semantic recovery')
    reports.append(report)
    print(json.dumps(report), flush=True)
destination = Path('/content/capacity-geometry-controls-v1.json')
destination.write_text(json.dumps(reports, indent=2, allow_nan=False))
print('CAPACITY_GEOMETRY_RESULTS', destination, flush=True)
