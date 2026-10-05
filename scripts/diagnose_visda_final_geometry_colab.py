"""Post-hoc semantic diagnostics of two cached representations, no model forward.

Real target labels are used ONLY here to describe failure mechanisms. This
script never fits K, selects thresholds, modifies models, or emits train labels.
"""
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score


ROOT = Path('/content/imp-runs/visda-frozenbn-capacity-10e-v1')
DATA = Path('/content/osda-visda-syn2real-v1')
KNOWN = [1, 2, 3, 6, 10, 11]


def read_labels(path):
    return np.array([int(row.rsplit(None, 1)[1]) for row in path.read_text().splitlines()
                     if row.strip()], dtype=np.int64)


def distances(rows, centers):
    return np.maximum((rows * rows).sum(1)[:, None]
                      + (centers * centers).sum(1)[None, :] - 2 * rows.dot(centers.T), 0)


def stage_report(cache, source_labels, target_labels):
    source = cache['source'].astype(np.float64)
    target = cache['target'].astype(np.float64)
    if source.shape != (len(source_labels), 256) or target.shape != (len(target_labels), 256):
        raise ValueError('Feature row/dimension mismatch')
    if not np.isfinite(source).all() or not np.isfinite(target).all():
        raise ValueError('Nonfinite cached features')
    centers = np.stack([source[source_labels == c].mean(0) for c in range(6)])
    known_mask = np.isin(target_labels, KNOWN)
    target_mapped = np.full(len(target_labels), -1, dtype=np.int64)
    for c, raw in enumerate(KNOWN):
        target_mapped[target_labels == raw] = c
    nearest, radius = [], []
    for start in range(0, len(target), 2048):
        d = distances(target[start:start + 2048], centers)
        nearest.append(d.argmin(1))
        radius.append(d.min(1))
    nearest, radius = np.concatenate(nearest), np.concatenate(radius)
    pair_dist = distances(centers, centers)
    np.fill_diagonal(pair_dist, np.inf)
    per_class = []
    for c, raw in enumerate(KNOWN):
        s, t = source[source_labels == c], target[target_labels == raw]
        tm = t.mean(0)
        per_class.append(dict(raw_class=raw, source_count=len(s), target_count=len(t),
            source_within_mean_squared=float(((s - centers[c]) ** 2).sum(1).mean()),
            target_within_mean_squared=float(((t - tm) ** 2).sum(1).mean()),
            cross_domain_center_squared=float(((tm - centers[c]) ** 2).sum()),
            nearest_other_source_center_squared=float(pair_dist[c].min()),
            nearest_source_identity_accuracy=float((nearest[target_labels == raw] == c).mean())))
    report = dict(per_known_class=per_class,
        nearest_source_identity_macro=float(np.mean([r['nearest_source_identity_accuracy'] for r in per_class])),
        nearest_source_identity_micro=float((nearest[known_mask] == target_mapped[known_mask]).mean()),
        distance_unknown_AUROC=float(roc_auc_score(~known_mask, radius)),
        nearest_source_squared_distance_median=dict(
            known=float(np.median(radius[known_mask])), unknown=float(np.median(radius[~known_mask]))))
    if 'target_logits' in cache.files:
        logits = cache['target_logits']
        if logits.shape != (len(target), 8) or not np.isfinite(logits).all():
            raise ValueError('Unexpected final logits')
        predicted = logits.argmax(1)
        known_only = logits[:, :6].argmax(1)
        margin = logits[:, 6:].max(1) - logits[:, :6].max(1)
        report['head_diagnostic'] = dict(
            known_predicted_unknown_fraction=float((predicted[known_mask] >= 6).mean()),
            unknown_predicted_known_fraction=float((predicted[~known_mask] < 6).mean()),
            known_only_identity_macro=float(np.mean([
                (known_only[target_labels == raw] == c).mean() for c, raw in enumerate(KNOWN)])),
            unknown_minus_known_logit_margin_median=dict(
                known=float(np.median(margin[known_mask])), unknown=float(np.median(margin[~known_mask]))),
            predicted_slot_counts=np.bincount(predicted, minlength=8).tolist())
    return report


def main():
    folder = ROOT / 'fixed2-final10-geometry-v1'
    output = folder / 'posthoc-geometry.json'
    if output.exists():
        raise FileExistsError('Preserve existing diagnostic')
    manifest = json.loads((folder / 'manifest.json').read_text())
    if not manifest['complete'] or manifest['epoch'] != 10 or manifest['target_labels_used']:
        raise ValueError('Require predeclared completed label-free cache')
    raw_source = read_labels(DATA / 'source-known-6.txt')
    if sorted(set(raw_source.tolist())) != KNOWN:
        raise ValueError('Known class identity mismatch')
    source_labels = np.array([KNOWN.index(int(y)) for y in raw_source], dtype=np.int64)
    target_labels = read_labels(DATA / 'target-real-12.txt')
    stages = {}
    for name, path in [('source3', ROOT / 'source/features.npz'), ('fixed2_final10', folder / 'features.npz')]:
        with np.load(str(path), allow_pickle=False) as cache:
            if 'source_labels' in cache.files and not np.array_equal(cache['source_labels'], source_labels):
                raise ValueError('Source-stage cache ordering mismatch')
            stages[name] = stage_report(cache, source_labels, target_labels)
    report = dict(stages=stages, selection='Fixed final10, not target-best checkpoint',
        target_labels_used_for_posthoc_diagnostics=True, target_labels_used_for_training_or_structure=False,
        caveat='Geometry describes association, not loss-module causality. AUROC is distance ranking only, not trained rejection or threshold selection.')
    output.write_text(json.dumps(report, indent=2, allow_nan=False))
    print('FINAL_GEOMETRY_POSTHOC', json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
