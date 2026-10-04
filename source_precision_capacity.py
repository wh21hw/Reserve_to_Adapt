"""Frozen v1 K-estimation protocol; only the returned count enters K-only RTA.

Source labels calibrate anchors, scales and empirical prior precision. No target
labels accepted. K is modeling capacity, not a certified semantic class count.
"""
import numpy as np
from robust_capacity import fit_robust_capacity


def estimate_capacity(source, source_labels, target):
    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    labels = np.asarray(source_labels)
    if source.ndim != 2 or labels.shape != (len(source),) or labels.dtype.kind not in 'iu':
        raise ValueError('Expected source feature matrix and integer label vector')
    classes = np.unique(labels)
    if not np.array_equal(classes, np.arange(len(classes))) or len(classes) == 0:
        raise ValueError('Source classes must be contiguous 0..C-1')
    rng = np.random.RandomState(2026)
    calibration, validation = [], []
    for c in classes:
        indices = np.flatnonzero(labels == c)
        if len(indices) < 2:
            raise ValueError('At least two source examples required per class')
        rng.shuffle(indices)
        cut = max(1, int(.7*len(indices)))
        calibration.extend(indices[:cut])
        validation.extend(indices[cut:])
    calibration, validation = np.asarray(calibration), np.asarray(validation)
    anchors = np.stack([source[calibration][labels[calibration] == c].mean(0) for c in classes])
    counts = np.asarray([(labels[calibration] == c).sum() for c in classes])
    residual = source[calibration]-anchors[labels[calibration]]
    penalty = max(float(np.quantile((residual**2).sum(1), .99)), 1e-8)
    result = fit_robust_capacity(target, anchors, penalty, prior_strength=counts,
                                 reference_samples=float(counts.mean()), birth_order='before_update')
    if not result['converged']:
        raise RuntimeError('Capacity inference did not converge; do not silently pass count to RTA')
    settings = dict(version='source-precision-capacity-v1', C=len(classes), K=result['K'],
                    calibration_fraction=.7, calibration_seed=2026,
                    calibration_rows=len(calibration), validation_rows=len(validation),
                    source_rows=len(source), target_rows=len(target), penalty=penalty,
                    quantile=.99, source_prior_counts=counts.tolist(),
                    reference_samples=float(counts.mean()), known_centers_fixed=False,
                    birth_order='before_update', prediction_change=False,
                    classifier_prototype_initialization=False, target_labels_used=False,
                    semantic_unknown_count=None,
                    zero_capacity_policy='return 0 explicitly; RTA caller must not force K1',
                    caveat='empirical power-weighted objective, not full DP posterior')
    return result, settings
