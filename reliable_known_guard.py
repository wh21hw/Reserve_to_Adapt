"""Source-calibrated agreement guard; not a guarantee of semantic knownness."""
import numpy as np


def known_probabilities(logits, classes):
    value = np.asarray(logits, dtype=np.float64)[:, :classes]
    value = value - value.max(1, keepdims=True)
    value = np.exp(value)
    return value / value.sum(1, keepdims=True)


def infer_protected_known(source, labels, source_logits, target, target_logits,
                          radius_quantile=.99, confidence_quantile=.01):
    """Uses source truth, target features/predictions only; no target truth input.

    The known-only probabilities deliberately do not require a known output to
    beat an unknown output: that competition is the failure being investigated.
    Thresholds are per-source-class and the quantiles must be predeclared.
    """
    source, target = np.asarray(source, dtype=np.float64), np.asarray(target, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    classes = len(np.unique(labels))
    if not np.array_equal(np.unique(labels), np.arange(classes)):
        raise ValueError('Source labels must be contiguous known identities')
    if source.ndim != 2 or target.ndim != 2 or source.shape[1] != target.shape[1]:
        raise ValueError('Feature dimensions differ')
    if len(labels) != len(source) or len(source_logits) != len(source) or len(target_logits) != len(target):
        raise ValueError('Row ordering/length mismatch')
    if not 0 < radius_quantile < 1 or not 0 < confidence_quantile < 1:
        raise ValueError('Quantiles must be interior')
    if not all(np.isfinite(a).all() for a in (source, target, source_logits, target_logits)):
        raise ValueError('Nonfinite guard inputs')
    centers = np.stack([source[labels == c].mean(0) for c in range(classes)])
    ps, pt = known_probabilities(source_logits, classes), known_probabilities(target_logits, classes)
    radii = np.array([np.quantile(((source[labels == c] - centers[c]) ** 2).sum(1), radius_quantile)
                      for c in range(classes)])
    confidence = np.array([np.quantile(ps[labels == c, c], confidence_quantile)
                           for c in range(classes)])
    head_identity = pt.argmax(1)
    nearest, distance = [], []
    for start in range(0, len(target), 2048):
        rows = target[start:start + 2048]
        d = np.maximum((rows * rows).sum(1)[:, None] + (centers * centers).sum(1)[None, :]
                       - 2 * rows.dot(centers.T), 0)
        nearest.append(d.argmin(1))
        distance.append(d.min(1))
    nearest, distance = np.concatenate(nearest), np.concatenate(distance)
    agreement = nearest == head_identity
    protected = agreement & (distance <= radii[nearest]) & (pt.max(1) >= confidence[nearest])
    return dict(protected_known=protected, known_identity=nearest,
                agreement=agreement, radii=radii, confidence=confidence,
                radius_quantile=radius_quantile, confidence_quantile=confidence_quantile)


def cluster_supported_known(guard, assignments, minimum_fraction=.5):
    """Require a strict majority of the entire cluster to support one identity.

    A candidate cluster is NOT automatically unknown or known. Noise never gets
    cluster support. Output only filters existing eligible points, not labels
    previously uncertain members by majority propagation.
    """
    ids = np.asarray(assignments, dtype=np.int64)
    eligible = np.asarray(guard['protected_known'], dtype=bool)
    identity = np.asarray(guard['known_identity'], dtype=np.int64)
    if ids.shape != eligible.shape or identity.shape != ids.shape:
        raise ValueError('Cluster/guard ordering mismatch')
    if not .5 <= minimum_fraction < 1:
        raise ValueError('Require unambiguous majority')
    keep = np.zeros(len(ids), dtype=bool)
    details = []
    for cluster in np.unique(ids[ids >= 0]):
        members = ids == cluster
        votes = identity[members & eligible]
        if len(votes) == 0:
            continue
        counts = np.bincount(votes)
        winner = int(counts.argmax())
        fraction = float(counts[winner] / members.sum())
        supported = fraction > minimum_fraction
        if supported:
            keep |= members & eligible & (identity == winner)
        details.append(dict(cluster=int(cluster), members=int(members.sum()),
                            identity=winner, fraction=fraction, supported=supported))
    return keep, details
