"""Candidate identity reconciliation, not a guarantee of known semantics.

Assumes all C source identities occur in the target and one prototype per class.
Match C identities to occupied target clusters using known-only head likelihood;
empty source anchors do not count as target structures. No target truth input.
"""
import numpy as np
from scipy.optimize import linear_sum_assignment


def reconcile_known_identities(assignments, known_logits, classes):
    ids = np.asarray(assignments)
    logits = np.asarray(known_logits, dtype=np.float64)
    if ids.ndim != 1 or ids.dtype.kind not in 'iu' or (ids < -1).any():
        raise ValueError('Expected integer cluster IDs; noise is -1')
    if logits.shape != (len(ids), classes) or classes < 1 or not np.isfinite(logits).all():
        raise ValueError('Expected finite known-only logits in identical sample order')
    clusters = np.unique(ids[ids >= 0])
    if len(clusters) < classes:
        raise ValueError('Insufficient occupied clusters for all-known one-to-one assumption')
    value = logits - logits.max(axis=1, keepdims=True)
    log_probability = value - np.log(np.exp(value).sum(axis=1, keepdims=True))
    # Equal cluster weighting: no dominant mixed cluster wins by size alone.
    cost = np.stack([-log_probability[ids == cluster].mean(0) for cluster in clusters], axis=1)
    known, columns = linear_sum_assignment(cost)
    mapping = {int(clusters[column]):int(label) for label,column in zip(known,columns)}
    unmatched = [int(cluster) for cluster in clusters if int(cluster) not in mapping]
    mapping.update({cluster:classes+i for i,cluster in enumerate(unmatched)})
    reconciled = np.full(len(ids), -1, dtype=np.int64)
    for cluster, label in mapping.items():
        reconciled[ids == cluster] = label
    return dict(assignments=reconciled, K=len(unmatched), occupied_clusters=len(clusters),
        cluster_to_identity=mapping, known_matching_cost=cost[known,columns].tolist(),
        assumptions='Every source identity occurs in target; one target prototype per known identity',
        target_labels_used=False, semantic_unknown_count=None)
