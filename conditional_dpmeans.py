"""DP-means on a preselected candidate pool, not a semantic-count guarantee."""
import numpy as np


def fit_candidate_dpmeans(features, penalty, max_steps=100, max_clusters=100):
    x = np.asarray(features, dtype=np.float64)
    if x.ndim != 2 or not len(x) or not np.isfinite(x).all() or penalty <= 0 or not np.isfinite(penalty):
        raise ValueError('Invalid candidate features or squared-distance penalty')
    if max_steps < 1 or max_clusters < 1:
        raise ValueError('Invalid compute limits')
    order = np.lexsort([x[:, j] for j in reversed(range(x.shape[1]))])
    x = x[order]
    centers = x.mean(0, keepdims=True)
    tolerance = 1e-10 * max(1., len(x) * penalty)

    def distance(mu):
        return np.maximum((x*x).sum(1)[:, None] + (mu*mu).sum(1)[None] - 2*x.dot(mu.T), 0.)

    def objective(mu):
        return float(distance(mu).min(1).sum() + penalty*len(mu))

    history = [dict(step=0, K=1, objective=objective(centers))]
    converged = False
    for step in range(1, max_steps + 1):
        before = objective(centers)
        # Each birth decreases J: the farthest point alone saves > lambda.
        residual = distance(centers).min(1)
        if residual.max() > penalty + tolerance:
            if len(centers) >= max_clusters:
                raise RuntimeError('Cluster limit reached; count is censored')
            centers = np.vstack([centers, x[int(residual.argmax())]])
        ids = distance(centers).argmin(1)
        centers = np.stack([x[ids == j].mean(0) for j in range(len(centers)) if (ids == j).any()])
        after = objective(centers)
        if after > before + tolerance:
            raise RuntimeError('DP-means objective increased')
        history.append(dict(step=step, K=len(centers), objective=after))
        if before - after <= tolerance:
            converged = True
            break
    sorted_ids = distance(centers).argmin(1)
    assignments = np.empty(len(x), dtype=int)
    assignments[order] = sorted_ids
    return dict(K=len(centers), centers=centers, assignments=assignments,
                counts=np.bincount(assignments, minlength=len(centers)),
                converged=converged, history=history)
