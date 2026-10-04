"""Source-anchored capped penalized clustering; a proposed extension, not DP posterior.

J = w * sum_i min(min_j squared_distance(x_i, mu_j), lambda)
    + lambda * K_unknown + kappa * sum_known squared_distance(mu_c, anchor_c).
Default w=1 preserves the original probe. With reference_samples R, w=R/N.
Noise has assignment -1 and is not counted as a semantic unknown component.
"""
import numpy as np


def fit_robust_capacity(target, anchors, penalty, prior_strength=5., max_steps=100,
                        max_candidates=100, reference_samples=None, birth_order='after_update',
                        known_centers_fixed=False):
    x, a = np.asarray(target, dtype=np.float64), np.asarray(anchors, dtype=np.float64)
    if (x.ndim != 2 or a.ndim != 2 or not len(x) or not len(a)
            or x.shape[1] != a.shape[1] or not np.isfinite(x).all()
            or not np.isfinite(a).all() or not np.isfinite(penalty)
            or penalty <= 0
            or max_steps < 1 or max_candidates < 1):
        raise ValueError('Invalid features or settings')
    prior = np.broadcast_to(np.asarray(prior_strength, dtype=np.float64), (len(a),)).copy()
    if not np.isfinite(prior).all() or (prior < 0).any():
        raise ValueError('Invalid prior precision')
    if reference_samples is not None and (not np.isfinite(reference_samples) or reference_samples <= 0):
        raise ValueError('reference_samples must be finite and positive')
    weight = 1. if reference_samples is None else float(reference_samples)/len(x)
    if birth_order not in ('before_update', 'after_update'):
        raise ValueError('Invalid birth_order')
    # Canonical ordering makes proposal ties independent of incoming row order.
    order = np.lexsort([x[:, j] for j in reversed(range(x.shape[1]))])
    x = x[order]
    c, centers = len(a), a.copy()
    tolerance = 1e-10 * max(1., weight*len(x)*penalty)

    def distances(left, right):
        return np.maximum((left*left).sum(1)[:, None] + (right*right).sum(1)[None]
                          - 2*left.dot(right.T), 0.)

    def objective(mu):
        return float(weight*np.minimum(distances(x, mu).min(1), penalty).sum()
                     + penalty*(len(mu)-c) + (prior*((mu[:c]-a)**2).sum(1)).sum())

    def assign(mu):
        d = distances(x, mu)
        ids = d.argmin(1)
        ids[d.min(1) >= penalty] = -1
        return ids

    pairwise = distances(x, x)
    def propose(mu):
        residual = np.minimum(distances(x, mu).min(1), penalty)
        gain = weight*np.maximum(residual[:, None]-pairwise, 0.).sum(0)-penalty
        proposal = int(gain.argmax())
        if gain[proposal] > tolerance:
            if len(mu)-c >= max_candidates:
                raise RuntimeError('Candidate capacity exhausted; count is censored')
            return np.vstack([mu, x[proposal]])
        return mu
    history = [dict(step=0, objective=objective(centers), K=0)]
    converged = False
    for step in range(1, max_steps+1):
        before = objective(centers)
        if birth_order == 'before_update':
            centers = propose(centers)
        ids = assign(centers)
        # Fixed assignments: anchored means minimize the same quadratic objective.
        for j in range(len(centers)):
            members = x[ids == j]
            if j < c:
                if known_centers_fixed:
                    continue
                if weight*len(members)+prior[j] > 0:
                    centers[j] = (weight*members.sum(0)+prior[j]*a[j])/(weight*len(members)+prior[j])
            elif len(members):
                centers[j] = members.mean(0)
        # Delete a candidate only if the full penalized objective decreases.
        while len(centers) > c:
            trials = [np.delete(centers, j, axis=0) for j in range(c, len(centers))]
            costs = np.asarray([objective(mu) for mu in trials])
            best = costs.argmin()
            if costs[best] >= objective(centers)-tolerance:
                break
            centers = trials[best]
        # Whole-data residual improvement, not one-point distance threshold.
        if birth_order == 'after_update':
            centers = propose(centers)
        after = objective(centers)
        if after > before+tolerance:
            raise RuntimeError('Objective increased')
        history.append(dict(step=step, objective=after, K=len(centers)-c))
        if before-after <= tolerance:
            converged = True
            break
    ids = assign(centers)
    original_ids = np.empty_like(ids)
    original_ids[order] = ids
    return dict(centers=centers, assignments=original_ids, K=len(centers)-c,
                counts=np.asarray([(ids == j).sum() for j in range(len(centers))]),
                noise_count=int((ids == -1).sum()), converged=converged, history=history,
                penalty=float(penalty), prior_strength=prior.tolist(),
                reference_samples=reference_samples, observation_weight=weight,
                birth_order=birth_order,
                known_centers_fixed=bool(known_centers_fixed),
                semantic_unknown_count=None)
