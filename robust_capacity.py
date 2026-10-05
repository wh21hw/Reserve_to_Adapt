"""Source-anchored capped penalized clustering; a proposed extension, not DP posterior.

J = w * sum_i min(min_j squared_distance(x_i, mu_j), lambda)
    + beta * K_unknown + kappa * sum_known squared_distance(mu_c, anchor_c).
beta defaults to lambda; an explicit birth_penalty separates the two costs.
Default w=1 preserves the original probe. With reference_samples R, w=R/N.
Noise has assignment -1 and is not counted as a semantic unknown component.
"""
import numpy as np


def fit_robust_capacity(target, anchors, penalty, prior_strength=5., max_steps=100,
                        max_candidates=100, reference_samples=None, birth_order='after_update',
                        known_centers_fixed=False, birth_penalty=None, proposal_block_size=None,
                        shared_shift_precision=None):
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
    if shared_shift_precision is not None and (not np.isfinite(shared_shift_precision)
            or shared_shift_precision <= 0 or known_centers_fixed):
        raise ValueError('Shared shift requires positive precision and movable known centers')
    shift = np.zeros(a.shape[1], dtype=np.float64)
    cost = float(penalty if birth_penalty is None else birth_penalty)
    if not np.isfinite(cost) or cost <= 0:
        raise ValueError('birth_penalty must be finite and positive')
    if birth_order not in ('before_update', 'after_update'):
        raise ValueError('Invalid birth_order')
    if proposal_block_size is not None and (isinstance(proposal_block_size, bool)
            or not isinstance(proposal_block_size, (int, np.integer)) or proposal_block_size < 1):
        raise ValueError('proposal_block_size must be a positive integer or None')
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
                     + cost*(len(mu)-c) + (prior*((mu[:c]-a-shift)**2).sum(1)).sum()
                     + (0. if shared_shift_precision is None else shared_shift_precision*shift.dot(shift)))

    def assign(mu):
        d = distances(x, mu)
        ids = d.argmin(1)
        ids[d.min(1) >= penalty] = -1
        return ids

    # Optional exact candidate-column blocks avoid storing the N x N matrix.
    # All sample proposals and the original row reduction remain included.
    # This bounds memory, not the quadratic proposal-computation time.
    pairwise = distances(x, x) if proposal_block_size is None else None
    def propose(mu):
        residual = np.minimum(distances(x, mu).min(1), penalty)
        if pairwise is not None:
            gain = weight*np.maximum(residual[:, None]-pairwise, 0.).sum(0)-cost
        else:
            gain = np.empty(len(x), dtype=np.float64)
            for start in range(0, len(x), proposal_block_size):
                end = min(start+proposal_block_size, len(x))
                block = distances(x, x[start:end])
                gain[start:end] = weight*np.maximum(residual[:, None]-block, 0.).sum(0)-cost
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
        if shared_shift_precision is not None:
            # Exact joint minimizer of the known-center/shift quadratic for
            # fixed assignments. Empty anchored classes follow the shared shift.
            mass = weight*np.array([(ids == j).sum() for j in range(c)])
            sums = np.stack([weight*x[ids == j].sum(0) for j in range(c)])
            denom = mass+prior
            fractions = np.divide(prior, denom, out=np.zeros(c), where=denom > 0)
            effective = fractions*mass
            shift = (fractions[:, None]*(sums-mass[:, None]*a)).sum(0)/(shared_shift_precision+effective.sum())
        # Fixed assignments: anchored means minimize the same quadratic objective.
        for j in range(len(centers)):
            members = x[ids == j]
            if j < c:
                if known_centers_fixed:
                    continue
                if weight*len(members)+prior[j] > 0:
                    centers[j] = (weight*members.sum(0)+prior[j]*(a[j]+shift))/(weight*len(members)+prior[j])
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
                penalty=float(penalty), birth_penalty=cost, prior_strength=prior.tolist(),
                reference_samples=reference_samples, observation_weight=weight,
                birth_order=birth_order,
                known_centers_fixed=bool(known_centers_fixed),
                proposal_block_size=proposal_block_size,
                shared_shift_precision=shared_shift_precision,
                shared_shift=shift.tolist(),
                semantic_unknown_count=None)
