"""Source-anchored IMP-inspired prototype inference, not a full DP posterior.

Known prototype identities are retained. Newly created components are candidates,
NOT automatically unknown semantic classes. This first experiment uses a declared
distance threshold and shared observation variance to isolate the source prior.
"""
import math
import numpy as np

import torch


class SourceAnchoredIMP:
    def __init__(self, threshold, observation_variance, prior_strength=5.0,
                 steps=5, max_prototypes=100, min_variance=1e-8,
                 known_centers_fixed=False):
        for name, value in [('threshold', threshold), ('observation_variance', observation_variance),
                            ('min_variance', min_variance)]:
            if not math.isfinite(value) or value <= 0:
                raise ValueError(name + ' must be finite and positive')
        if not math.isfinite(prior_strength) or prior_strength < 0:
            raise ValueError('prior_strength must be finite and nonnegative')
        if steps < 1 or max_prototypes < 1:
            raise ValueError('steps and max_prototypes must be positive')
        self.threshold = threshold
        self.observation_variance = observation_variance
        self.prior_strength = prior_strength
        self.steps = steps
        self.max_prototypes = max_prototypes
        self.min_variance = min_variance
        self.known_centers_fixed = bool(known_centers_fixed)

    def _assign(self, features, centers, known_count=None, known_compatibility=None):
        distances = (features[:, None] - centers[None]).square().sum(dim=-1)
        variance = max(self.observation_variance, self.min_variance,
                       torch.finfo(features.dtype).tiny)
        shifted = distances - distances.min(dim=1, keepdim=True)[0]
        logits = -shifted / (2 * variance)
        if known_compatibility is not None and len(centers) > known_count:
            # Hierarchical mixture weights: fixed compatibility is shared across
            # known components, its complement across candidate components.
            # This is a plug-in relation prior, NOT a calibrated semantic posterior.
            p = known_compatibility[:, None]
            logits[:, :known_count] += (p / known_count).log()
            logits[:, known_count:] += ((1-p) / (len(centers)-known_count)).log()
        return torch.softmax(logits, dim=1)

    @torch.no_grad()
    def fit(self, target_features, source_centers, known_compatibility=None,
            birth_unknown_min=0.5, birth_strategy='sequential'):
        for values in [target_features, source_centers]:
            if values.ndim != 2 or not values.is_floating_point() or min(values.shape) < 1:
                raise ValueError('Expected non-empty floating-point matrices')
            if not torch.isfinite(values).all():
                raise ValueError('Non-finite input')
        if target_features.shape[1] != source_centers.shape[1]:
            raise ValueError('Feature dimensions differ')
        if target_features.dtype != source_centers.dtype or target_features.device != source_centers.device:
            raise ValueError('Input dtype and device must match')
        known_count = len(source_centers)
        if not math.isfinite(birth_unknown_min) or not 0 <= birth_unknown_min <= 1:
            raise ValueError('birth_unknown_min must be in [0,1]')
        if known_compatibility is not None:
            p = known_compatibility
            if (p.shape != (len(target_features),) or p.dtype != target_features.dtype
                    or p.device != target_features.device or not torch.isfinite(p).all()
                    or (p < 0).any() or (p > 1).any()):
                raise ValueError('Compatibility must match target rows/dtype/device and be in [0,1]')
            known_compatibility = p.detach()
        if birth_strategy not in ['sequential', 'farthest']:
            raise ValueError('Unknown birth_strategy')
        restore_order = None
        if birth_strategy == 'farthest':
            # Canonical reductions/tie-breaking remove input-order randomness.
            # This is deterministic geometry, not evidence of semantic correctness.
            values = target_features.detach().double().cpu().numpy()
            keys = ([known_compatibility.double().cpu().numpy()] if known_compatibility is not None else [])
            keys += [values[:, column] for column in reversed(range(values.shape[1]))]
            order = torch.as_tensor(np.lexsort(keys).copy(), device=target_features.device)
            restore_order = order.argsort()
            target_features = target_features[order]
            if known_compatibility is not None:
                known_compatibility = known_compatibility[order]
        if known_count > self.max_prototypes:
            raise ValueError('Capacity is smaller than the known prototype count')
        anchors = source_centers.clone()
        centers = anchors.clone()
        history = []
        for iteration in range(self.steps):
            if birth_strategy == 'farthest':
                eligible = (1-known_compatibility >= birth_unknown_min) if known_compatibility is not None else torch.ones(len(target_features),device=target_features.device,dtype=torch.bool)
                distances = (target_features[:,None]-centers[None]).square().sum(-1).min(1)[0]
                while eligible.any():
                    maximum, index = distances.masked_fill(~eligible, -float('inf')).max(0)
                    if maximum <= self.threshold:
                        break
                    if len(centers) >= self.max_prototypes:
                        raise RuntimeError('Prototype capacity exceeded; count is censored')
                    feature = target_features[index]
                    centers = torch.cat([centers, feature[None]], dim=0)
                    distances = torch.minimum(distances, (target_features-feature).square().sum(1))
            else:
                for index, feature in enumerate(target_features):
                    if known_compatibility is not None and 1-known_compatibility[index] < birth_unknown_min:
                        continue
                    distance = (centers - feature).square().sum(dim=1).min()
                    if distance > self.threshold:
                        if len(centers) >= self.max_prototypes:
                            raise RuntimeError('Prototype capacity exceeded; count is censored')
                        centers = torch.cat([centers, feature[None]], dim=0)
            assignments = self._assign(target_features, centers, known_count, known_compatibility)
            mass = assignments.sum(dim=0)
            sums = assignments.t().matmul(target_features)
            # Conditional Gaussian MAP update with fixed responsibilities:
            # kappa is the prior-to-observation precision ratio (pseudo-count).
            updated = centers.clone()
            denom = mass[:known_count] + self.prior_strength
            supported = denom > 0
            known_update = (sums[:known_count] + self.prior_strength * anchors)
            updated[:known_count][supported] = known_update[supported] / denom[supported, None]
            if self.known_centers_fixed:
                updated[:known_count] = anchors
            if len(centers) > known_count:
                keep = mass[known_count:] >= 1e-8
                candidates = sums[known_count:][keep] / mass[known_count:][keep, None]
                centers = torch.cat([updated[:known_count], candidates], dim=0)
            else:
                centers = updated
            history.append(dict(iteration=iteration + 1, prototypes=len(centers)))
        assignments = self._assign(target_features, centers, known_count, known_compatibility)
        if restore_order is not None:
            assignments = assignments[restore_order]
        if not torch.isfinite(centers).all() or not torch.isfinite(assignments).all():
            raise RuntimeError('Non-finite inference result')
        return dict(known_prototypes=centers[:known_count],
                    candidate_prototypes=centers[known_count:],
                    responsibilities=assignments, effective_counts=assignments.sum(dim=0),
                    history=history, known_count=known_count,
                    candidate_count=len(centers) - known_count,
                    relation_constrained=known_compatibility is not None,
                    known_centers_fixed=self.known_centers_fixed,
                    birth_strategy=birth_strategy,
                    compatibility_ignored_without_candidates=known_compatibility is not None and len(centers) == known_count,
                    semantic_unknown_count=None)
