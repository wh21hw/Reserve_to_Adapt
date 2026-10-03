"""Source-anchored IMP-inspired prototype inference, not a full DP posterior.

Known prototype identities are retained. Newly created components are candidates,
NOT automatically unknown semantic classes. This first experiment uses a declared
distance threshold and shared observation variance to isolate the source prior.
"""
import math

import torch


class SourceAnchoredIMP:
    def __init__(self, threshold, observation_variance, prior_strength=5.0,
                 steps=5, max_prototypes=100, min_variance=1e-8):
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

    def _assign(self, features, centers):
        distances = (features[:, None] - centers[None]).square().sum(dim=-1)
        variance = max(self.observation_variance, self.min_variance,
                       torch.finfo(features.dtype).tiny)
        shifted = distances - distances.min(dim=1, keepdim=True)[0]
        return torch.softmax(-shifted / (2 * variance), dim=1)

    @torch.no_grad()
    def fit(self, target_features, source_centers):
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
        if known_count > self.max_prototypes:
            raise ValueError('Capacity is smaller than the known prototype count')
        anchors = source_centers.clone()
        centers = anchors.clone()
        history = []
        for iteration in range(self.steps):
            for feature in target_features:
                distance = (centers - feature).square().sum(dim=1).min()
                if distance > self.threshold:
                    if len(centers) >= self.max_prototypes:
                        raise RuntimeError('Prototype capacity exceeded; count is censored')
                    centers = torch.cat([centers, feature[None]], dim=0)
            assignments = self._assign(target_features, centers)
            mass = assignments.sum(dim=0)
            sums = assignments.t().matmul(target_features)
            # Conditional Gaussian MAP update with fixed responsibilities:
            # kappa is the prior-to-observation precision ratio (pseudo-count).
            updated = centers.clone()
            denom = mass[:known_count] + self.prior_strength
            supported = denom > 0
            known_update = (sums[:known_count] + self.prior_strength * anchors)
            updated[:known_count][supported] = known_update[supported] / denom[supported, None]
            if len(centers) > known_count:
                keep = mass[known_count:] >= 1e-8
                candidates = sums[known_count:][keep] / mass[known_count:][keep, None]
                centers = torch.cat([updated[:known_count], candidates], dim=0)
            else:
                centers = updated
            history.append(dict(iteration=iteration + 1, prototypes=len(centers)))
        assignments = self._assign(target_features, centers)
        if not torch.isfinite(centers).all() or not torch.isfinite(assignments).all():
            raise RuntimeError('Non-finite inference result')
        return dict(known_prototypes=centers[:known_count],
                    candidate_prototypes=centers[known_count:],
                    responsibilities=assignments, effective_counts=assignments.sum(dim=0),
                    history=history, known_count=known_count,
                    candidate_count=len(centers) - known_count,
                    semantic_unknown_count=None)
