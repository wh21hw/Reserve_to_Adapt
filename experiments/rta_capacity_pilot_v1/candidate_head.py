"""Experimental latent capacity from supported candidates, NOT semantic class count.

Call before constructing optimizers. Replacing a Parameter invalidates any existing
optimizer reference; a continuation trainer must explicitly migrate/reset states
identically in every arm. This helper never silently truncates to a true class count.
"""
import math
import torch
from torch import nn


@torch.no_grad()
def initialize_candidate_head(classifier, candidates, masses, known_classes,
                              min_effective_mass=5., max_slots=None):
    layer = classifier.fc
    if not isinstance(layer, nn.Linear) or layer.bias is not None:
        raise ValueError('Expected bias-free Linear classifier.fc')
    if not isinstance(known_classes, int) or not 0 < known_classes <= layer.out_features:
        raise ValueError('Invalid known class count')
    if (candidates.ndim != 2 or candidates.shape[1] != layer.in_features
            or masses.shape != (len(candidates),) or candidates.dtype != layer.weight.dtype
            or candidates.device != layer.weight.device or masses.device != candidates.device
            or not torch.isfinite(candidates).all() or not torch.isfinite(masses).all()
            or (masses < 0).any()):
        raise ValueError('Invalid candidate centers/supports')
    if not math.isfinite(min_effective_mass) or min_effective_mass <= 0:
        raise ValueError('Support floor must be finite and positive')
    if max_slots is not None and (not isinstance(max_slots,int) or max_slots < 1):
        raise ValueError('max_slots must be a positive integer')
    indices = (masses >= min_effective_mass).nonzero().flatten().tolist()
    indices.sort(key=lambda index: (-float(masses[index]), index))
    if max_slots is not None:
        indices = indices[:max_slots]
    if not indices:
        raise RuntimeError('No supported candidates; do not silently choose a capacity')
    selected = candidates[indices]
    if (selected.norm(dim=1) <= 1e-8).any():
        raise ValueError('Cannot initialize direction from zero candidate')
    old_parameter = layer.weight
    known_weight = old_parameter[:known_classes].detach().clone()
    scale = known_weight.norm(dim=1).mean()
    if not torch.isfinite(scale) or scale <= 1e-8:
        raise ValueError('Known head has invalid weight scale')
    weights = torch.cat([known_weight, torch.nn.functional.normalize(selected,dim=1)*scale])
    layer.weight = nn.Parameter(weights, requires_grad=old_parameter.requires_grad)
    layer.out_features = len(weights)
    return dict(unknown_capacity=len(indices), semantic_unknown_count=None,
                selected_indices=indices, selected_mass=masses[indices].tolist(),
                optimizer_rebuild_required=True)
