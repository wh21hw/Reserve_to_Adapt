"""Fixed group-mass control for an exploratory capacity ablation, not a DPMM.

All known logits unchanged. Unknown logits receive log(reference_slots / K).
This removes duplicated-slot inflation of GROUP probability, not every
capacity dependence of argmax decisions, pseudo-label CE, or flat entropy.
"""
import math
import torch


def uniform_capacity_log_prior(known_classes, unknown_slots, reference_slots=2,
                               *, device=None, dtype=torch.float32):
    for value in (known_classes, unknown_slots, reference_slots):
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            raise ValueError('Class/slot counts must be positive integers')
    prior = torch.zeros(known_classes + unknown_slots, device=device, dtype=dtype)
    prior[known_classes:] = math.log(reference_slots / unknown_slots)
    return prior


def apply_capacity_log_prior(logits, prior):
    if (logits.ndim != 2 or prior.shape != (logits.shape[1],)
            or logits.device != prior.device or logits.dtype != prior.dtype
            or not logits.is_floating_point() or not torch.isfinite(logits).all()
            or not torch.isfinite(prior).all()):
        raise ValueError('Invalid logits/prior shapes, dtype, device or values')
    return logits + prior
