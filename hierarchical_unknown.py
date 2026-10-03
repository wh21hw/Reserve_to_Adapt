"""Unknown components are latent: marginalize them into ONE semantic output.

Finite discriminative mixture control, NOT a DP posterior or class-count proof.
reference_mass=2 preserves the original two-slot control's total prior scale.
"""
import math
import torch


def marginalize_unknown(logits, known_classes, log_weights, reference_mass=2.):
    if (not isinstance(known_classes,int) or isinstance(known_classes,bool)
            or known_classes < 1 or logits.ndim != 2 or len(logits) < 1
            or logits.shape[1] <= known_classes or not logits.is_floating_point()):
        raise ValueError('Expected finite known+latent logits, with at least one latent component')
    count=logits.shape[1]-known_classes
    if (log_weights.shape != (count,) or log_weights.device != logits.device
            or log_weights.dtype != logits.dtype or not torch.isfinite(logits).all()
            or not torch.isfinite(log_weights).all()
            or not torch.allclose(log_weights.logsumexp(0),log_weights.new_zeros(()),atol=1e-6,rtol=0)):
        raise ValueError('Component log weights must be finite, normalized, matching dtype/device')
    if not math.isfinite(reference_mass) or reference_mass <= 0:
        raise ValueError('Reference group mass must be finite and positive')
    weighted=logits[:,known_classes:]+log_weights
    log_evidence=weighted.logsumexp(1,keepdim=True)
    semantic_logits=torch.cat([logits[:,:known_classes],log_evidence+math.log(reference_mass)],1)
    log_conditional=weighted-log_evidence
    return semantic_logits,log_conditional


def semantic_entropy(semantic_logits, instance_weight=None):
    log_prob=semantic_logits.log_softmax(1)
    values=-(log_prob.exp()*log_prob).sum(1)
    if instance_weight is not None:
        if instance_weight.shape != values.shape or not torch.isfinite(instance_weight).all() or (instance_weight<0).any():
            raise ValueError('Invalid instance weights')
        values=values*instance_weight
    return values.mean()
