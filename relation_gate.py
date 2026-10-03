"""RTA released-code relation score; mixture groups are NOT semantic classes."""
import torch


@torch.no_grad()
def source_relation_scores(source_logits, source_labels, target_logits, known_classes):
    if not isinstance(known_classes, int) or known_classes < 2:
        raise ValueError('known_classes must be an integer >=2')
    for logits in [source_logits, target_logits]:
        if logits.ndim != 2 or len(logits) == 0 or logits.shape[1] < known_classes:
            raise ValueError('Invalid logit shape')
        if not logits.is_floating_point() or not torch.isfinite(logits).all():
            raise ValueError('Logits must be finite floating-point values')
    if source_logits.device != target_logits.device:
        raise ValueError('Logit devices differ')
    if source_labels.shape != (len(source_logits),) or source_labels.dtype != torch.long:
        raise ValueError('Expected one int64 source label per sample')
    if source_labels.device != source_logits.device:
        raise ValueError('Label/logit devices differ')
    if not torch.equal(torch.unique(source_labels), torch.arange(known_classes, device=source_labels.device)):
        raise ValueError('Every known class must occur; no other source class allowed')
    # Float64/log_softmax avoid log(softmax(...)) underflow, without changing KL direction.
    source_prob = source_logits[:, :known_classes].double().softmax(1)
    prototypes = torch.stack([source_prob[source_labels == c].mean(0) for c in range(known_classes)])
    target_log_prob = target_logits[:, :known_classes].double().log_softmax(1)
    pseudo = target_log_prob.argmax(1)
    selected = prototypes[pseudo]
    score = torch.nn.functional.kl_div(target_log_prob, selected, reduction='none').sum(1)
    if not torch.isfinite(score).all() or (score < -1e-10).any():
        raise RuntimeError('Invalid KL result')
    return dict(source_soft_prototypes=prototypes, pseudo_class=pseudo,
                scores=score.clamp_min(0), kl_direction='source prototype || target probability')
