"""Finite conditional unknown mixture: fixed count prior and detached geometry.

This is NOT a Dirichlet-process posterior or an estimate of semantic class count.
Inputs contain no target semantic labels. Keep group-level unknown evidence and
conditional component structure separate (see hierarchical_unknown.py).
"""
import torch


def _finite(value, name):
    if not torch.is_floating_point(value) or not torch.isfinite(value).all():
        raise ValueError(f'{name} must be a finite floating tensor')


def dirichlet_log_prior(expected_counts, concentration=1.0, base_weights=None):
    """Posterior mean (n + alpha*b)/(sum(n)+alpha), using frozen soft counts.

    Pass base_weights explicitly when splitting only some components: both
    expected counts and base prior mass must split together. Default is uniform.
    This finite conjugate update conditions on responsibilities; it does not
    integrate uncertainty in representation, assignments, or number of atoms.
    """
    _finite(expected_counts, 'expected_counts')
    if expected_counts.ndim != 1 or expected_counts.numel() == 0 or (expected_counts < 0).any():
        raise ValueError('Expected nonempty nonnegative component counts')
    alpha = expected_counts.new_tensor(concentration)
    if alpha.ndim != 0 or not torch.isfinite(alpha) or alpha <= 0:
        raise ValueError('concentration must be finite and positive')
    if base_weights is None:
        base_weights = torch.ones_like(expected_counts)
    _finite(base_weights, 'base_weights')
    if base_weights.shape != expected_counts.shape or (base_weights <= 0).any():
        raise ValueError('Base weights must match counts and be strictly positive')
    base = base_weights.detach().to(expected_counts) / base_weights.sum().to(expected_counts)
    posterior = expected_counts.detach() + alpha * base
    log_prior = posterior.log() - posterior.sum().log()
    _finite(log_prior, 'log_prior')
    return log_prior.detach()


def geometric_log_teacher(features, centers, log_prior, variance, variance_floor=1e-8):
    """Detached log q(k|x,U) from fixed centers and source-calibrated variance."""
    for tensor, name in ((features, 'features'), (centers, 'centers'), (log_prior, 'log_prior')):
        _finite(tensor, name)
    if features.ndim != 2 or centers.ndim != 2 or features.shape[1] != centers.shape[1]:
        raise ValueError('Feature and center dimensions must match')
    if log_prior.shape != (centers.shape[0],) or centers.shape[0] == 0:
        raise ValueError('Prior must match nonempty centers')
    scale = features.new_tensor(variance)
    floor = features.new_tensor(variance_floor)
    if scale.ndim != 0 or not torch.isfinite(scale) or scale < 0:
        raise ValueError('Variance must be finite, scalar and nonnegative')
    if floor.ndim != 0 or not torch.isfinite(floor) or floor <= 0:
        raise ValueError('Variance floor must be finite and positive')
    distance = (features.detach()[:, None, :] - centers.detach().to(features)[None, :, :]).square().sum(-1)
    logits = log_prior.detach().to(features)[None, :] - distance / (2 * scale.clamp_min(floor))
    teacher = logits.log_softmax(-1)
    _finite(teacher, 'teacher')
    return teacher.detach()


def conditional_structure_kl(student_log_probs, teacher_log_probs, unknown_weights=None):
    """KL(q_teacher || q_student) only inside U; normalize by weight mass.

    Teacher/gate detached. A zero-weight batch produces zero loss/gradient.
    Student log probabilities must already include the chosen mixture prior.
    """
    _finite(student_log_probs, 'student_log_probs')
    _finite(teacher_log_probs, 'teacher_log_probs')
    if student_log_probs.ndim != 2 or student_log_probs.shape != teacher_log_probs.shape:
        raise ValueError('Student and teacher shapes must match')
    teacher = teacher_log_probs.detach().to(student_log_probs)
    for value in (student_log_probs, teacher):
        if not torch.allclose(value.logsumexp(-1), value.new_zeros(value.shape[0]), atol=1e-5, rtol=0):
            raise ValueError('Expected normalized conditional log probabilities')
    per_row = (teacher.exp() * (teacher - student_log_probs)).sum(-1)
    if unknown_weights is None:
        unknown_weights = torch.ones_like(per_row)
    _finite(unknown_weights, 'unknown_weights')
    if unknown_weights.shape != per_row.shape or (unknown_weights < 0).any():
        raise ValueError('Unknown weights must be nonnegative and match batch')
    weights = unknown_weights.detach().to(per_row)
    loss = (per_row * weights).sum() / weights.sum().clamp_min(1e-8)
    _finite(loss, 'structure_loss')
    return loss
