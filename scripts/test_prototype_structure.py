"""CPU algebra/gradient checks; no labels, no training, no candidate-count claim."""
import json
from pathlib import Path
import sys
import torch

sys.path.insert(0, '/content')
from prototype_structure import dirichlet_log_prior, geometric_log_teacher, conditional_structure_kl

torch.set_num_threads(2)
torch.manual_seed(1)
counts = torch.tensor([7., 3.], dtype=torch.float64)
prior = dirichlet_log_prior(counts)
assert torch.allclose(prior.exp(), torch.tensor([7.5/11, 3.5/11], dtype=counts.dtype))
features = torch.randn(5, 3, dtype=counts.dtype, requires_grad=True)
centers = torch.randn(2, 3, dtype=counts.dtype, requires_grad=True)
teacher = geometric_log_teacher(features, centers, prior, .2)
raw = torch.randn(5, 2, dtype=counts.dtype, requires_grad=True)
student = (raw + prior).log_softmax(-1)
weights = torch.tensor([1., .2, 0., .7, 1.], dtype=counts.dtype, requires_grad=True)
loss = conditional_structure_kl(student, teacher, weights)
gradient = torch.autograd.grad(loss, raw, retain_graph=True)[0]
# Split ONLY the second component; carry its base prior mass as well as counts.
index = torch.tensor([0, 1, 1])
split_prior = dirichlet_log_prior(torch.tensor([7., 1.5, 1.5], dtype=counts.dtype),
                                  base_weights=torch.tensor([.5, .25, .25], dtype=counts.dtype))
split_teacher = geometric_log_teacher(features, centers[index], split_prior, .2)
split_student = (raw[:, index] + split_prior).log_softmax(-1)
split_loss = conditional_structure_kl(split_student, split_teacher, weights)
split_gradient = torch.autograd.grad(split_loss, raw, retain_graph=True)[0]
assert torch.allclose(loss, split_loss, atol=1e-13, rtol=1e-13)
assert torch.allclose(gradient, split_gradient, atol=1e-13, rtol=1e-13)
loss.backward()
assert features.grad is None and centers.grad is None and weights.grad is None
assert torch.isfinite(raw.grad).all()
zero_student = (raw + prior).log_softmax(-1)  # new graph after the earlier backward
zero_loss = conditional_structure_kl(zero_student, teacher, torch.zeros(5, dtype=counts.dtype))
assert zero_loss.item() == 0
assert torch.equal(torch.autograd.grad(zero_loss, raw)[0], torch.zeros_like(raw))
zero_variance = geometric_log_teacher(features, centers, prior, 0.)
assert torch.isfinite(zero_variance).all()
for invalid in (torch.tensor([float('nan'), 1.]), torch.tensor([-1., 1.])):
    try:
        dirichlet_log_prior(invalid)
    except ValueError:
        pass
    else:
        raise AssertionError('Invalid counts accepted')
report = dict(posterior_mean=True, partial_component_split_loss_invariant=True,
              shared_logit_gradient_invariant=True, teacher_and_gate_detached=True,
              zero_weight_zero_gradient=True, zero_variance_positive_floor=True,
              target_labels_read=False, optimizer_steps=0, device='CPU')
with Path('/content/prototype-structure-unit-v1.json').open('x') as stream:
    json.dump(report, stream, indent=2)
print('PROTOTYPE_STRUCTURE_UNIT_PASS', json.dumps(report), flush=True)
