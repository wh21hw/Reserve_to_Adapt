"""Frozen CPU eval-head gradient audit including actual BN/activation; no SGD."""
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, '/content')
from prototype_structure import dirichlet_log_prior, geometric_log_teacher, conditional_structure_kl
from hierarchical_unknown import marginalize_unknown, semantic_entropy

for filename, expected in {
    'prototype_structure.py': '4f7ed0fa3bce8f2a2cea88d964917c3f69e333ee6b5257aa5e218deb04f964d8',
    'hierarchical_unknown.py': '58e9a31e674926c76aa52a8f13799a575d9dcc41a86da6d3a03141494fd76fc7',
}.items():
    assert hashlib.sha256((Path('/content') / filename).read_bytes()).hexdigest() == expected
torch.set_num_threads(2)
feature_path = Path('/content/imp-runs/multitask-a2w-seed1-warm-features-v1/features.npz')
assert hashlib.sha256(feature_path.read_bytes()).hexdigest() == 'a9d1a40aeb863b72aafefaed965d22d8eb3b3099516aee117f1897bdb4fb5459'
features = np.load(feature_path, allow_pickle=False)
assert 'target_labels' not in features.files
proposal = np.load('/content/imp-runs/matched-warm-candidates-v1/original.npz', allow_pickle=False)
source = torch.from_numpy(features['source']).double()
target = torch.from_numpy(features['target']).double()
labels = torch.from_numpy(features['source_labels'])
source_centers = torch.stack([source[labels == c].mean(0) for c in range(10)])
variance = float((source - source_centers[labels]).square().mean().clamp_min(1e-8))
counts = proposal['responsibilities'].sum(0)[10:]
selected = sorted(np.flatnonzero(counts >= 5), key=lambda index: (-counts[index], index))
assert selected
centers = torch.as_tensor(proposal['candidates'][selected], dtype=torch.float64)
prior = dirichlet_log_prior(torch.as_tensor(counts[selected], dtype=torch.float64), concentration=1.)
teacher = geometric_log_teacher(target, centers, prior, variance)
known_p = torch.from_numpy(proposal['known_compatibility']).double()
unknown_p = 1 - known_p
checkpoint_path = Path('/content/imp-runs/rta-multitask-baseline-v1/office31-a2w_seed1/warmup-complete.pt')
assert hashlib.sha256(checkpoint_path.read_bytes()).hexdigest() == 'ff88bdc9d49a1a8c151baa8b4ca166549a2723195882ec37fe5056e0d91f1eb1'
checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
state = checkpoint['model']
old_weight = state['1.fc.weight'].double()
known_w = old_weight[:10]
def head_inputs(value):
    prefix = '1.main.1.0.'
    normalized = F.batch_norm(value, state[prefix + 'running_mean'].double(),
                              state[prefix + 'running_var'].double(),
                              state[prefix + 'weight'].double(), state[prefix + 'bias'].double(),
                              training=False, eps=1e-5)
    return F.leaky_relu(normalized, negative_slope=.2)

parity_errors = {}
for name, value in (('source', source), ('target', target)):
    logits = head_inputs(value) @ old_weight.T
    cached = torch.from_numpy(features[name + '_logits']).double()
    error = float((logits - cached).abs().max())
    assert torch.allclose(logits, cached, atol=1e-5, rtol=1e-5), f'Eval-head parity failed: {name}, {error}'
    parity_errors[name] = error
weight = torch.cat([known_w, F.normalize(centers, dim=1) * known_w.norm(dim=1).mean()]).requires_grad_()
target = target.requires_grad_()
semantic, conditional = marginalize_unknown(head_inputs(target) @ weight.T, 10, prior)
source_semantic, _ = marginalize_unknown(head_inputs(source) @ weight.T, 10, prior)
losses = dict(source_ce=F.cross_entropy(source_semantic, labels),
              unknown_group_proxy=(-(semantic.log_softmax(1)[:, 10]) * unknown_p).mean(),
              semantic_entropy_proxy=semantic_entropy(semantic, known_p),
              conditional_structure=conditional_structure_kl(conditional, teacher, unknown_p))
gradients = {}
report = dict(feature_sha256=hashlib.sha256(feature_path.read_bytes()).hexdigest(),
              proposal_sha256=hashlib.sha256(Path('/content/imp-runs/matched-warm-candidates-v1/original.npz').read_bytes()).hexdigest(),
              checkpoint_sha256=hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
              selected_indices=[int(index) for index in selected], components=len(selected),
              unknown_gate_mass=float(unknown_p.sum()), unknown_gate_ge_half=int((unknown_p >= .5).sum()),
              prior_weights=prior.exp().tolist(), variance=variance,
              gate_weighted_teacher_entropy=float((-(teacher.exp() * teacher).sum(1) * unknown_p).sum() / unknown_p.sum()),
              eval_head_logit_max_abs_errors=parity_errors,
              losses={}, comparisons={}, target_labels_read=False, optimizer_steps=0,
              caveat='Frozen eval BN/LeakyReLU/head proxy; no train-BN updates, image augmentation, backbone gradients, actual RTA top16 selection, virtual/adversarial loss or SGD. No lambda selected.')
for name, loss in losses.items():
    grads = torch.autograd.grad(loss, (weight, target), retain_graph=True, allow_unused=True)
    head, feature = grads
    assert torch.isfinite(head).all() and (feature is None or torch.isfinite(feature).all())
    gradients[name] = head
    report['losses'][name] = dict(value=float(loss.detach()), head_norm=float(head.norm()),
                                known_head_norm=float(head[:10].norm()), unknown_head_norm=float(head[10:].norm()),
                                feature_norm=None if feature is None else float(feature.norm()))
structure = gradients['conditional_structure']
assert torch.count_nonzero(structure[:10]) == 0
for name in ('source_ce', 'unknown_group_proxy', 'semantic_entropy_proxy'):
    reference = gradients[name]
    report['comparisons'][name] = dict(structure_head_norm_ratio=float(structure.norm() / reference.norm().clamp_min(1e-12)),
                                       head_cosine=float(F.cosine_similarity(structure.flatten(), reference.flatten(), dim=0)))
with Path('/content/matched-structure-gradient-v2.json').open('x') as stream:
    json.dump(report, stream, indent=2, allow_nan=False)
print('MATCHED_STRUCTURE_GRADIENT_PASS', json.dumps(report), flush=True)
