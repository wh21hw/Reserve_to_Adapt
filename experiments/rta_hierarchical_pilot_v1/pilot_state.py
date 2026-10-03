"""Controlled post-warmup handoff; not an exact uninterrupted RTA resume."""
import copy
import hashlib
import random
from pathlib import Path

import faiss
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from sklearn.mixture import BayesianGaussianMixture

from candidate_head import initialize_candidate_head
from relation_gate import source_relation_scores

CHECKPOINT = Path('/content/imp-runs/rta-space-warmup-l4-v1/a2w_seed1/last.pt')
FEATURES = Path('/content/imp-runs/rta-space-warmup-l4-features-v1/features.npz')
PROPOSAL = Path('/content/imp-runs/farthest-constrained-frozen-v1/gate-seed1-original.npz')
EXPECTED = {
    CHECKPOINT: '0631572940759e0677632a1eadeb5bb4748912fa071e706a508df23411e03cd3',
    FEATURES: '75cabbddd74d812dd9780b95248e1cc7dc8147d82ac336659162a2f642241823',
    PROPOSAL: '5bfd8dd4e51d0f7a818fa240b526ed942c3f61ebb63fca48f3bc4a4145aff9c2',
}


def tensor_hash(value):
    return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def reset_seed(seed=1):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def initialize(net, discriminator, bank, arm):
    if arm not in ('original2', 'proto2', 'proto18'):
        raise ValueError('Unknown pilot arm')
    for path, expected in EXPECTED.items():
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise RuntimeError(f'Input hash mismatch: {path}')
    checkpoint = torch.load(CHECKPOINT, map_location='cpu', weights_only=False)
    if checkpoint['epoch'] != 4:
        raise RuntimeError('Expected fixed epoch4 checkpoint')
    incompatibility = net.load_state_dict(checkpoint['model'], strict=False)
    assert incompatibility.missing_keys == ['1.unknown_log_weights']
    assert not incompatibility.unexpected_keys
    discriminator.load_state_dict(checkpoint['discriminator'], strict=True)
    known = net[1].fc.weight[:10].detach().clone()
    proposal = np.load(PROPOSAL, allow_pickle=False)
    device = known.device
    if arm == 'original2':
        head_audit = dict(unknown_capacity=2, selected_indices=None,
                          semantic_unknown_count=None)
    else:
        head_audit = initialize_candidate_head(
            net[1], torch.as_tensor(proposal['candidates'], device=device, dtype=known.dtype),
            torch.as_tensor(proposal['responsibilities'].sum(0)[10:], device=device),
            10, min_effective_mass=5., max_slots=2 if arm == 'proto2' else None)
    assert head_audit['unknown_capacity'] == (18 if arm == 'proto18' else 2)
    count = head_audit['unknown_capacity']
    net[1].unknown_log_weights = torch.full((count,), -np.log(count), device=device, dtype=known.dtype)
    assert torch.equal(known, net[1].fc.weight[:10])
    assert net[1].main[1][2] is net[1].fc
    frozen = np.load(FEATURES, allow_pickle=False)
    assert 'target_labels' not in frozen.files
    relation = source_relation_scores(torch.from_numpy(frozen['source_logits']),
        torch.from_numpy(frozen['source_labels']).long(),
        torch.from_numpy(frozen['target_logits']), 10)
    bank.src_ctrs.copy_(relation['source_soft_prototypes'].to(bank.src_ctrs))
    gmm = BayesianGaussianMixture(n_components=4, max_iter=800, random_state=1).fit(
        relation['scores'].numpy()[:, None])
    if not gmm.converged_:
        raise RuntimeError('Shared initial GMM did not converge')
    source_centers = np.stack([frozen['source'][frozen['source_labels'] == c].mean(0)
                               for c in range(10)])
    kmeans = faiss.Kmeans(256, 20, niter=800, verbose=False,
                          min_points_per_centroid=1, gpu=False, seed=1)
    kmeans.train(frozen['target'])
    _, matched = linear_sum_assignment(np.linalg.norm(
        source_centers[:, None] - kmeans.centroids[None], axis=-1))
    virtual = kmeans.centroids[[i for i in range(20) if i not in matched]]
    audit = dict(arm=arm, head=head_audit, checkpoint_epoch=4,
        semantic_output_classes=11, unknown_log_weights=net[1].unknown_log_weights.cpu().tolist(),
        unknown_reference_mass=2., model='Latent unknown marginalization, not a DP posterior',
        input_sha256={str(p): h for p, h in EXPECTED.items()},
        known_weights_sha256=tensor_hash(known),
        source_relation_bank_sha256=tensor_hash(bank.src_ctrs),
        virtual_sha256=tensor_hash(torch.from_numpy(virtual)),
        gmm_means=gmm.means_.tolist(), gmm_iterations=int(gmm.n_iter_),
        state_policy='Frozen center-crop bank/GMM/virtual rebuild shared across arms; unknown momentum reset in ALL arms; no exact resume claim',
        target_labels_used_for_initialization=False)
    return checkpoint, gmm, torch.from_numpy(virtual).to(device), audit


def restore_optimizers(checkpoint, net, discriminator, wrappers, steps_per_epoch, audit):
    """Keep known-row/non-head SGD states; explicitly reset ALL unknown-row momentum."""
    feature, classifier, adversarial = wrappers
    feature.optimizer.load_state_dict(checkpoint['optimizer_feature'])
    adversarial.optimizer.load_state_dict(checkpoint['optimizer_discriminator'])
    state = copy.deepcopy(checkpoint['optimizer_cls'])
    old_ids = [i for g in state['param_groups'] for i in g['params']]
    parameters = list(net[1].parameters())
    assert len(old_ids) == len(parameters)
    for key, parameter in zip(old_ids, parameters):
        for name, value in state['state'].get(key, {}).items():
            if not torch.is_tensor(value):
                raise RuntimeError('Unexpected SGD state type')
            if parameter is net[1].fc.weight:
                assert name == 'momentum_buffer' and value.shape == (12, 256)
                momentum = torch.zeros_like(parameter, device='cpu')
                momentum[:10].copy_(value[:10])
                state['state'][key][name] = momentum
            elif value.shape != parameter.shape:
                raise RuntimeError('Non-head optimizer state shape mismatch')
    classifier.optimizer.load_state_dict(state)
    for wrapper in wrappers:
        wrapper.global_step = 4 * steps_per_epoch
        for group in wrapper.optimizer.param_groups:
            for parameter in group['params']:
                for value in wrapper.optimizer.state.get(parameter, {}).values():
                    assert value.shape == parameter.shape and torch.isfinite(value).all()
    momentum = classifier.optimizer.state[net[1].fc.weight]['momentum_buffer']
    assert torch.count_nonzero(momentum[10:]) == 0
    old_fc = next(
        key for key, parameter in zip(old_ids, parameters) if parameter is net[1].fc.weight)
    assert torch.equal(momentum[:10].cpu(), checkpoint['optimizer_cls']['state'][old_fc]['momentum_buffer'][:10])
    discriminator.grl.global_step = 8 * steps_per_epoch
    audit.update(known_momentum_sha256=tensor_hash(momentum[:10]),
        unknown_momentum_zero=True, optimizer_steps=4 * steps_per_epoch,
        grl_steps=8 * steps_per_epoch, head_shape=list(net[1].fc.weight.shape))
    reset_seed(1)
