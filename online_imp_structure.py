"""Online IMP structures, transported head state and optional structure labels.

Empirical mixture inference, not a DP posterior. Target semantics never enter.
"""
import json
import time
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment
import torch
from torch import nn
from torch.utils.data import DataLoader

from alternating_konly import _FrozenImages
from source_anchored_imp import SourceAnchoredIMP


def infer_structure(source, source_labels, target, target_logits, known):
    anchors = torch.stack([source[source_labels == c].mean(0) for c in range(known)])
    src_dist = (source - anchors[source_labels]).square().sum(1)
    distances = (target[:, None] - anchors[None]).square().sum(-1)
    confidence, predicted = target_logits[:, :known].softmax(1).max(1)
    support = ((confidence >= .8) & (target_logits.argmax(1) < known)
               & (distances.argmin(1) == predicted))
    radii, counts, residuals = [], [], [src_dist]
    for c in range(known):
        src_radius = max(float(np.quantile(src_dist[source_labels == c].numpy(), .99)), 1e-8)
        selected = distances[support & (predicted == c), c]
        counts.append(len(selected))
        if len(selected) >= 5:
            radius = .5 * (src_radius + float(np.quantile(selected.numpy(), .99)))
            residuals.append(selected)
        else:
            radius = src_radius
        radii.append(max(radius, 1e-8))
    # One squared-distance threshold, with class-dependent calibration evidence.
    radius = float(np.average(radii, weights=np.bincount(source_labels.numpy(), minlength=known)))
    variance = max(float(torch.cat(residuals).mean()) / source.shape[1], 1e-8)
    result = SourceAnchoredIMP(radius, variance, prior_strength=5., steps=5,
        max_prototypes=100, known_centers_fixed=False).fit(target, anchors, birth_strategy='farthest')
    centers = torch.cat([result['known_prototypes'], result['candidate_prototypes']])
    assignments = result['responsibilities'].argmax(1).numpy()
    occupied = np.unique(assignments[assignments >= known])
    if not len(occupied):
        raise RuntimeError('IMP inferred no occupied unknown structure; do not force K1')
    mapping = {int(old): known + j for j, old in enumerate(occupied)}
    assignments = np.asarray([mapping.get(int(a), int(a)) for a in assignments], dtype=np.int64)
    candidates = centers[torch.as_tensor(occupied)]
    candidate_distance = (candidates[:, None] - anchors[None]).square().sum(-1)
    # Conservative geometric screening, not a semantic guarantee.
    reliable = (candidate_distance > torch.tensor(radii)[None]).all(1).numpy()
    report = dict(K=len(occupied), V=int(reliable.sum()), threshold=radius,
        variance=variance, class_radii=radii, target_support_per_known=counts,
        target_support_count=int(support.sum()), raw_candidates=result['candidate_count'],
        prior_strength=5., steps=5, target_labels_used=False,
        candidate_count_is_semantic_count=False)
    return assignments, candidates, reliable, report


@torch.no_grad()
def transport_head(cls, optimizer, known, assignments, previous, old_probabilities):
    old = cls.fc
    new_k = int(assignments.max()) + 1 - known
    old_k = old.out_features - known
    membership = np.eye(new_k)[np.maximum(assignments-known, 0)]
    membership[assignments < known] = 0
    if previous is None:
        old_membership = old_probabilities[:, known:].numpy()
    else:
        old_membership = np.eye(old_k)[np.maximum(previous-known, 0)]
        old_membership[previous < known] = 0
    overlap = membership.T @ old_membership
    denom = np.linalg.norm(membership, axis=0)[:, None] * np.linalg.norm(old_membership, axis=0)[None]
    similarity = overlap / np.maximum(denom, 1e-12)
    rows, cols = linear_sum_assignment(-similarity)
    pairs = [(known+int(r), known+int(c)) for r, c in zip(rows, cols) if overlap[r, c] > 0]
    replacement = nn.Linear(old.in_features, known+new_k, bias=False).to(old.weight)
    replacement.weight[:known].copy_(old.weight[:known])
    scale = old.weight.norm(dim=1).mean()
    replacement.weight[known:].mul_(scale / replacement.weight[known:].norm(dim=1, keepdim=True).clamp_min(1e-8))
    for dest, origin in pairs:
        replacement.weight[dest].copy_(old.weight[origin])
    found = 0
    for group in optimizer.param_groups:
        for i, param in enumerate(group['params']):
            if param is old.weight:
                group['params'][i] = replacement.weight
                found += 1
    if found != 1:
        raise RuntimeError('Expected exactly one live classifier SGD parameter')
    state = optimizer.state.pop(old.weight, {})
    updated = {}
    for key, value in state.items():
        if torch.is_tensor(value) and value.shape == old.weight.shape:
            buffer = torch.zeros_like(replacement.weight)
            buffer[:known].copy_(value[:known])
            for dest, origin in pairs:
                buffer[dest].copy_(value[origin])
            updated[key] = buffer
        else:
            updated[key] = value
    optimizer.state[replacement.weight] = updated
    cls.fc = replacement
    cls.main[1][2] = replacement
    return dict(old_K=old_k, new_K=new_k, row_mapping=pairs,
        new_random_rows=new_k-len(pairs), known_rows_preserved=True,
        matched_momentum_preserved=True, scheduler_reset=False)


class OnlineStructure:
    def __init__(self, args, use_labels):
        self.args, self.use_labels = args, use_labels
        source_rows = [r.rsplit(None, 1) for r in Path(args.source).read_text().splitlines() if r.strip()]
        self.source_names = [r[0] for r in source_rows]
        self.source_labels = torch.tensor([int(r[1]) for r in source_rows])
        self.target_names = [r.rsplit(None, 1)[0] for r in Path(args.target).read_text().splitlines() if r.strip()]
        self.assignments = None
        self.reliable = None
        self.counts = None

    @torch.no_grad()
    def refresh(self, net, cls, optimizer, epoch, reset_correspondence=False):
        started = time.monotonic()
        modes = [(module, module.training) for module in net.modules()]
        device = cls.fc.weight.device
        net.eval()
        arrays = {}
        try:
            for offset, (split, names) in enumerate((('source', self.source_names), ('target', self.target_names))):
                loader = DataLoader(_FrozenImages(names, self.args.data_dir), batch_size=64,
                    shuffle=False, num_workers=4, pin_memory=True,
                    generator=torch.Generator().manual_seed(7102026+offset))
                features, logits = [], []
                for images in loader:
                    _, feature, logit, _ = net(images.to(device))
                    features.append(feature.cpu()); logits.append(logit.cpu())
                arrays[split], arrays[split+'_logits'] = torch.cat(features), torch.cat(logits)
                if not torch.isfinite(arrays[split]).all() or not torch.isfinite(arrays[split+'_logits']).all():
                    raise RuntimeError('Nonfinite online feature extraction')
        finally:
            for module, mode in modes:
                module.training = mode
        assignments, candidates, reliable, report = infer_structure(arrays['source'],
            self.source_labels, arrays['target'], arrays['target_logits'], self.args.shared_classes)
        change = None if self.assignments is None else float((assignments != self.assignments).mean())
        report['transport'] = transport_head(cls, optimizer.optimizer, self.args.shared_classes,
            assignments, None if reset_correspondence else self.assignments,
            arrays['target_logits'].softmax(1))
        self.assignments, self.reliable = assignments, reliable
        self.args.all_classes = cls.fc.out_features
        self.counts = np.zeros(cls.fc.out_features-self.args.shared_classes, dtype=np.int64)
        self.overridden = self.selected = 0
        report.update(epoch=epoch+1, refresh_seconds=time.monotonic()-started,
            use_structure_labels=self.use_labels, raw_assignment_change_fraction=change,
            candidate_member_counts=np.bincount(assignments, minlength=self.args.all_classes)[self.args.shared_classes:].tolist(),
            target_known_structure_members=int((assignments < self.args.shared_classes).sum()),
            reliable_candidate_indices=np.flatnonzero(reliable).tolist())
        path = Path(self.args.log_dir)
        with (path/'online-structure-history.jsonl').open('a') as stream:
            stream.write(json.dumps(report, allow_nan=False)+'\n')
        np.savez_compressed(path/'current-structure.npz', assignments=assignments,
            reliable=reliable, target_paths=np.asarray(self.target_names))
        config_path = path/'config.json'
        config = json.loads(config_path.read_text())
        config.update(all_classes=self.args.all_classes, design='online-imp-structure-v1',
            use_structure_labels=self.use_labels, refresh='Every epoch, current complete eval features')
        config_path.write_text(json.dumps(config, indent=2))
        print('ONLINE_IMP_REFRESH', json.dumps(report), flush=True)
        return candidates[torch.from_numpy(reliable)].to(device)

    def labels(self, indices, fallback):
        ids = self.assignments[np.asarray(indices)]
        known = self.args.shared_classes
        mask = ids >= known
        mask[mask] &= self.reliable[ids[mask]-known]
        labels = fallback.detach().clone()
        if self.use_labels:
            labels[torch.as_tensor(mask, device=labels.device)] = torch.as_tensor(ids[mask], device=labels.device)
        self.selected += len(ids)
        self.overridden += int(mask.sum()) if self.use_labels else 0
        self.counts += np.bincount(labels.cpu().numpy()-known, minlength=len(self.counts))
        return labels

    def finish_epoch(self, epoch):
        row = dict(epoch=epoch+1, K=len(self.counts), pseudo_slot_counts=self.counts.tolist(),
            selected=self.selected, structure_labels_used=self.overridden)
        with (Path(self.args.log_dir)/'online-label-history.jsonl').open('a') as stream:
            stream.write(json.dumps(row)+'\n')
        print('ONLINE_IMP_LABEL_USAGE', json.dumps(row), flush=True)
