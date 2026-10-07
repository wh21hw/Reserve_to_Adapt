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


def infer_structure(source, source_labels, target, target_logits, known, calibration_mode='confidence', allow_zero=False, initial_candidates=None, merge_mode='none'):
    if calibration_mode not in ('confidence','structure_supported'):
        raise ValueError('Unknown target calibration support policy')
    anchors = torch.stack([source[source_labels == c].mean(0) for c in range(known)])
    src_dist = (source - anchors[source_labels]).square().sum(1)
    distances = (target[:, None] - anchors[None]).square().sum(-1)
    confidence, predicted = target_logits[:, :known].softmax(1).max(1)
    support = ((confidence >= .8) & (target_logits.argmax(1) < known)
               & (distances.argmin(1) == predicted))
    support_before = int(support.sum())
    preliminary_known = None
    if calibration_mode == 'structure_supported':
        # Same CURRENT coordinates, source-only initial scale. This is another
        # source-anchored geometric vote, not independent semantic evidence.
        initial_radius = max(float(np.quantile(src_dist.numpy(), .99)),1e-8)
        initial_variance = max(float(src_dist.mean())/source.shape[1],1e-8)
        preliminary = SourceAnchoredIMP(initial_radius,initial_variance,prior_strength=5.,
            steps=5,max_prototypes=100,known_centers_fixed=False).fit(target,anchors,birth_strategy='farthest')
        initial_assignment = preliminary['responsibilities'].argmax(1)
        preliminary_known = int((initial_assignment < known).sum())
        support &= initial_assignment == predicted
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
        max_prototypes=100, known_centers_fixed=False).fit(target, anchors, birth_strategy='farthest',initial_candidates=initial_candidates)
    centers = torch.cat([result['known_prototypes'], result['candidate_prototypes']])
    assignments = result['responsibilities'].argmax(1).numpy()
    occupied = np.unique(assignments[assignments >= known])
    if not len(occupied) and not allow_zero:
        raise RuntimeError('IMP inferred no occupied unknown structure; do not force K1')
    mapping = {int(old): known + j for j, old in enumerate(occupied)}
    assignments = np.asarray([mapping.get(int(a), int(a)) for a in assignments], dtype=np.int64)
    candidates = centers[torch.as_tensor(occupied)]
    if merge_mode not in ('none','objective'):raise ValueError('Unknown merge mode')
    merge_report=dict(enabled=False)
    if merge_mode=='objective':
        from dpmeans_merge import merge_candidates
        candidates,assignments,merge_report=merge_candidates(target,centers[:known],candidates,assignments,radius)
        occupied=np.arange(known,known+len(candidates))
        if not len(candidates) and not allow_zero:
            raise RuntimeError('Merge inferred zero occupied candidates; do not force K1')
    candidate_distance = (candidates[:, None] - anchors[None]).square().sum(-1)
    # Conservative geometric screening, not a semantic guarantee.
    reliable = (candidate_distance > torch.tensor(radii)[None]).all(1).numpy()
    report = dict(K=len(occupied), V=int(reliable.sum()), threshold=radius,
        variance=variance, class_radii=radii, target_support_per_known=counts,
        target_support_count=int(support.sum()), raw_candidates=result['candidate_count'],
        prior_strength=5., steps=5, target_labels_used=False,
        candidate_count_is_semantic_count=False, calibration_mode=calibration_mode,
        initial_candidate_count=0 if initial_candidates is None else len(initial_candidates),
        candidate_merge=merge_report,
        target_support_before_structure=support_before, preliminary_known_members=preliminary_known,
        _target_support_mask=support.numpy())
    return assignments, candidates, reliable, report


def current_member_centers(features, previous, known):
    """Re-embed old memberships in CURRENT coordinates; never reuse stale vectors."""
    if previous is None:return None
    previous=np.asarray(previous)
    if previous.shape!=(len(features),):raise ValueError('Previous membership row mismatch')
    ids=np.unique(previous[previous>=known])
    if not len(ids):return features.new_empty((0,features.shape[1]))
    return torch.stack([features[torch.from_numpy(previous==c)].mean(0) for c in ids])


def membership_stability(current,previous,known):
    """Known identities fixed; optimally match unknown IDs without target truth."""
    if previous is None:return dict(matched_assignment_change_fraction=None,unknown_status_change_fraction=None)
    current,previous=np.asarray(current),np.asarray(previous)
    if current.shape!=previous.shape:raise ValueError('Membership shape mismatch')
    old_ids=np.unique(previous[previous>=known]);new_ids=np.unique(current[current>=known])
    overlap=np.asarray([[np.sum((current==n)&(previous==o)) for o in old_ids] for n in new_ids],dtype=np.int64).reshape(len(new_ids),len(old_ids))
    rows,cols=linear_sum_assignment(-overlap)
    known_retained=int(np.sum((current<known)&(previous<known)&(current==previous)))
    unknown_retained=int(overlap[rows,cols].sum())
    both_unknown=int(np.sum((current>=known)&(previous>=known)))
    return dict(matched_assignment_change_fraction=1-(known_retained+unknown_retained)/len(current),
        unknown_status_change_fraction=float(((current>=known)!=(previous>=known)).mean()),
        matched_unknown_retention_given_both_unknown=None if not both_unknown else unknown_retained/both_unknown,
        stability_target_semantics_used=False)


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
    # Head size must not perturb later DataLoader/augmentation CPU RNG streams.
    # Init remains stochastic from the current state, but restores that stream.
    with torch.random.fork_rng(devices=[]):
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
        self.calibration_mode = 'confidence'
        self.teacher_space = 'bottleneck'
        self.label_scope = 'screened'
        self.known_veto = False
        self.known_scope = 'none'
        self.entropy_candidate_scale = 0.0  # Legacy hard veto when entropy scope is enabled.
        self.veto_eligibility = 'raw'
        self.initialization_mode = 'source_only'
        self.merge_mode = 'none'
        self.selection_mode = 'rta_only'
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
                features, logits, backbone = [], [], []
                for images in loader:
                    raw, feature, logit, _ = net(images.to(device))
                    features.append(feature.cpu()); logits.append(logit.cpu())
                    if self.teacher_space == 'backbone':
                        backbone.append(torch.nn.functional.normalize(raw,dim=1,eps=1e-8).cpu())
                arrays[split], arrays[split+'_logits'] = torch.cat(features), torch.cat(logits)
                if backbone:
                    arrays[split+'_backbone'] = torch.cat(backbone)
                if not torch.isfinite(arrays[split]).all() or not torch.isfinite(arrays[split+'_logits']).all():
                    raise RuntimeError('Nonfinite online feature extraction')
        finally:
            for module, mode in modes:
                module.training = mode
        if self.teacher_space not in ('bottleneck','backbone'):
            raise ValueError('Unknown IMP teacher feature space')
        key = '_backbone' if self.teacher_space == 'backbone' else ''
        if self.initialization_mode not in ('source_only','current_members'):
            raise ValueError('Unknown structure initialization mode')
        initial=current_member_centers(arrays['target'+key],self.assignments,self.args.shared_classes) if self.initialization_mode=='current_members' else None
        assignments, candidates, reliable, report = infer_structure(arrays['source'+key],
            self.source_labels, arrays['target'+key], arrays['target_logits'], self.args.shared_classes,
            self.calibration_mode,initial_candidates=initial,merge_mode=self.merge_mode)
        support_mask = report.pop('_target_support_mask')
        virtual = candidates[torch.from_numpy(reliable)]
        if self.teacher_space == 'backbone':
            # RTA virtual directions stay in its256-dimensional feature space.
            # No2048 prototype is installed in a256-dimensional head/virtual dot.
            _, virtual_candidates, virtual_reliable, virtual_report = infer_structure(arrays['source'],
                self.source_labels, arrays['target'], arrays['target_logits'], self.args.shared_classes,
                self.calibration_mode, allow_zero=True)
            virtual = virtual_candidates[torch.from_numpy(virtual_reliable)]
            report['virtual_threshold'] = virtual_report['threshold']
            report['virtual_candidate_count'] = virtual_report['K']
        report.update(V=len(virtual),teacher_space=self.teacher_space,
            teacher_dimension=int(arrays['source'+key].shape[1]),virtual_space='bottleneck256',
            label_reliable_candidates=int(reliable.sum()),label_scope=self.label_scope,
            label_eligible_candidates=len(reliable) if self.label_scope=='all_candidates' else int(reliable.sum()))
        change = None if self.assignments is None else float((assignments != self.assignments).mean())
        report.update(membership_stability(assignments,self.assignments,self.args.shared_classes),
            initialization_mode=self.initialization_mode)
        report['transport'] = transport_head(cls, optimizer.optimizer, self.args.shared_classes,
            assignments, None if reset_correspondence else self.assignments,
            arrays['target_logits'].softmax(1))
        self.assignments, self.reliable = assignments, reliable
        self.args.all_classes = cls.fc.out_features
        self.counts = np.zeros(cls.fc.out_features-self.args.shared_classes, dtype=np.int64)
        self.selected_by_sample = np.zeros(len(self.target_names),dtype=np.int32)
        self.overridden_by_sample = np.zeros(len(self.target_names),dtype=np.int32)
        self.teacher_added_by_sample = np.zeros(len(self.target_names),dtype=np.int32)
        self.known_original_by_sample = np.zeros(len(self.target_names),dtype=np.float32)
        self.known_effective_by_sample = np.zeros(len(self.target_names),dtype=np.float32)
        self.entropy_effective_by_sample = np.zeros(len(self.target_names),dtype=np.float32)
        self.alignment_effective_by_sample = np.zeros(len(self.target_names),dtype=np.float32)
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
            reliable=reliable, target_paths=np.asarray(self.target_names), target_support_mask=support_mask,
            teacher_space=np.asarray(self.teacher_space),label_scope=np.asarray(self.label_scope))
        np.savez_compressed(path/('structure-membership-%03d.npz'%(epoch+1)),
            assignments=assignments,reliable=reliable,target_paths=np.asarray(self.target_names))
        config_path = path/'config.json'
        config = json.loads(config_path.read_text())
        config.update(all_classes=self.args.all_classes, design='online-imp-structure-v1',
            use_structure_labels=self.use_labels, refresh='Every epoch, current complete eval features')
        config_path.write_text(json.dumps(config, indent=2))
        print('ONLINE_IMP_REFRESH', json.dumps(report), flush=True)
        return virtual.to(device)

    def select_unknown(self,indices,original):
        """Union reliable teacher members with original selector; preserve its order."""
        original=original.view(-1)
        indices=np.asarray(indices)
        self.last_teacher_added=np.empty(0,dtype=np.int64)
        if self.selection_mode=='rta_only':return original
        if self.selection_mode!='reliable_union':raise ValueError('Unknown CE selection mode')
        ids=self.assignments[indices];known=self.args.shared_classes
        eligible=ids>=known
        eligible[eligible]&=self.reliable[ids[eligible]-known]
        eligible[original.detach().cpu().numpy()]=False
        extra=np.flatnonzero(eligible)
        self.last_teacher_added=indices[extra]
        return torch.cat([original,torch.as_tensor(extra,device=original.device,dtype=original.dtype)])

    def labels(self, indices, fallback):
        ids = self.assignments[np.asarray(indices)]
        known = self.args.shared_classes
        mask = ids >= known
        scope=getattr(self,'label_scope','screened')
        if scope=='screened':
            mask[mask] &= self.reliable[ids[mask]-known]
        elif scope!='all_candidates':
            raise ValueError('Unknown structure label coverage policy')
        labels = fallback.detach().clone()
        if self.use_labels:
            labels[torch.as_tensor(mask, device=labels.device)] = torch.as_tensor(ids[mask], device=labels.device)
        self.selected += len(ids)
        self.overridden += int(mask.sum()) if self.use_labels else 0
        self.counts += np.bincount(labels.cpu().numpy()-known, minlength=len(self.counts))
        if hasattr(self,'selected_by_sample'):
            np.add.at(self.selected_by_sample,np.asarray(indices),1)
            if hasattr(self,'teacher_added_by_sample'):
                added=np.isin(np.asarray(indices),getattr(self,'last_teacher_added',[]))
                np.add.at(self.teacher_added_by_sample,np.asarray(indices)[added],1)
            if self.use_labels:
                np.add.at(self.overridden_by_sample,np.asarray(indices)[mask],1)
        return labels

    def known_weights(self, indices, original):
        """Veto only known entropy/target alignment, never modify the r selector."""
        indices=np.asarray(indices)
        known=self.assignments[indices] < self.args.shared_classes
        original=original.detach()
        effective=original*torch.as_tensor(known,device=original.device,dtype=original.dtype) if self.known_veto else original
        np.add.at(self.known_original_by_sample,indices,original.cpu().numpy())
        np.add.at(self.known_effective_by_sample,indices,effective.cpu().numpy())
        return effective

    def objective_weights(self, indices, original):
        indices=np.asarray(indices);original=original.detach()
        scope=self.known_scope
        if scope not in ('none','both','entropy','alignment'):raise ValueError('Unknown known-objective scope')
        ids=self.assignments[indices]
        blocked=ids>=self.args.shared_classes
        if self.veto_eligibility=='screened':
            blocked[blocked]&=self.reliable[ids[blocked]-self.args.shared_classes]
        elif self.veto_eligibility!='raw':raise ValueError('Unknown veto eligibility')
        eligible=torch.as_tensor(~blocked,
            device=original.device,dtype=original.dtype)
        masked=original*eligible
        scale=float(self.entropy_candidate_scale)
        if not 0.0<=scale<=1.0:raise ValueError('Entropy candidate scale must lie in [0,1]')
        entropy_masked=original*(eligible+(1-eligible)*scale)
        entropy=entropy_masked if scope in ('both','entropy') else original
        alignment=masked if scope in ('both','alignment') else original
        effective=entropy if scope=='entropy' else masked if scope!='none' else original
        np.add.at(self.known_original_by_sample,indices,original.cpu().numpy())
        np.add.at(self.known_effective_by_sample,indices,effective.cpu().numpy())
        np.add.at(self.entropy_effective_by_sample,indices,entropy.cpu().numpy())
        np.add.at(self.alignment_effective_by_sample,indices,alignment.cpu().numpy())
        return entropy,alignment

    def finish_epoch(self, epoch):
        row = dict(epoch=epoch+1, K=len(self.counts), pseudo_slot_counts=self.counts.tolist(),
            selected=self.selected, structure_labels_used=self.overridden,
            known_weight_original=float(self.known_original_by_sample.sum()),
            known_weight_effective=float(self.known_effective_by_sample.sum()),
            known_veto_enabled=self.known_scope!='none',known_objective_scope=self.known_scope,
            entropy_candidate_scale=self.entropy_candidate_scale)
        row.update(unknown_selection_mode=self.selection_mode,teacher_added_actual=int(self.teacher_added_by_sample.sum()),
            original_selector_actual=self.selected-int(self.teacher_added_by_sample.sum()))
        with (Path(self.args.log_dir)/'online-label-history.jsonl').open('a') as stream:
            stream.write(json.dumps(row)+'\n')
        print('ONLINE_IMP_LABEL_USAGE', json.dumps(row), flush=True)
        np.savez_compressed(Path(self.args.log_dir)/('sample-exposure-%03d.npz'%(epoch+1)),
            selected=self.selected_by_sample,overridden=self.overridden_by_sample,
            teacher_added=self.teacher_added_by_sample,
            target_paths=np.asarray(self.target_names),unknown_ce_active=np.asarray(epoch>3),
            known_weight_original=self.known_original_by_sample,known_weight_effective=self.known_effective_by_sample,
            entropy_weight_effective=self.entropy_effective_by_sample,alignment_weight_effective=self.alignment_effective_by_sample)
