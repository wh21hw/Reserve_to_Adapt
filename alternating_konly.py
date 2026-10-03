"""K refresh and state-preserving head resize; no new training objective.

Refreshes use current source/target representations, never old feature anchors.
Unknown row correspondence uses current predictions vs new responsibilities,
not geometry between a pre-BN prototype and a post-BN classifier weight.
"""
import numpy as np
import json
from pathlib import Path
from PIL import Image
from scipy.optimize import linear_sum_assignment
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from source_anchored_imp import SourceAnchoredIMP


@torch.no_grad()
def estimate_current_K(source, source_labels, target, known_count):
    if source.shape[1] != target.shape[1]:
        raise ValueError('Source/target must use the same current representation')
    centers = torch.stack([source[source_labels == c].mean(0) for c in range(known_count)])
    residual = source - centers[source_labels]
    threshold = max(float(np.quantile(residual.square().sum(1).cpu().numpy(), .99)), 1e-8)
    variance = max(float(residual.square().mean()), 1e-8)
    model = SourceAnchoredIMP(threshold, variance, prior_strength=5., steps=5,
                              max_prototypes=100, known_centers_fixed=False)
    result = model.fit(target, centers, known_compatibility=None, birth_strategy='farthest')
    return result, dict(threshold=threshold, variance=variance, source_quantile=.99,
                        prior_strength=5., steps=5, target_labels_used=False)


@torch.no_grad()
def resize_unknown_head(cls, optimizer, known_count, responsibilities, old_probabilities):
    old_fc = cls.fc
    old_K = old_fc.out_features - known_count
    new_K = responsibilities.shape[1] - known_count
    if new_K <= 0:
        raise RuntimeError('K=0 inferred; save estimate and stop rather than force K=1')
    if old_fc.bias is not None or cls.main[1][2] is not old_fc:
        raise ValueError('Expected original RTA bias-free aliased head')
    if old_probabilities.shape != (len(responsibilities), known_count + old_K):
        raise ValueError('Prediction rows/columns do not match the current head')
    if new_K == old_K:
        # K is the ONLY cluster output used. Do not gratuitously permute weights
        # or reset SGD when the inferred capacity has not changed.
        return dict(old_K=old_K, new_K=new_K, changed=False, row_mapping=[])
    new_membership = responsibilities[:, known_count:].double().cpu().numpy()
    old_membership = old_probabilities[:, known_count:].double().cpu().numpy()
    overlap = new_membership.T @ old_membership
    denominator = np.linalg.norm(new_membership, axis=0)[:, None] * np.linalg.norm(old_membership, axis=0)[None, :]
    similarity = overlap / np.maximum(denominator, 1e-12)
    new_rows, old_rows = linear_sum_assignment(-similarity)
    new_fc = nn.Linear(old_fc.in_features, known_count + new_K, bias=False).to(old_fc.weight)
    new_fc.weight[:known_count].copy_(old_fc.weight[:known_count])
    # New slots use random directions, with the original RTA norm-matching idea;
    # IMP prototype directions are not installed in the classifier.
    scale = old_fc.weight.norm(dim=1).mean()
    new_fc.weight[known_count:].mul_(scale / new_fc.weight[known_count:].norm(dim=1, keepdim=True).clamp_min(1e-8))
    row_mapping = [(known_count + int(n), known_count + int(o)) for n, o in zip(new_rows, old_rows)]
    for new_row, old_row in row_mapping:
        new_fc.weight[new_row].copy_(old_fc.weight[old_row])
    replaced = 0
    for group in optimizer.param_groups:
        for index, parameter in enumerate(group['params']):
            if parameter is old_fc.weight:
                group['params'][index] = new_fc.weight
                replaced += 1
    if replaced != 1:
        raise RuntimeError('Expected one head parameter in SGD')
    old_state = optimizer.state.pop(old_fc.weight, {})
    new_state = {}
    for name, value in old_state.items():
        if torch.is_tensor(value) and value.shape == old_fc.weight.shape:
            updated = torch.zeros_like(new_fc.weight)
            updated[:known_count].copy_(value[:known_count])
            for new_row, old_row in row_mapping:
                updated[new_row].copy_(value[old_row])
            new_state[name] = updated
        elif torch.is_tensor(value) and value.ndim != 0:
            raise ValueError('Unexpected SGD head state: ' + name)
        else:
            new_state[name] = value
    optimizer.state[new_fc.weight] = new_state
    cls.fc = new_fc
    cls.main[1][2] = new_fc
    return dict(old_K=old_K, new_K=new_K, changed=True, row_mapping=row_mapping,
                retained_unknown_rows=len(row_mapping), new_random_rows=max(0, new_K-old_K),
                removed_unknown_rows=max(0, old_K-new_K), correspondence='Unlabeled prediction/responsibility overlap',
                known_rows_preserved=True, matched_SGD_momentum_preserved=True,
                optimizer_schedule_reset=False, prototype_direction_initialization=False)


class _FrozenImages(Dataset):
    def __init__(self, names, root):
        self.names, self.root = names, Path(root)
        self.transform = transforms.Compose([transforms.Resize((256, 256)),
                          transforms.CenterCrop(224), transforms.ToTensor()])

    def __len__(self):
        return len(self.names)

    def __getitem__(self, index):
        with Image.open(self.root/self.names[index]) as image:
            return self.transform(image.convert('RGB'))


@torch.no_grad()
def refresh_current_network(net, cls, optimizer_wrapper, args, completed_epochs):
    """Called at an epoch boundary; source anchors are always recomputed."""
    output = Path(args.log_dir)/('refresh-%03d' % completed_epochs)
    output.mkdir(exist_ok=False)
    source_rows = [row.rsplit(None, 1) for row in Path(args.source).read_text().splitlines() if row.strip()]
    source_names = [row[0] for row in source_rows]
    labels = torch.tensor([int(row[1]) for row in source_rows], dtype=torch.long)
    if set(labels.tolist()) != set(range(args.shared_classes)):
        raise ValueError('This first alternating entry supports contiguous known source IDs')
    # Discard target semantic labels; only filenames enter feature extraction.
    target_names = [row.rsplit(None, 1)[0] for row in Path(args.target).read_text().splitlines() if row.strip()]
    device = cls.fc.weight.device
    was_training = net.training
    net.eval()
    try:
        def extract(names, seed_offset):
            features, probabilities = [], []
            generator = torch.Generator().manual_seed(3 + completed_epochs + seed_offset)
            loader = DataLoader(_FrozenImages(names, args.data_dir), batch_size=64,
                                shuffle=False, num_workers=4, pin_memory=True, generator=generator)
            for images in loader:
                outputs = net(images.to(device))
                features.append(outputs[1].cpu())
                probabilities.append(outputs[-1].cpu())
            return torch.cat(features), torch.cat(probabilities)
        source, _ = extract(source_names, 0)
        target, old_probabilities = extract(target_names, 1)
    finally:
        net.train(was_training)
    if not all(torch.isfinite(x).all() for x in (source, target, old_probabilities)):
        raise RuntimeError('Nonfinite current features/probabilities')
    result, settings = estimate_current_K(source, labels, target, args.shared_classes)
    report = dict(completed_epochs=completed_epochs, K=result['candidate_count'],
                  known_classes=args.shared_classes, history=result['history'],
                  effective_counts=result['effective_counts'].tolist(), **settings)
    (output/'estimate.json').write_text(json.dumps(report, indent=2, allow_nan=False))
    np.savez_compressed(output/'prototypes.npz', known=result['known_prototypes'].numpy(),
                        unknown=result['candidate_prototypes'].numpy(),
                        responsibilities=result['responsibilities'].numpy())
    head = resize_unknown_head(cls, optimizer_wrapper.optimizer, args.shared_classes,
                               result['responsibilities'], old_probabilities)
    args.all_classes = args.shared_classes + report['K']
    report['head'] = head
    report['source_and_target_recomputed_in_current_coordinates'] = True
    report['source_labels_used'] = True
    report['additional_training_objective'] = False
    (output/'estimate.json').write_text(json.dumps(report, indent=2, allow_nan=False))
    with (Path(args.log_dir)/'K-history.jsonl').open('a') as stream:
        stream.write(json.dumps(report, allow_nan=False)+'\n')
    config_path = Path(args.log_dir)/'config.json'
    config = json.loads(config_path.read_text())
    config.setdefault('initial_all_classes', config['all_classes'])
    config['all_classes'] = args.all_classes
    config['latest_K_refresh_completed_epochs'] = completed_epochs
    config_path.write_text(json.dumps(config, indent=2, allow_nan=False))
    print('ALTERNATING_K_REFRESH', json.dumps(report, allow_nan=False), flush=True)
    return report
