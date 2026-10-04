"""Fusion-v1: initial IMP capacity plus refreshed IMP virtual directions.

No target labels, new losses, or IMP-based classifier weight initialization.
K is fixed after the initial estimate; V may change during RTA training.
"""
import json
from pathlib import Path

import numpy as np
import torch

from alternating_konly import estimate_current_K, _FrozenImages
from torch.utils.data import DataLoader


@torch.no_grad()
def initial_structure(features_path, known_count):
    with np.load(features_path) as data:
        source = torch.from_numpy(data['source']).float()
        target = torch.from_numpy(data['target']).float()
        labels = torch.from_numpy(data['source_labels']).long()
    if set(labels.tolist()) != set(range(known_count)):
        raise ValueError('Source labels must cover all contiguous known classes')
    result, settings = estimate_current_K(source, labels, target, known_count)
    if result['candidate_count'] < 1:
        raise RuntimeError('No unknown capacity inferred; do not silently force K=1')
    return result, settings


def save_structure(result, settings, log_dir, completed_epochs, fixed_K):
    report = dict(stage='fusion-v1', completed_epochs=completed_epochs,
                  K=int(fixed_K), V=result['candidate_count'],
                  Q=result['known_count'] + result['candidate_count'],
                  known_centers_fixed=False, history=result['history'],
                  head_resized=False, prototype_head_initialization=False,
                  **settings)
    with (Path(log_dir) / 'fusion-history.jsonl').open('a') as stream:
        stream.write(json.dumps(report, allow_nan=False) + '\n')
    print('FUSION_STRUCTURE', json.dumps(report, allow_nan=False), flush=True)


@torch.no_grad()
def refresh_virtual(net, args, completed_epochs):
    """Full deterministic eval extraction in the CURRENT feature coordinates."""
    rows = [r.rsplit(None, 1) for r in Path(args.source).read_text().splitlines() if r.strip()]
    labels = torch.tensor([int(r[1]) for r in rows], dtype=torch.long)
    # Only target filenames are retained; semantic labels never enter inference.
    target_names = [r.rsplit(None, 1)[0] for r in Path(args.target).read_text().splitlines() if r.strip()]
    source_names = [r[0] for r in rows]
    device = next(net.parameters()).device
    modes = [(module, module.training) for module in net.modules()]
    # Local loader generators do not perturb the RTA sampling RNG.
    net.eval()
    try:
        def extract(names, offset):
            loader = DataLoader(_FrozenImages(names, args.data_dir), batch_size=args.batch_size,
                                shuffle=False, num_workers=4, pin_memory=True,
                                generator=torch.Generator().manual_seed(completed_epochs + offset))
            return torch.cat([net(images.to(device))[1].cpu() for images in loader])
        source, target = extract(source_names, 100), extract(target_names, 200)
    finally:
        for module, training in modes:
            module.training = training
    result, settings = estimate_current_K(source, labels, target, args.shared_classes)
    save_structure(result, settings, args.log_dir, completed_epochs,
                   args.all_classes - args.shared_classes)
    # Empty V is valid; virtual CE then has no extra virtual directions.
    return result['candidate_prototypes'].to(device)
