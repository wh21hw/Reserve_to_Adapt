"""Shared, K-independent source supervision before IMP and fixed-K controls.

Uses the RTA network architecture, C outputs only. Target names are read for
frozen feature extraction; target labels never enter training or the artifact.
"""
import argparse
import json
import random
from pathlib import Path
import sys
import time

import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset
import torchvision
from torchvision import transforms


class Images(Dataset):
    def __init__(self, names, root, transform, labels=None):
        self.names, self.root, self.transform, self.labels = names, root, transform, labels

    def __len__(self):
        return len(self.names)

    def __getitem__(self, index):
        with Image.open(self.root / self.names[index]) as image:
            value = self.transform(image.convert('RGB'))
        return value if self.labels is None else (value, int(self.labels[index]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--code-root', required=True)
    parser.add_argument('--source', required=True)
    parser.add_argument('--target', required=True)
    parser.add_argument('--data-root', required=True)
    parser.add_argument('--weights', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--epochs', type=int, default=3)
    args = parser.parse_args()
    root, output = Path(args.data_root), Path(args.output)
    if output.exists():
        raise RuntimeError('Do not overwrite a previous experiment')
    if args.epochs < 1:
        raise ValueError('epochs must be positive')
    if not torch.cuda.is_available() or 'L4' not in torch.cuda.get_device_name():
        raise RuntimeError('This run requires the existing L4 GPU')
    sys.path.insert(0, args.code_root)
    from networks import ResNetFc, CLS
    from utilities import OptimWithSheduler, inverseDecaySheduler
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.set_num_threads(2)
    source_rows = [r.rsplit(None, 1) for r in Path(args.source).read_text().splitlines() if r.strip()]
    source_names = [r[0] for r in source_rows]
    original_labels = np.array([int(r[1]) for r in source_rows], dtype=np.int64)
    known_ids = sorted(set(original_labels.tolist()))
    mapping = {label: index for index, label in enumerate(known_ids)}
    labels = np.array([mapping[int(label)] for label in original_labels], dtype=np.int64)
    target_names = [r.rsplit(None, 1)[0] for r in Path(args.target).read_text().splitlines() if r.strip()]
    C = len(known_ids)
    augment = transforms.Compose([transforms.Resize((256, 256)), transforms.RandomCrop(224),
                                  transforms.RandomHorizontalFlip(), transforms.ToTensor()])
    fixed = transforms.Compose([transforms.Resize((256, 256)), transforms.CenterCrop(224),
                                transforms.ToTensor()])
    loader = DataLoader(Images(source_names, root, augment, labels), batch_size=64,
                        shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
    net = torch.nn.Sequential(ResNetFc(model_path=args.weights), CLS(2048, C)).cuda()
    schedule = lambda step, lr: inverseDecaySheduler(step, lr, gamma=10, power=.75, max_iter=10000)
    optimizers = [OptimWithSheduler(torch.optim.SGD(module.parameters(), lr=lr,
                  momentum=.9, nesterov=True, weight_decay=5e-4), schedule)
                  for module, lr in zip(net, (5e-5, 5e-4))]
    config = dict(stage='shared source-only C-output prior, independent of unknown K',
                  seed=args.seed, epochs=args.epochs, known_classes=C, original_known_ids=known_ids,
                  source_samples=len(source_names), target_samples=len(target_names),
                  batch_size=64, workers=4, loss='source cross entropy only',
                  target_labels_used=False, network_code=args.code_root,
                  initialization_policy='Both fixed-K and estimated-K RTA arms use this same model',
                  environment=dict(python=sys.version, torch=torch.__version__,
                      torchvision=torchvision.__version__, cuda=torch.version.cuda,
                      gpu=torch.cuda.get_device_name()))
    output.mkdir(parents=True)
    (output/'config.json').write_text(json.dumps(config, indent=2))
    print('SOURCE_PRIOR_START', json.dumps(config), flush=True)
    history = []
    for epoch in range(args.epochs):
        net.train()
        losses, correct, total, start = [], 0, 0, time.time()
        for images, truth in loader:
            images, truth = images.cuda(), truth.cuda()
            for optimizer in optimizers:
                optimizer.zero_grad()
            logits = net(images)[2]
            loss = torch.nn.functional.cross_entropy(logits, truth)
            if not torch.isfinite(loss):
                raise RuntimeError('Nonfinite source loss; no update')
            loss.backward()
            if not all(p.grad is None or torch.isfinite(p.grad).all() for p in net.parameters()):
                raise RuntimeError('Nonfinite source gradients; no update')
            for optimizer in optimizers:
                optimizer.step()
            losses.append(float(loss.detach()))
            correct += int((logits.argmax(1) == truth).sum())
            total += len(truth)
        row = dict(epoch=epoch+1, loss=float(np.mean(losses)), accuracy=correct/total,
                   optimizer_steps=[o.global_step for o in optimizers], seconds=time.time()-start)
        history.append(row)
        with (output/'history.jsonl').open('a') as stream:
            stream.write(json.dumps(row)+'\n')
        print('SOURCE_PRIOR_EPOCH', json.dumps(row), flush=True)
    torch.save(dict(model=net.state_dict(), config=config, history=history), output/'source-final.pt')
    net.eval()

    @torch.no_grad()
    def extract(names):
        features = []
        data = DataLoader(Images(names, root, fixed), batch_size=64, shuffle=False,
                          num_workers=4, pin_memory=True)
        for images in data:
            features.append(net(images.cuda())[1].cpu().numpy())
        values = np.concatenate(features)
        if not np.isfinite(values).all():
            raise RuntimeError('Nonfinite frozen features')
        return values

    source, target = extract(source_names), extract(target_names)
    centers = np.stack([source[labels == c].mean(0) for c in range(C)])
    residual = source - centers[labels]
    np.savez_compressed(output/'features.npz', source=source, target=target,
                        source_labels=labels, source_centers=centers)
    summary = dict(complete=True, known_classes=C, source_shape=list(source.shape),
                   target_shape=list(target.shape), epochs=args.epochs,
                   source_variance=max(float(np.mean(residual**2)), 1e-8),
                   source_distance_quantile99=float(np.quantile(np.sum(residual**2, axis=1), .99)),
                   target_labels_in_features=False, K_estimated=False)
    (output/'summary.json').write_text(json.dumps(summary, indent=2))
    print('SOURCE_PRIOR_COMPLETE', json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
