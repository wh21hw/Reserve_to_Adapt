"""Source-only warmup and deterministic frozen features for structure diagnostics.

This is NOT the official RTA baseline: there is no target training, virtual loss,
domain adversary, or target-label model selection. Target labels are saved only
in a separate evaluation artifact, never returned by the feature loader.
"""
import argparse
import hashlib
import json
from pathlib import Path
import random
import sys
import time

import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import models, transforms


class Images(Dataset):
    def __init__(self, paths, transform, labels=None):
        self.paths, self.transform, self.labels = paths, transform, labels

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        with Image.open(self.paths[index]) as image:
            tensor = self.transform(image.convert('RGB'))
        return (tensor, self.labels[index]) if self.labels is not None else tensor


class SourceModel(nn.Module):
    def __init__(self, weights):
        super().__init__()
        backbone = models.resnet50(weights=None)
        # The verified historical torchvision checkpoint uses legacy tar serialization.
        # Permit that loader ONLY for the pinned, previously audited weight bytes.
        weight_hash = hashlib.sha256(Path(weights).read_bytes()).hexdigest()
        if weight_hash != '19c8e3572231adff6824a2da93fd67b5986919a2e65f8b6007eab4edee220097':
            raise ValueError('Unexpected pretrained weights; legacy loader disabled')
        backbone.load_state_dict(torch.load(weights, map_location='cpu', weights_only=False))
        backbone.fc = nn.Identity()
        self.backbone = backbone
        self.bottleneck = nn.Linear(2048, 256)
        self.classifier = nn.Sequential(nn.BatchNorm1d(256), nn.LeakyReLU(.2),
                                        nn.Linear(256, 12, bias=False))
        self.register_buffer('mean', torch.tensor([.485, .456, .406])[None, :, None, None])
        self.register_buffer('std', torch.tensor([.229, .224, .225])[None, :, None, None])

    def forward(self, image):
        features = nn.functional.normalize(self.bottleneck(self.backbone((image-self.mean)/self.std)), dim=1)
        return features, self.classifier(features)


def read_list(path, root):
    rows = [line.rsplit(' ', 1) for line in path.read_text().splitlines() if line.strip()]
    paths = [str(root/image) for image, _ in rows]
    if not all(Path(path).is_file() for path in paths):
        raise RuntimeError('Missing dataset image')
    return paths, np.array([int(label) for _, label in rows], dtype=np.int64)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source-list', required=True)
    parser.add_argument('--target-list', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--seed', type=int, default=1)
    args = parser.parse_args()
    out = Path(args.output)
    if out.exists():
        raise RuntimeError('Output already exists; refusing overwrite')
    if args.epochs < 1:
        raise ValueError('epochs must be positive')
    if not torch.cuda.is_available() or 'L4' not in torch.cuda.get_device_name():
        raise RuntimeError('This source warmup requires the verified L4 runtime')
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.set_num_threads(2)
    root = Path('/content/osda-datasets')
    source_path, target_path = Path(args.source_list), Path(args.target_list)
    source_images, source_labels = read_list(source_path, root)
    target_images, target_eval_labels = read_list(target_path, root)
    if set(source_labels) != set(range(10)):
        raise ValueError('Expected source classes 0-9')
    augment = transforms.Compose([transforms.Resize((256, 256)), transforms.RandomCrop(224),
                                  transforms.RandomHorizontalFlip(), transforms.ToTensor()])
    deterministic = transforms.Compose([transforms.Resize((256, 256)), transforms.CenterCrop(224),
                                        transforms.ToTensor()])
    generator = torch.Generator().manual_seed(args.seed)
    training = DataLoader(Images(source_images, augment, source_labels), batch_size=64,
                          shuffle=True, num_workers=0, drop_last=True, generator=generator)
    model = SourceModel(root/'resnet50-19c8e357.pth').cuda()
    optimizer = torch.optim.SGD([
        dict(params=model.backbone.parameters(), lr=5e-5, initial_lr=5e-5),
        dict(params=list(model.bottleneck.parameters()) + list(model.classifier.parameters()),
             lr=5e-4, initial_lr=5e-4)], momentum=.9, weight_decay=5e-4, nesterov=True)
    out.mkdir(parents=True)
    config = dict(stage='source-only warmup, then frozen deterministic features', seed=args.seed,
                  epochs=args.epochs, batch_size=64, source_samples=len(source_images),
                  target_samples=len(target_images), drop_last=True, workers=0,
                  losses='source CE over 12 outputs only; no target training',
                  selection='fixed final epoch; no target-label selection',
                  environment=dict(python=sys.version, torch=torch.__version__,
                                   torchvision=torchvision.__version__, cuda=torch.version.cuda,
                                   gpu=torch.cuda.get_device_name()),
                  source_list_sha256=hashlib.sha256(source_path.read_bytes()).hexdigest(),
                  target_list_sha256=hashlib.sha256(target_path.read_bytes()).hexdigest(),
                  pretrained_sha256=hashlib.sha256((root/'resnet50-19c8e357.pth').read_bytes()).hexdigest(),
                  script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  schedule='inverse decay gamma=10,power=.75,max_iter=10000')
    (out/'config.json').write_text(json.dumps(config, indent=2))
    print('SOURCE_WARMUP_CONFIG', json.dumps(config), flush=True)
    step = 0
    for epoch in range(args.epochs):
        model.train()
        losses, correct, samples = [], 0, 0
        started = time.time()
        for images, labels in training:
            images, labels = images.cuda(), labels.cuda()
            optimizer.zero_grad(set_to_none=True)
            _, logits = model(images)
            loss = nn.functional.cross_entropy(logits, labels)
            if not torch.isfinite(loss):
                raise RuntimeError('Nonfinite source loss')
            loss.backward()
            for group in optimizer.param_groups:
                group['lr'] = group['initial_lr'] * (1 + 10 * step / 10000) ** (-.75)
            optimizer.step()
            step += 1
            losses.append(loss.item())
            correct += int(logits.argmax(dim=1).eq(labels).sum())
            samples += len(labels)
        row = dict(epoch=epoch+1, steps=step, source_loss=float(np.mean(losses)),
                   source_train_accuracy=correct/samples, seconds=time.time()-started)
        with (out/'history.jsonl').open('a') as stream:
            stream.write(json.dumps(row)+'\n')
        print('SOURCE_EPOCH', json.dumps(row), flush=True)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    torch.save(dict(model=model.state_dict(), config=config, step=step), out/'source-final.pt')

    @torch.no_grad()
    def extract(paths):
        loader = DataLoader(Images(paths, deterministic), batch_size=64, shuffle=False, num_workers=0)
        features, logits = [], []
        for batch in loader:
            feature, prediction = model(batch.cuda())
            features.append(feature.cpu().numpy())
            logits.append(prediction.cpu().numpy())
        return np.concatenate(features), np.concatenate(logits)

    source_features, source_logits = extract(source_images)
    target_features, target_logits = extract(target_images)
    if not np.isfinite(source_features).all() or not np.isfinite(target_features).all():
        raise RuntimeError('Nonfinite frozen features')
    np.savez_compressed(out/'features.npz', source=source_features, source_labels=source_labels,
                        target=target_features, source_logits=source_logits, target_logits=target_logits)
    np.savez_compressed(out/'evaluation-only.npz', target_labels=target_eval_labels)
    manifest = dict(files={name: hashlib.sha256((out/name).read_bytes()).hexdigest()
                           for name in ['features.npz', 'evaluation-only.npz', 'source-final.pt']},
                    source_shape=list(source_features.shape), target_shape=list(target_features.shape),
                    frozen_source_accuracy=float((source_logits.argmax(axis=1) == source_labels).mean()),
                    steps=step, all_features_finite=True)
    (out/'manifest.json').write_text(json.dumps(manifest, indent=2))
    print('FROZEN_FEATURES_COMPLETE', json.dumps(manifest), flush=True)


if __name__ == '__main__':
    main()
