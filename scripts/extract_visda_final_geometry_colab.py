"""One failure-mechanism cache at the predeclared fixed-K final epoch.

Not checkpoint selection or repeated accuracy evaluation. Target labels are
discarded at the list boundary; post-hoc semantic analysis stays separate.
Run with /content/rta-py38/bin/python after the estimated arm finishes.
"""
import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from torchvision import transforms


ROOT = Path('/content/imp-runs/visda-frozenbn-capacity-10e-v1')
DATA = Path('/content/osda-visda-syn2real-v1')
CODE = Path('/content/rta-legacy-l4-bridge-v1')
OUTPUT = ROOT / 'fixed2-final10-geometry-v1'


def read_names(path):
    return [row.rsplit(None, 1)[0] for row in path.read_text().splitlines() if row.strip()]


class Images(Dataset):
    def __init__(self, names):
        self.names = names
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)), transforms.CenterCrop(224), transforms.ToTensor()])

    def __len__(self):
        return len(self.names)

    def __getitem__(self, index):
        with Image.open(DATA / self.names[index]) as image:
            return self.transform(image.convert('RGB'))


def require_finished_arms():
    for arm in ('fixed2', 'estimated'):
        history = ROOT / arm / 'visda-synthetic2real_seed1/history.jsonl'
        rows = [json.loads(line) for line in history.read_text().splitlines()]
        if [row['epoch'] for row in rows] != list(range(1, 11)):
            raise RuntimeError('Wait for both declared ten-epoch arms before GPU extraction')
    # Completed epoch rows can appear while a training launcher is still saving.
    for process in Path('/proc').iterdir():
        if not process.name.isdigit():
            continue
        try:
            argv = (process / 'cmdline').read_bytes().split(b'\0')
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        if any(Path(arg.decode(errors='replace')).name in (
                'train_visda_frozenbn_rta_entry.py', 'run_visda_frozenbn_rta_colab.py')
                for arg in argv if arg):
            raise RuntimeError('Training launcher still live; do not share GPU')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--check-only', action='store_true',
                        help='CPU checkpoint/interface check only; no image forward or output')
    args = parser.parse_args()
    torch.set_num_threads(2)
    checkpoint_path = ROOT / 'fixed2/visda-synthetic2real_seed1/last.pt'
    checkpoint = torch.load(str(checkpoint_path), map_location='cpu')
    if checkpoint['epoch'] != 10:
        raise ValueError('Require final epoch10, never oracle best.pt')
    sys.path.insert(0, str(CODE))
    from networks import ResNetFc, CLS
    model = torch.nn.Sequential(
        ResNetFc(model_path='/content/osda-datasets/resnet50-19c8e357.pth'), CLS(2048, 8))
    model.load_state_dict(checkpoint['model'], strict=True)
    source_names = read_names(DATA / 'source-known-6.txt')
    target_names = read_names(DATA / 'target-real-12.txt')
    if (len(source_names), len(target_names)) != (79765, 55388):
        raise ValueError('Unexpected input lists')
    if args.check_only:
        print('FINAL_GEOMETRY_INTERFACE_OK', json.dumps(dict(epoch=10, C=6, K=2,
              source_rows=len(source_names), target_rows=len(target_names))), flush=True)
        return
    require_finished_arms()
    if OUTPUT.exists():
        raise FileExistsError('Preserve previous cache; do not repeat checkpoint extraction')
    if not torch.cuda.is_available() or 'L4' not in torch.cuda.get_device_name():
        raise RuntimeError('Use the existing L4 runtime')
    OUTPUT.mkdir()
    model.cuda().eval()
    started = time.time()
    arrays = {}
    with torch.no_grad():
        for split, names in (('source', source_names), ('target', target_names)):
            features, logits = [], []
            loader = DataLoader(Images(names), batch_size=64, shuffle=False,
                                num_workers=4, pin_memory=True)
            for index, images in enumerate(loader, 1):
                _, feature, logit, _ = model(images.cuda(non_blocking=True))
                if feature.shape[1] != 256 or logit.shape[1] != 8:
                    raise ValueError('Unexpected RTA feature/head dimensions')
                if not torch.isfinite(feature).all() or not torch.isfinite(logit).all():
                    raise ValueError('Nonfinite diagnostic cache')
                features.append(feature.cpu().numpy())
                logits.append(logit.cpu().numpy())
                if index == 1 or index % 200 == 0:
                    print('FINAL_GEOMETRY_PROGRESS', split, index, len(loader), flush=True)
            arrays[split] = np.concatenate(features)
            arrays[split + '_logits'] = np.concatenate(logits)
    np.savez_compressed(str(OUTPUT / 'features.npz'), **arrays)
    manifest = dict(checkpoint=str(checkpoint_path), epoch=10, C=6, K=2,
        selection='Predeclared fixed-K final failure diagnostic, not best checkpoint',
        target_labels_used=False, source_list=str(DATA / 'source-known-6.txt'),
        target_list=str(DATA / 'target-real-12.txt'), shuffled=False,
        shapes={key: list(value.shape) for key, value in arrays.items()},
        seconds=time.time()-started, complete=True,
        purpose='Compare representation geometry with source-stage cache; no training or K fitting')
    (OUTPUT / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    print('FINAL_GEOMETRY_COMPLETE', json.dumps(manifest), flush=True)


if __name__ == '__main__':
    main()
