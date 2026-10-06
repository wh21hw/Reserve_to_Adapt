"""One frozen final10 cache for a temporal capacity-inference study.

No SGD, target truth, GMM fitting, K selection, or checkpoint accuracy evaluation.
Known relation scores are reconstructed under the frozen model, not historical
training-time EMA/augmentation scores. Both source and target use one model state.
"""
import argparse
import json
from pathlib import Path
import shutil
import sys
import time
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from torchvision import transforms


class Images(Dataset):
    def __init__(self, names, root):
        self.names, self.root = names, root
        self.transform = transforms.Compose([transforms.Resize((256,256)),
            transforms.CenterCrop(224), transforms.ToTensor()])

    def __len__(self):
        return len(self.names)

    def __getitem__(self,index):
        with Image.open(self.root/self.names[index]) as image:
            return self.transform(image.convert('RGB'))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', default='/content/drive/MyDrive/OSDA/runs/a2w-reconciled-identity-10e-v1/argmax-last.pt')
    parser.add_argument('--code-root', default='/content/rta-legacy-l4-bridge-v1')
    parser.add_argument('--data-root', default='/content/osda-office31-a2w-v1')
    parser.add_argument('--weights', default='/content/osda-datasets/resnet50-19c8e357.pth')
    parser.add_argument('--output', default='/content/imp-runs/a2w-current-relation-snapshot-v1')
    parser.add_argument('--device', choices=['cpu','cuda'], default='cuda')
    parser.add_argument('--check-only', action='store_true', help='CPU state/interface load, no images/output')
    args = parser.parse_args()
    checkpoint_path, output, data = Path(args.checkpoint), Path(args.output), Path(args.data_root)
    if not checkpoint_path.name.endswith('last.pt'):
        raise ValueError('Use the declared final checkpoint, not oracle best')
    checkpoint = torch.load(str(checkpoint_path),map_location='cpu')
    if checkpoint['epoch'] != 10 or checkpoint['model']['1.fc.weight'].shape != (18,256):
        raise ValueError('Require the completed C10/K8 final10 model')
    torch.set_num_threads(2)
    sys.path.insert(0,args.code_root)
    from networks import ResNetFc,CLS
    from relation_gate import source_relation_scores
    model = torch.nn.Sequential(ResNetFc(model_path=args.weights),CLS(2048,18))
    model.load_state_dict(checkpoint['model'],strict=True)
    del checkpoint
    source_rows = [line.rsplit(None,1) for line in
        (data/'amazon_0-9_train_all.txt').read_text().splitlines() if line.strip()]
    source_names = [row[0] for row in source_rows]
    source_labels = np.asarray([int(row[1]) for row in source_rows],dtype=np.int64)
    # Discard target semantic columns at the boundary; never parse their values.
    target_names = [line.rsplit(None,1)[0] for line in
        (data/'webcam_0-9_20-30_test.txt').read_text().splitlines() if line.strip()]
    if len(source_names) != 958 or len(target_names) != 564 or not np.array_equal(np.unique(source_labels),np.arange(10)):
        raise ValueError('Unexpected complete A2W task inputs')
    if args.check_only:
        print('A2W_CURRENT_RELATION_INTERFACE_OK; no image forward, cache or training',flush=True)
        return
    if output.exists():
        raise FileExistsError('Reuse the existing frozen cache; do not repeat extraction')
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; do not silently change device')
    device = torch.device(args.device)
    model.to(device).eval()
    started, arrays = time.time(), {}
    with torch.no_grad():
        for split,names in [('source',source_names),('target',target_names)]:
            features, logits = [],[]
            # Smaller inference batches are not a training batch-size change.
            loader = DataLoader(Images(names,data),batch_size=32,shuffle=False,
                num_workers=2,pin_memory=args.device=='cuda')
            for images in loader:
                _,feature,logit,_ = model(images.to(device))
                if feature.shape[1] != 256 or logit.shape[1] != 18:
                    raise ValueError('Unexpected feature/head interface')
                if not torch.isfinite(feature).all() or not torch.isfinite(logit).all():
                    raise RuntimeError('Nonfinite frozen outputs')
                features.append(feature.cpu().numpy()); logits.append(logit.cpu().numpy())
            arrays[split] = np.concatenate(features)
            arrays[split+'_logits'] = np.concatenate(logits)
            print('CACHED_FROZEN_SPLIT',split,len(names),flush=True)
    labels = torch.from_numpy(source_labels)
    source_logits = torch.from_numpy(arrays['source_logits'])
    for split in ('source','target'):
        relation = source_relation_scores(source_logits,labels,torch.from_numpy(arrays[split+'_logits']),10)
        arrays[split+'_relation_kl'] = relation['scores'].numpy()
    arrays['source_soft_prototypes'] = relation['source_soft_prototypes'].numpy()
    arrays['source_labels'] = source_labels
    arrays['target_paths'] = np.asarray(target_names)
    output.mkdir(parents=True)
    np.savez_compressed(output/'features.npz',**arrays)
    manifest = dict(checkpoint=str(checkpoint_path),epoch=10,C=10,K=8,
        selection='Declared final10 argmax control, never target-oracle best',
        target_labels_used=False,training=False,GMM_fitted=False,K_estimated=False,
        reconstruction='Current frozen-model source probability means; not saved historical relation bank/GMM',
        transform='Resize256, CenterCrop224, ToTensor; eval BN; full actual head output',
        kl_direction='source class probability mean || target C-only conditional probability',
        device=args.device,shuffled=False,seconds=time.time()-started,
        shapes={key:list(value.shape) for key,value in arrays.items()},complete=True)
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2,allow_nan=False))
    saved = Path('/content/drive/MyDrive/OSDA/runs/a2w-current-relation-snapshot-v1')
    if not saved.parent.is_dir():
        raise RuntimeError('Drive missing; local cache preserved, do not discard runtime')
    saved.mkdir(exist_ok=False)
    for filename in ('features.npz','manifest.json'):
        shutil.copyfile(output/filename,saved/filename)
    print('A2W_CURRENT_RELATION_CACHE_COMPLETE',json.dumps(manifest),flush=True)


if __name__ == '__main__':
    main()
