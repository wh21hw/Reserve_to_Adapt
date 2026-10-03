"""Freeze fixed fourth warmup checkpoint; never use target-selected best.pt."""
import hashlib
import json
from pathlib import Path
import sys
import zipfile
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

code = Path('/content/rta-l4-control-v1')
sys.path.insert(0, str(code))
from networks import ResNetFc, CLS
sys.path.insert(0, '/content')
from source_warmup_features import Images, read_list

run = Path('/content/imp-runs/rta-space-warmup-l4-v1')
checkpoint_path = run/'a2w_seed1/last.pt'
output = Path('/content/imp-runs/rta-space-warmup-l4-features-v1')
if output.exists():
    raise RuntimeError('Refusing overwrite')
checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
assert checkpoint['epoch'] == 4
history = [json.loads(row) for row in (run/'a2w_seed1/history.jsonl').read_text().splitlines()]
assert [row['epoch'] for row in history] == [1,2,3,4]
torch.set_num_threads(2)
net = torch.nn.Sequential(ResNetFc(model_path='/content/osda-datasets/resnet50-19c8e357.pth'), CLS(2048,12)).cuda()
net.load_state_dict(checkpoint['model'])
net.eval()
root = Path('/content/osda-datasets')
source_list, target_list = Path('/content/amazon_0-9_train_all.txt'), Path('/content/webcam_0-9_20-30_test.txt')
source_paths, source_labels = read_list(source_list, root)
target_paths, target_labels = read_list(target_list, root)
transform = transforms.Compose([transforms.Resize((256,256)), transforms.CenterCrop(224), transforms.ToTensor()])

@torch.no_grad()
def extract(paths):
    features, logits = [], []
    for images in DataLoader(Images(paths, transform), batch_size=64, shuffle=False, num_workers=0):
        _, feature, logit, _ = net(images.cuda())
        features.append(feature.cpu().numpy())
        logits.append(logit.cpu().numpy())
    return np.concatenate(features), np.concatenate(logits)

source, source_logits = extract(source_paths)
target, target_logits = extract(target_paths)
assert source.shape == (958,256) and target.shape == (564,256)
assert all(np.isfinite(values).all() for values in [source,target,source_logits,target_logits])
output.mkdir(parents=True)
np.savez_compressed(output/'features.npz', source=source, target=target, source_labels=source_labels,
                    source_logits=source_logits, target_logits=target_logits)
np.savez_compressed(output/'evaluation-only.npz', target_labels=target_labels)
manifest = dict(stage='fixed fourth epoch RTA-space warmup in native L4 environment',
    checkpoint_sha256=hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(), epoch=4,
    selection='fixed-last; best.pt excluded', source_shape=list(source.shape), target_shape=list(target.shape),
    source_accuracy=float((source_logits.argmax(1)==source_labels).mean()),
    files={name: hashlib.sha256((output/name).read_bytes()).hexdigest() for name in ['features.npz','evaluation-only.npz']})
(output/'manifest.json').write_text(json.dumps(manifest, indent=2))
archive = Path('/content/rta-space-warmup-l4-features-v1.zip')
with zipfile.ZipFile(archive, 'x', zipfile.ZIP_DEFLATED) as bundle:
    for path in output.iterdir():
        bundle.write(path, path.name)
    for name in ['audit.json','console.log','a2w_seed1/config.json','a2w_seed1/history.jsonl']:
        bundle.write(run/name, 'training/'+name)
    for path in code.glob('*.py'):
        bundle.write(path, 'code/'+path.name)
print('RTA_FROZEN_FEATURES', json.dumps(manifest), flush=True)
print('RTA_FEATURE_ARCHIVE', archive.stat().st_size, hashlib.sha256(archive.read_bytes()).hexdigest(), flush=True)
del net, checkpoint
torch.cuda.empty_cache()
