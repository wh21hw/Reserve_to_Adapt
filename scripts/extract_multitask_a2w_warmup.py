"""Export seed1 fixed warm checkpoint features; target labels segregated for audit."""
import hashlib
import json
from pathlib import Path
import sys
import zipfile

import numpy as np
from PIL import Image
import torch
from torchvision import transforms

code = Path('/content/rta_multitask_baseline_v1')
sys.path.insert(0, str(code))
import networks
assert Path(networks.__file__).resolve() == code / 'networks.py'
run = Path('/content/imp-runs/rta-multitask-baseline-v1/office31-a2w_seed1')
checkpoint_path = run / 'warmup-complete.pt'
digest = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
assert digest == 'ff88bdc9d49a1a8c151baa8b4ca166549a2723195882ec37fe5056e0d91f1eb1'
audit = json.loads((run / 'audit-summary.json').read_text())
assert audit['epochs_verified'] == 70 and audit['completed_seed1']
config = json.loads((run / 'config.json').read_text())
launch = json.loads(Path('/content/rta-multitask-a2w-seed1-launch-v1.json').read_text())
for name, expected in launch['code_sha256'].items():
    assert hashlib.sha256((code / name).read_bytes()).hexdigest() == expected
output = Path('/content/imp-runs/multitask-a2w-seed1-warm-features-v1')
assert not output.exists(), 'Refusing overwrite'
checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
assert checkpoint['epoch'] == 4
torch.set_num_threads(2)
net = torch.nn.Sequential(
    networks.ResNetFc(model_path='/content/osda-datasets/resnet50-19c8e357.pth'),
    networks.CLS(2048, 12)).cuda().eval()
net.load_state_dict(checkpoint['model'], strict=True)
transform = transforms.Compose([
    transforms.Resize((256, 256)), transforms.CenterCrop(224), transforms.ToTensor()])

@torch.no_grad()
def extract(list_path):
    # Ignore target labels throughout feature extraction. Parse them only below
    # into a separate evaluation-only artifact, never an IMP input.
    names = [line.rsplit(None, 1)[0] for line in Path(list_path).read_text().splitlines() if line.strip()]
    features, logits = [], []
    for start in range(0, len(names), 64):
        images = []
        for name in names[start:start + 64]:
            with Image.open(Path(config['data_dir']) / name) as image:
                images.append(transform(image.convert('RGB')))
        _, feature, logit, _ = net(torch.stack(images).cuda())
        features.append(feature.cpu().numpy())
        logits.append(logit.cpu().numpy())
    return np.concatenate(features), np.concatenate(logits)

source, source_logits = extract(config['source'])
target, target_logits = extract(config['target'])
assert source.shape == (958, 256) and target.shape == (564, 256)
assert all(np.isfinite(value).all() for value in (source, target, source_logits, target_logits))
labels = lambda path: np.array([int(line.rsplit(None, 1)[1]) for line in Path(path).read_text().splitlines() if line.strip()])
output.mkdir(parents=True)
np.savez_compressed(output / 'features.npz', source=source, target=target,
                    source_labels=labels(config['source']), source_logits=source_logits,
                    target_logits=target_logits)
np.savez_compressed(output / 'evaluation-only.npz', target_labels=labels(config['target']))
manifest = dict(task='office31-a2w', seed=1, epoch=4, checkpoint_sha256=digest,
                selection='fixed warmup-complete; target-selected best excluded',
                target_labels_in_features=False, optimizer_steps_executed=0,
                code_sha256=launch['code_sha256'],
                files={name: hashlib.sha256((output / name).read_bytes()).hexdigest()
                       for name in ('features.npz', 'evaluation-only.npz')})
with (output / 'manifest.json').open('x') as stream:
    json.dump(manifest, stream, indent=2, allow_nan=False)
archive_path = Path('/content/multitask-a2w-seed1-warm-features-v1.zip')
with zipfile.ZipFile(archive_path, 'x', zipfile.ZIP_DEFLATED) as archive:
    for path in output.iterdir():
        archive.write(path, path.name)
    archive.write(Path(__file__), 'export-worker.py')
print('WARM_FEATURE_EXPORT', json.dumps(manifest), flush=True)
print('FEATURE_ARCHIVE', archive_path.name, archive_path.stat().st_size,
      hashlib.sha256(archive_path.read_bytes()).hexdigest(), flush=True)
