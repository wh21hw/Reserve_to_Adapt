"""Unpack pinned isolated control and test legacy architecture on L4."""
import hashlib
from pathlib import Path
import sys
import tarfile
import torch

archive = Path('/content/rta-l4-control-v1.tar.gz')
assert hashlib.sha256(archive.read_bytes()).hexdigest() == '5e14fa31467b1eb8d2a960273994bf48f7a9edeac03b081bc0738cb4a923ba58'
root = Path('/content/rta-l4-control-v1')
if root.exists():
    raise RuntimeError('Refusing overwrite')
root.mkdir()
with tarfile.open(archive) as bundle:
    for member in bundle.getmembers():
        if not (root/member.name).resolve().is_relative_to(root.resolve()) or not (member.isfile() or member.isdir()):
            raise RuntimeError('Unsafe member')
    bundle.extractall(root, filter='data')
sys.path.insert(0, str(root))
from networks import ResNetFc, CLS
backbone = ResNetFc(model_path='/content/osda-datasets/resnet50-19c8e357.pth').cuda()
classifier = CLS(2048, 12).cuda()
inputs = torch.rand(4, 3, 224, 224, device='cuda')
_, feature, logits, probability = classifier(backbone(inputs))
assert feature.shape == (4,256) and logits.shape == (4,12)
virtual = classifier.virt_forward(torch.randn(10,256,device='cuda'), feature, logits, torch.tensor([0,1,2,3],device='cuda'))
loss = -virtual[torch.arange(4,device='cuda'), torch.arange(4,device='cuda')].log().mean()
loss.backward()
assert torch.isfinite(loss) and torch.isfinite(classifier.fc.weight.grad).all()
print('RTA_CONTROL_CODE_PREFLIGHT_PASS', float(loss.detach()), flush=True)
