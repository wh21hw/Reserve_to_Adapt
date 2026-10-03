"""One actual-image forward/backward compatibility check before legacy training."""
import json
from pathlib import Path
import subprocess
import tarfile

root = Path('/content/rta-legacy-l4-bridge-v1')
if not root.exists():
    root.mkdir()
    with tarfile.open('/content/rta-baseline-code-v2-bridge.tar.gz') as archive:
        for member in archive.getmembers():
            if member.issym() or member.islnk() or not (root / member.name).resolve().is_relative_to(root.resolve()):
                raise ValueError('Unsafe archive path')
        archive.extractall(root)
code = r'''
import json, sys
from pathlib import Path
import torch, torchvision, faiss, numpy, scipy, sklearn
from PIL import Image
from torchvision import transforms
sys.path.insert(0, '/content/rta-legacy-l4-bridge-v1')
from networks import ResNetFc, CLS
torch.set_num_threads(2)
assert torch.cuda.is_available()
net = torch.nn.Sequential(ResNetFc(model_path='/content/osda-datasets/resnet50-19c8e357.pth'), CLS(2048,12)).cuda().train()
transform=transforms.Compose([transforms.Resize((256,256)),transforms.CenterCrop(224),transforms.ToTensor()])
rows=[line.rsplit(None,1) for line in Path('/content/amazon_0-9_train_all.txt').read_text().splitlines() if line.strip()][:8]
images=[]
for name,label in rows:
    with Image.open(Path('/content/osda-datasets')/name) as im:
        images.append(transform(im.convert('RGB')))
logits=net(torch.stack(images).cuda())[2]
loss=torch.nn.functional.cross_entropy(logits,torch.tensor([int(r[1]) for r in rows],device='cuda'))
loss.backward()
assert torch.isfinite(loss) and all(p.grad is None or torch.isfinite(p.grad).all() for p in net.parameters())
print('LEGACY_GPU_CHECK',json.dumps(dict(python=sys.version,torch=torch.__version__,torchvision=torchvision.__version__,cuda=torch.version.cuda,numpy=numpy.__version__,scipy=scipy.__version__,sklearn=sklearn.__version__,faiss=faiss.__version__,gpu=torch.cuda.get_device_name(),loss=float(loss.detach()),finite_gradients=True,batch_size=8,optimizer_updates=0)))
'''
result = subprocess.run(['/content/rta-py38/bin/python', '-u', '-c', code], capture_output=True, text=True)
print(result.stdout, flush=True)
print(result.stderr, flush=True)
if result.returncode:
    raise RuntimeError('Legacy L4 compatibility check failed; no training started')
row = next(line for line in result.stdout.splitlines() if line.startswith('LEGACY_GPU_CHECK '))
Path('/content/rta-legacy-l4-compatibility-v1.json').write_text(json.dumps(json.loads(row.split(' ',1)[1]), indent=2))
