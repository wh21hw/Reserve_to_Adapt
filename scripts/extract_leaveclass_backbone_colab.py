"""Frozen, unit-normalized backbone features from the existing C20 checkpoint."""
from pathlib import Path
import sys
import json
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from train_source_prior_konly import Images

root=Path('/content/imp-runs/source-leaveclass-officehome-v1/seed1')
output=root/'backbone-features.npz'
if output.exists():
    raise FileExistsError('Preserve previous extraction')
sys.path.insert(0,'/content/rta-legacy-l4-bridge-v1')
from networks import ResNetFc, CLS
torch.set_num_threads(2)
state=torch.load(root/'source/source-final.pt',map_location='cpu')
if state['config']['known_classes']!=20:
    raise ValueError('Expected the C20 leave-class checkpoint')
net=torch.nn.Sequential(ResNetFc(model_path='/content/osda-datasets/resnet50-19c8e357.pth'),CLS(2048,20))
net.load_state_dict(state['model']); net=net.cuda().eval()
names=[line.rsplit(None,1)[0] for line in (root/'extract-all-source.txt').read_text().splitlines() if line.strip()]
fixed=transforms.Compose([transforms.Resize((256,256)),transforms.CenterCrop(224),transforms.ToTensor()])
loader=DataLoader(Images(names,Path('/content/osda-officehome-pr2rw-v1'),fixed),batch_size=64,
                  shuffle=False,num_workers=4,pin_memory=True)
features=[]
with torch.no_grad():
    for images in loader:
        value=net[0](images.cuda())
        value=torch.nn.functional.normalize(value,p=2,dim=1,eps=1e-8)
        features.append(value.cpu().numpy())
x=np.concatenate(features)
if x.shape!=(1785,2048) or not np.isfinite(x).all():
    raise ValueError('Invalid frozen backbone features')
np.savez_compressed(output,target=x)
print('BACKBONE_EXTRACTION_COMPLETE',json.dumps(dict(shape=list(x.shape),normalized=True,
    checkpoint='source/source-final.pt',retrained=False,real_target_used=False)),flush=True)
