"""Fixed-final occupancy diagnostics; no target labels or model selection."""
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

sys.path.insert(0,'/content/rta-capacity-pilot-v1')
from networks import ResNetFc, CLS
sys.path.insert(0,'/content')
from source_warmup_features import Images

root=Path('/content/imp-runs/rta-capacity-pilot-v1')
image_root=Path('/content/osda-datasets')
paths=[str(image_root/line.rsplit(' ',1)[0]) for line in
       Path('/content/webcam_0-9_20-30_test.txt').read_text().splitlines() if line.strip()]
transform=transforms.Compose([transforms.Resize((256,256)),transforms.CenterCrop(224),transforms.ToTensor()])
torch.set_num_threads(2)
rows=[]
for arm,width in [('original2',12),('proto2',12),('proto18',28)]:
    checkpoint_path=root/arm/'a2w_seed1/last.pt'
    checkpoint=torch.load(checkpoint_path,map_location='cpu',weights_only=False)
    assert checkpoint['epoch']==6
    net=torch.nn.Sequential(ResNetFc(model_path='/content/osda-datasets/resnet50-19c8e357.pth'),CLS(2048,width)).cuda().eval()
    net.load_state_dict(checkpoint['model'])
    logits=[]
    with torch.no_grad():
        for images in DataLoader(Images(paths,transform),batch_size=64,shuffle=False,num_workers=0):
            logits.append(net(images.cuda())[2].cpu())
    logits=torch.cat(logits)
    assert torch.isfinite(logits).all() and len(logits)==564
    probabilities=logits.softmax(1)
    hard=torch.bincount(logits.argmax(1),minlength=width)
    unknown_probability=probabilities[:,10:].sum(1)
    unknown_logits=logits[:,10:]
    group_margin=unknown_logits.logsumexp(1)-logits[:,:10].logsumexp(1)
    row=dict(arm=arm,checkpoint_sha256=hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
        known_hard_count=int(hard[:10].sum()),unknown_hard_count=int(hard[10:].sum()),
        unknown_slot_hard_counts=hard[10:].tolist(),unknown_slot_soft_mass=probabilities[:,10:].sum(0).tolist(),
        occupied_unknown_slots=int((hard[10:]>0).sum()),mean_unknown_group_probability=float(unknown_probability.mean()),
        mean_group_logsumexp_margin=float(group_margin.mean()),
        target_labels_read=False,selection='Fixed final epoch6, no capacity retuning from these results')
    np.savez_compressed(root/arm/'final-target-logits.npz',logits=logits.numpy())
    rows.append(row)
    print('PILOT_OCCUPANCY',json.dumps(row),flush=True)
    del net,checkpoint,logits,probabilities,images
    torch.cuda.empty_cache()
with (root/'occupancy.json').open('x') as stream:
    json.dump(rows,stream,indent=2,allow_nan=False)
