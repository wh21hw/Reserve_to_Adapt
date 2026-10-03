"""Audit fixed-final prior control, occupancy, and preserve artifacts."""
import hashlib
import json
from pathlib import Path
import sys
import zipfile
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

code=Path('/content/rta-capacity-prior-pilot-v1')
sys.path.insert(0,str(code))
import networks
assert Path(networks.__file__).resolve()==code/'networks.py'
from networks import ResNetFc,CLS
sys.path.insert(0,'/content')
from source_warmup_features import Images

root=Path('/content/imp-runs/rta-capacity-prior-pilot-v1/uniform18')
run=root/'a2w_seed1'
assert json.loads((root/'process-status.json').read_text())['exit_code']==0
history=[json.loads(line) for line in (run/'history.jsonl').read_text().splitlines()]
assert [row['epoch'] for row in history]==[5,6]
checkpoint_path=run/'last.pt'
checkpoint=torch.load(checkpoint_path,map_location='cpu',weights_only=False)
assert checkpoint['epoch']==6
prior=checkpoint['model']['1.capacity_log_prior']
assert prior.shape==(28,) and torch.count_nonzero(prior[:10])==0
assert torch.allclose(prior[10:],torch.full((18,),-np.log(9)))
torch.set_num_threads(2)
net=torch.nn.Sequential(ResNetFc(model_path='/content/osda-datasets/resnet50-19c8e357.pth'),CLS(2048,28)).cuda().eval()
net.load_state_dict(checkpoint['model'],strict=True)
paths=[str(Path('/content/osda-datasets')/line.rsplit(' ',1)[0]) for line in
    Path('/content/webcam_0-9_20-30_test.txt').read_text().splitlines() if line.strip()]
transform=transforms.Compose([transforms.Resize((256,256)),transforms.CenterCrop(224),transforms.ToTensor()])
logits=[]
with torch.no_grad():
    for images in DataLoader(Images(paths,transform),batch_size=64,shuffle=False,num_workers=0):
        logits.append(net(images.cuda())[2].cpu())
logits=torch.cat(logits)
assert logits.shape==(564,28) and torch.isfinite(logits).all()
probabilities=logits.softmax(1)
hard=torch.bincount(logits.argmax(1),minlength=28)
np.savez_compressed(root/'final-target-logits.npz',logits=logits.numpy())
handoff=json.loads((run/'handoff.json').read_text())
baseline=json.loads(Path('/content/imp-runs/rta-capacity-pilot-v1/summary.json').read_text())['arms'][2]
for key in ('known_weights_sha256','known_momentum_sha256','source_relation_bank_sha256','virtual_sha256','gmm_means','optimizer_steps','grl_steps'):
    assert handoff[key]==baseline['handoff'][key]
report=dict(stage='Fixed capacity18 group prior control, two adaptation epochs, seed1',
    final=history[-1],history=history,checkpoint_sha256=hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
    handoff=handoff,shared_handoff_matches_raw18=True,
    occupancy=dict(unknown_hard_count=int(hard[10:].sum()),known_hard_count=int(hard[:10].sum()),
        occupied_unknown_slots=int((hard[10:]>0).sum()),unknown_slot_hard_counts=hard[10:].tolist(),
        mean_unknown_group_probability=float(probabilities[:,10:].sum(1).mean()),target_labels_read=False),
    selection='Predeclared correction/budget; target labels for evaluation only, not capacity or bias tuning',
    limitation='Not complete DPMM; entropy, pseudo-label CE and per-slot argmax remain capacity-dependent',
    raw18_fixed_final=baseline['final'])
with (root/'summary.json').open('x') as stream:
    json.dump(report,stream,indent=2,allow_nan=False)
result_archive=Path('/content/rta-capacity-prior-pilot-v1-results.zip')
with zipfile.ZipFile(result_archive,'x',zipfile.ZIP_DEFLATED) as bundle:
    for path in root.rglob('*'):
        if path.is_file() and path.suffix!='.pt':
            bundle.write(path,str(path.relative_to(root)))
    for path in code.glob('*.py'):
        bundle.write(path,'code/'+path.name)
    for name in ('capacity-prior-unit-v1.json','capacity-prior-preflight-v1.json',
                 'capacity-prior-import-failure-v1.json','capacity-prior-preflight-console-v1.log'):
        bundle.write(Path('/content')/name,name)
    for name in ('collect_capacity_prior_worker_v1.py','preflight_capacity_prior_worker_v1.py'):
        bundle.write(Path('/content')/name,'workers/'+name)
checkpoint_archive=Path('/content/rta-capacity-prior-pilot-v1-checkpoint.zip')
with zipfile.ZipFile(checkpoint_archive,'x',zipfile.ZIP_STORED) as bundle:
    bundle.write(checkpoint_path,'last.pt')
    bundle.write(run/'handoff.json','handoff.json')
    bundle.write(root/'audit.json','audit.json')
print('CAPACITY_PRIOR_RESULTS',json.dumps(report),flush=True)
for archive in (result_archive,checkpoint_archive):
    print('CAPACITY_PRIOR_ARCHIVE',archive.name,archive.stat().st_size,hashlib.sha256(archive.read_bytes()).hexdigest(),flush=True)
del net,checkpoint,images,logits,probabilities
torch.cuda.empty_cache()
