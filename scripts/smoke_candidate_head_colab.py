"""Check pilot K=2 vs supported-candidate head wiring, not adaptation accuracy."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import zipfile
import numpy as np
import torch

head_path=Path('/content/candidate_head.py')
spec=importlib.util.spec_from_file_location('candidate_head',head_path)
head=importlib.util.module_from_spec(spec)
spec.loader.exec_module(head)
code=Path('/content/rta-l4-control-v1')
sys.path.insert(0,str(code))
from networks import ResNetFc,CLS
sys.path.insert(0,'/content')
from source_warmup_features import Images,read_list
from torchvision import transforms

checkpoint_path=Path('/content/imp-runs/rta-space-warmup-l4-v1/a2w_seed1/last.pt')
assert hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()=='0631572940759e0677632a1eadeb5bb4748912fa071e706a508df23411e03cd3'
checkpoint=torch.load(checkpoint_path,map_location='cpu',weights_only=False)
assert checkpoint['epoch']==4
proposal_path=Path('/content/imp-runs/farthest-constrained-frozen-v1/gate-seed1-original.npz')
proposal=np.load(proposal_path,allow_pickle=False)
candidates=torch.from_numpy(proposal['candidates']).cuda()
masses=torch.from_numpy(proposal['responsibilities'].sum(0)[10:]).cuda()
paths,labels=read_list(Path('/content/amazon_0-9_train_all.txt'),Path('/content/osda-datasets'))
transform=transforms.Compose([transforms.Resize((256,256)),transforms.CenterCrop(224),transforms.ToTensor()])
dataset=Images(paths[:4],transform)
images=torch.stack([dataset[index] for index in range(4)]).cuda()
truth=torch.from_numpy(labels[:4]).cuda()
torch.set_num_threads(2)
rows=[]
for limit in [2,None]:
    net=torch.nn.Sequential(ResNetFc(model_path='/content/osda-datasets/resnet50-19c8e357.pth'),CLS(2048,12)).cuda()
    net.load_state_dict(checkpoint['model'])
    known_before=net[1].fc.weight[:10].detach().clone()
    result=head.initialize_candidate_head(net[1],candidates,masses,10,min_effective_mass=5,max_slots=limit)
    assert torch.equal(net[1].fc.weight[:10],known_before)
    assert net[1].main[1][2] is net[1].fc
    optimizer=torch.optim.SGD(net.parameters(),lr=.0005)
    assert any(parameter is net[1].fc.weight for group in optimizer.param_groups for parameter in group['params'])
    net.train()
    _,feature,logits,_=net(images)
    assert logits.shape==(4,10+result['unknown_capacity'])
    loss=torch.nn.functional.cross_entropy(logits,truth)
    loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(net[1].fc.weight.grad).all()
    assert all(parameter.grad is None or torch.isfinite(parameter.grad).all() for parameter in net.parameters())
    rows.append(dict(result,limit=limit,logit_shape=list(logits.shape),source_batch_loss=float(loss.detach()),
        unknown_gradient_norm=float(net[1].fc.weight.grad[10:].norm())))
    del net,optimizer,known_before,feature,logits,loss
    torch.cuda.empty_cache()
report=dict(stage='candidate-capacity head forward/backward smoke, no optimizer steps',
    head_sha256=hashlib.sha256(head_path.read_bytes()).hexdigest(),
    proposal_sha256=hashlib.sha256(proposal_path.read_bytes()).hexdigest(),
    rule='Pilot only: mass>=5, primary predeclared gate seed1; not semantic class count or final selected capacity',
    target_labels_read=False,optimizer_states='New optimizer; no state-resume claim',arms=rows)
output=Path('/content/candidate-head-smoke-v1.json')
if output.exists():
    raise RuntimeError('Refusing overwrite')
output.write_text(json.dumps(report,indent=2,allow_nan=False))
archive=Path('/content/candidate-head-smoke-v1.zip')
with zipfile.ZipFile(archive,'x',zipfile.ZIP_DEFLATED) as bundle:
    bundle.write(output,output.name)
    bundle.write(head_path,head_path.name)
print('CANDIDATE_HEAD_SMOKE_PASS',json.dumps(report),flush=True)
print('HEAD_ARCHIVE',hashlib.sha256(archive.read_bytes()).hexdigest(),flush=True)
del images,candidates,masses,checkpoint
torch.cuda.empty_cache()
