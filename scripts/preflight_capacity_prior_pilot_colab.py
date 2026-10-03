"""Check real data handoff, loss gradients and persisted prior, without stepping."""
import json
from pathlib import Path
import sys
import torch
from torchvision import transforms

sys.path.insert(0,'/content/rta-capacity-prior-pilot-v1')
import networks
import pilot_state
assert Path(networks.__file__).resolve()==Path('/content/rta-capacity-prior-pilot-v1/networks.py')
assert Path(pilot_state.__file__).resolve()==Path('/content/rta-capacity-prior-pilot-v1/pilot_state.py')
from networks import ResNetFc,CLS,LargeAdversarialNetwork
from centroid import Centroids
from utilities import OptimWithSheduler,inverseDecaySheduler
from pilot_state import initialize,restore_optimizers
from capacity_prior import uniform_capacity_log_prior
sys.path.insert(0,'/content')
from source_warmup_features import Images,read_list

assert json.loads(Path('/content/capacity-prior-unit-v1.json').read_text())['duplicate_group_probability_invariant']
assert torch.cuda.is_available() and 'L4' in torch.cuda.get_device_name()
torch.set_num_threads(2)
net=torch.nn.Sequential(ResNetFc(model_path='/content/osda-datasets/resnet50-19c8e357.pth'),CLS(2048,12)).cuda()
discriminator=LargeAdversarialNetwork(256).cuda()
bank=Centroids(10,10,True)
checkpoint,gmm,virtual,audit=initialize(net,discriminator,bank,'proto18')
net[1].capacity_log_prior=uniform_capacity_log_prior(10,18,device='cuda')
scheduler=lambda step, initial_lr: inverseDecaySheduler(step,initial_lr,gamma=10,power=.75,max_iter=10000)
wrappers=[OptimWithSheduler(torch.optim.SGD(module.parameters(),lr=lr,weight_decay=5e-4,momentum=.9,nesterov=True),scheduler)
    for module,lr in [(net[0],5e-5),(net[1],5e-4),(discriminator,5e-4)]]
restore_optimizers(checkpoint,net,discriminator,wrappers,14,audit)
old=json.loads(Path('/content/capacity-pilot-handoff-v1.json').read_text())['arms'][2]
for key in ('known_weights_sha256','known_momentum_sha256','source_relation_bank_sha256','virtual_sha256','gmm_means','optimizer_steps','grl_steps'):
    assert audit[key]==old[key]
paths,labels=read_list(Path('/content/amazon_0-9_train_all.txt'),Path('/content/osda-datasets'))
transform=transforms.Compose([transforms.Resize((256,256)),transforms.CenterCrop(224),transforms.ToTensor()])
dataset=Images(paths[:4],transform)
images=torch.stack([dataset[i] for i in range(4)]).cuda()
net.train()
_,features,logits,probability=net(images)
assert logits.shape==(4,28)
loss=torch.nn.functional.cross_entropy(logits,torch.from_numpy(labels[:4]).cuda())
loss.backward()
assert torch.isfinite(loss) and all(p.grad is None or torch.isfinite(p.grad).all() for p in net.parameters())
state=net.state_dict()
assert '1.capacity_log_prior' in state and state['1.capacity_log_prior'].shape==(28,)
clone=torch.nn.Sequential(ResNetFc(model_path='/content/osda-datasets/resnet50-19c8e357.pth'),CLS(2048,28)).cuda()
clone.load_state_dict(state,strict=True)
assert torch.equal(clone[1].capacity_log_prior,net[1].capacity_log_prior)
net.eval(); clone.eval()
with torch.no_grad():
    assert torch.equal(net(images)[2],clone(images)[2])
report=dict(module_paths=dict(networks=networks.__file__,pilot_state=pilot_state.__file__),
    shared_handoff_matches_proto18=True,finite_source_backward=True,
    checkpoint_prior_persisted=True,reloaded_logits_equal=True,optimizer_steps_executed=0,
    capacity_log_prior=net[1].capacity_log_prior.cpu().tolist(),handoff=audit,target_labels_read=False)
with Path('/content/capacity-prior-preflight-v1.json').open('x') as stream:
    json.dump(report,stream,indent=2,allow_nan=False)
print('CAPACITY_PRIOR_PREFLIGHT_PASS',json.dumps(report),flush=True)
del net,clone,checkpoint,discriminator,bank,wrappers,images,features,logits,probability,loss,state
torch.cuda.empty_cache()
