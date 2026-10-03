"""Fresh-process three-arm real-data backward and checkpoint replay checks."""
import json
from pathlib import Path
import sys
import torch
from torchvision import transforms

code=Path('/content/rta-hierarchical-pilot-v1')
sys.path.insert(0,str(code))
import networks,pilot_state
assert Path(networks.__file__).resolve()==code/'networks.py'
assert Path(pilot_state.__file__).resolve()==code/'pilot_state.py'
from networks import ResNetFc,CLS,LargeAdversarialNetwork
from centroid import Centroids
from utilities import OptimWithSheduler,inverseDecaySheduler,CrossEntropyLoss
from pilot_state import initialize,restore_optimizers
from hierarchical_unknown import semantic_entropy
sys.path.insert(0,'/content')
from source_warmup_features import Images,read_list

assert json.loads(Path('/content/hierarchical-unit-v1.json').read_text())['semantic_logits_invariant']
assert torch.cuda.is_available() and 'L4' in torch.cuda.get_device_name()
torch.set_num_threads(2)
source_paths,source_labels=read_list(Path('/content/amazon_0-9_train_all.txt'),Path('/content/osda-datasets'))
target_paths=[str(Path('/content/osda-datasets')/line.rsplit(' ',1)[0]) for line in
    Path('/content/webcam_0-9_20-30_test.txt').read_text().splitlines() if line.strip()]
transform=transforms.Compose([transforms.Resize((256,256)),transforms.CenterCrop(224),transforms.ToTensor()])
source_dataset=Images(source_paths[:4],transform); target_dataset=Images(target_paths[:4],transform)
source_images=torch.stack([source_dataset[i] for i in range(4)]).cuda()
target_images=torch.stack([target_dataset[i] for i in range(4)]).cuda()
truth=torch.from_numpy(source_labels[:4]).cuda()
old=json.loads(Path('/content/capacity-pilot-handoff-v1.json').read_text())['arms']
rows=[]
for index,arm in enumerate(('original2','proto2','proto18')):
    net=torch.nn.Sequential(ResNetFc(model_path='/content/osda-datasets/resnet50-19c8e357.pth'),CLS(2048,12)).cuda()
    discriminator=LargeAdversarialNetwork(256).cuda()
    bank=Centroids(10,10,True)
    checkpoint,gmm,virtual,audit=initialize(net,discriminator,bank,arm)
    scheduler=lambda step,initial_lr:inverseDecaySheduler(step,initial_lr,gamma=10,power=.75,max_iter=10000)
    wrappers=[OptimWithSheduler(torch.optim.SGD(module.parameters(),lr=lr,weight_decay=5e-4,momentum=.9,nesterov=True),scheduler)
        for module,lr in [(net[0],5e-5),(net[1],5e-4),(discriminator,5e-4)]]
    restore_optimizers(checkpoint,net,discriminator,wrappers,14,audit)
    for key in ('known_weights_sha256','known_momentum_sha256','source_relation_bank_sha256','virtual_sha256','gmm_means','optimizer_steps','grl_steps'):
        assert audit[key]==old[index][key]
    net.train()
    _,sf,sl,_=net(source_images); _,tf,tl,_=net(target_images)
    source_group,_=net[1].group_forward(sl); target_group,conditional=net[1].group_forward(tl)
    assert source_group.shape==(4,11) and conditional.shape==(4,net[1].fc.out_features-10)
    virtual_probs=net[1].virt_forward(virtual,sf,source_group,truth)
    virtual_labels=torch.cat([torch.nn.functional.one_hot(truth,11).float(),torch.zeros(4,10,device='cuda')],1)
    source_ce=torch.nn.functional.cross_entropy(source_group,truth)
    unknown_ce=torch.nn.functional.cross_entropy(target_group,torch.full((4,),10,device='cuda',dtype=torch.long))
    virtual_ce=CrossEntropyLoss(virtual_labels,virtual_probs)
    adv=torch.nn.functional.binary_cross_entropy(discriminator(sf),torch.ones(4,1,device='cuda'))
    adv+=torch.nn.functional.binary_cross_entropy(discriminator(tf),torch.zeros(4,1,device='cuda'))
    loss=source_ce+.01*virtual_ce+.3*adv+semantic_entropy(target_group)+unknown_ce
    loss.backward()
    assert torch.isfinite(loss) and all(p.grad is None or torch.isfinite(p.grad).all() for module in (net,discriminator) for p in module.parameters())
    assert torch.allclose(conditional.exp().sum(1),torch.ones(4,device='cuda'),atol=1e-6,rtol=0)
    state=net.state_dict()
    assert state['1.unknown_log_weights'].shape==(net[1].fc.out_features-10,)
    clone=torch.nn.Sequential(ResNetFc(model_path='/content/osda-datasets/resnet50-19c8e357.pth'),CLS(2048,net[1].fc.out_features)).cuda()
    clone.load_state_dict(state,strict=True)
    net.eval(); clone.eval()
    with torch.no_grad():
        assert torch.equal(net[1].group_forward(net(source_images)[2])[0],clone[1].group_forward(clone(source_images)[2])[0])
    rows.append(dict(arm=arm,handoff=audit,loss=float(loss.detach()),semantic_shape=[4,11],
        unknown_gradient_norm=float(net[1].fc.weight.grad[10:].norm()),finite=True,reloaded_semantic_logits_equal=True))
    print('HIERARCHICAL_ARM_PREFLIGHT_PASS',json.dumps(rows[-1]),flush=True)
    del net,clone,discriminator,bank,checkpoint,wrappers,state,sf,tf,sl,tl,source_group,target_group,conditional,virtual_probs,loss,adv,source_ce,unknown_ce,virtual_ce
    torch.cuda.empty_cache()
report=dict(arms=rows,module_paths=dict(networks=networks.__file__,pilot_state=pilot_state.__file__),
    optimizer_steps_executed=0,target_labels_read=False)
with Path('/content/hierarchical-preflight-v1.json').open('x') as stream:
    json.dump(report,stream,indent=2,allow_nan=False)
print('HIERARCHICAL_PREFLIGHT_PASS',flush=True)
