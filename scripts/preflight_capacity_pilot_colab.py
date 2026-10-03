"""Verify three controlled handoffs before any adaptation optimizer step."""
import json
from pathlib import Path
import sys
import torch

code = Path('/content/rta-capacity-pilot-v1')
sys.path.insert(0, str(code))
from networks import ResNetFc, CLS, LargeAdversarialNetwork
from utilities import OptimWithSheduler, inverseDecaySheduler
from centroid import Centroids
from pilot_state import initialize, restore_optimizers

if not torch.cuda.is_available() or 'L4' not in torch.cuda.get_device_name():
    raise RuntimeError('Expected L4')
torch.set_num_threads(2)
rows = []
for arm in ('original2', 'proto2', 'proto18'):
    net = torch.nn.Sequential(ResNetFc(model_path='/content/osda-datasets/resnet50-19c8e357.pth'), CLS(2048,12)).cuda()
    discriminator = LargeAdversarialNetwork(256).cuda()
    bank = Centroids(10,10,True)
    checkpoint, gmm, virtual, audit = initialize(net, discriminator, bank, arm)
    scheduler = lambda step, initial_lr: inverseDecaySheduler(step, initial_lr, gamma=10, power=.75, max_iter=10000)
    wrappers = [OptimWithSheduler(torch.optim.SGD(module.parameters(), lr=lr,
        momentum=.9, weight_decay=5e-4, nesterov=True), scheduler)
        for module, lr in [(net[0],5e-5),(net[1],5e-4),(discriminator,5e-4)]]
    restore_optimizers(checkpoint,net,discriminator,wrappers,14,audit)
    assert any(p is net[1].fc.weight for g in wrappers[1].optimizer.param_groups for p in g['params'])
    rows.append(audit)
    print('HANDOFF_ARM_PASS',json.dumps(audit),flush=True)
    del net, discriminator, bank, wrappers, checkpoint, virtual
    torch.cuda.empty_cache()
for key in ('known_weights_sha256','known_momentum_sha256','source_relation_bank_sha256','virtual_sha256','gmm_means','optimizer_steps','grl_steps'):
    assert all(row[key] == rows[0][key] for row in rows)
output=Path('/content/capacity-pilot-handoff-v1.json')
with output.open('x') as stream:
    json.dump(dict(arms=rows, optimizer_steps_executed=0, matching_shared_states=True),stream,indent=2,allow_nan=False)
print('THREE_ARM_HANDOFF_PASS',str(output),flush=True)
