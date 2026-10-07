"""One common warm state -> matched resumed RTA arms, no new model objective."""
import random
import numpy as np
import torch
from torch import nn


def extra_state(net, discriminator, centroids, wrappers, structure, gmm, virtual):
    return dict(source_relation_bank=centroids.src_ctrs,target_relation_bank=centroids.tgt_ctrs,
        relation_mixture=gmm,virtual_templates=virtual,
        optimizer_steps=[wrapper.global_step for wrapper in wrappers],
        grl_steps=discriminator.grl.global_step,
        assignments=structure.assignments,reliable=structure.reliable,
        target_paths=structure.target_names,
        rng_python=random.getstate(),rng_numpy=np.random.get_state(),
        rng_torch=torch.get_rng_state(),rng_cuda=torch.cuda.get_rng_state_all())


def restore_fork(path,net,cls,discriminator,centroids,wrappers,structure,args):
    state=torch.load(path,map_location='cpu')
    if state['epoch']!=4 or state['target_paths']!=structure.target_names:
        raise ValueError('Require matching common4-epoch state and target paths')
    old=cls.fc
    total=state['model']['1.fc.weight'].shape[0]
    if total!=old.out_features:
        head=nn.Linear(old.in_features,total,bias=False).to(old.weight)
        cls.fc=head;cls.main[1][2]=head
        changed=0
        for group in wrappers[1].optimizer.param_groups:
            for index,param in enumerate(group['params']):
                if param is old.weight:
                    group['params'][index]=head.weight;changed+=1
        if changed!=1:
            raise RuntimeError('Expected one live classifier parameter for fork restoration')
    args.all_classes=total
    net.load_state_dict(state['model'])
    discriminator.load_state_dict(state['discriminator'])
    for wrapper,key,step in zip(wrappers,('optimizer_feature','optimizer_cls','optimizer_discriminator'),state['optimizer_steps']):
        wrapper.optimizer.load_state_dict(state[key]);wrapper.global_step=step
    device=cls.fc.weight.device
    centroids.src_ctrs=state['source_relation_bank'].to(device)
    centroids.tgt_ctrs=state['target_relation_bank'].to(device)
    discriminator.grl.global_step=state['grl_steps']
    structure.assignments=state['assignments'];structure.reliable=state['reliable']
    random.setstate(state['rng_python']);np.random.set_state(state['rng_numpy'])
    torch.set_rng_state(state['rng_torch']);torch.cuda.set_rng_state_all(state['rng_cuda'])
    # Targeted new-interface invariants; no image/checkpoint rescoring orhash.
    momentum=wrappers[1].optimizer.state[cls.fc.weight].get('momentum_buffer')
    if momentum is None or momentum.shape!=cls.fc.weight.shape or momentum.device!=cls.fc.weight.device:
        raise RuntimeError('Restored classifier momentum shape/device mismatch')
    if not torch.equal(torch.get_rng_state(),state['rng_torch']):
        raise RuntimeError('Fork CPU RNG restoration failed')
    print('ONLINE_COMMON_WARM_STATE_RESTORED',dict(epoch=4,K=total-args.shared_classes,
        optimizer_steps=state['optimizer_steps'],grl_steps=state['grl_steps']),flush=True)
    return state['epoch'],state['relation_mixture'],state['virtual_templates'].to(device)
