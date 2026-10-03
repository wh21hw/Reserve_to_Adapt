"""CPU validation of actual warm state migration for structure off/on arms."""
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import torch

code = Path('/content/rta_multitask_baseline_v1')
sys.path.insert(0, str(code))
import networks
from utilities import OptimWithSheduler, inverseDecaySheduler
sys.path.insert(0, '/content')
from matched_handoff import restore_sgd, restore_rng
from prototype_structure import dirichlet_log_prior

def tensor_hash(tensor):
    return hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()

torch.set_num_threads(2)
checkpoint_path = Path('/content/imp-runs/rta-multitask-baseline-v1/office31-a2w_seed1/warmup-complete.pt')
assert hashlib.sha256(checkpoint_path.read_bytes()).hexdigest() == 'ff88bdc9d49a1a8c151baa8b4ca166549a2723195882ec37fe5056e0d91f1eb1'
checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
assert checkpoint['epoch'] == 4 and checkpoint['optimizer_steps'] == [56, 56, 56]
proposal_path = Path('/content/imp-runs/matched-warm-candidates-v1/original.npz')
assert hashlib.sha256(proposal_path.read_bytes()).hexdigest() == '3d4b9fe2833e89f97863db3f718bcf00dd4899d1221c80393e1668c44883721b'
proposal = np.load(proposal_path, allow_pickle=False)
counts = proposal['responsibilities'].sum(0)[10:]
selected = sorted(np.flatnonzero(counts >= 5), key=lambda i: (-counts[i], i))
centers = torch.from_numpy(proposal['candidates'][selected])
prior = dirichlet_log_prior(torch.from_numpy(counts[selected]), concentration=1.)
rows, previous = [], None
for arm, coefficient in [('structure_off', 0.), ('structure_on', .1)]:
    net = torch.nn.Sequential(networks.ResNetFc(model_path='/content/osda-datasets/resnet50-19c8e357.pth'), networks.CLS(2048, 12))
    net.load_state_dict(checkpoint['model'], strict=True)
    signatures = [[(n, tuple(p.shape)) for n, p in module.named_parameters()] for module in net]
    known = net[1].fc.weight[:10].detach().clone()
    net[1].fc.weight = torch.nn.Parameter(torch.cat([known, torch.nn.functional.normalize(centers, dim=1) * known.norm(dim=1).mean()]))
    net[1].fc.out_features = 10 + len(selected)
    net[1].register_buffer('unknown_log_weights', prior.clone())
    assert net[1].main[1][2] is net[1].fc
    discriminator = networks.LargeAdversarialNetwork(256)
    discriminator.load_state_dict(checkpoint['discriminator'], strict=True)
    modules = [net[0], net[1], discriminator]
    signatures.append([(n, tuple(p.shape)) for n, p in discriminator.named_parameters()])
    scheduler = lambda step, initial_lr: inverseDecaySheduler(step, initial_lr, gamma=10, power=.75, max_iter=10000)
    wrappers = [OptimWithSheduler(torch.optim.SGD(module.parameters(), lr=lr, momentum=.9, nesterov=True, weight_decay=5e-4), scheduler)
                for module, lr in zip(modules, (5e-5, 5e-4, 5e-4))]
    audits = []
    for i, (module, wrapper, key) in enumerate(zip(modules, wrappers, ('optimizer_feature', 'optimizer_cls', 'optimizer_discriminator'))):
        audits.append(restore_sgd(wrapper.optimizer, checkpoint[key], signatures[i], module.named_parameters(),
                                  replaced_head='fc.weight' if i == 1 else None, known_rows=10 if i == 1 else None))
        wrapper.global_step = checkpoint['optimizer_steps'][i]
    discriminator.grl.global_step = checkpoint['grl_steps']
    restore_rng(checkpoint)
    head_momentum = wrappers[1].optimizer.state[net[1].fc.weight]['momentum_buffer']
    # Check all unchanged parameters and known head rows against saved named-order states.
    for module, wrapper, key, signature in zip(modules, wrappers,
          ('optimizer_feature', 'optimizer_cls', 'optimizer_discriminator'), signatures):
        ids = [pid for group in checkpoint[key]['param_groups'] for pid in group['params']]
        for pid, (name, parameter) in zip(ids, module.named_parameters()):
            for kind, old in checkpoint[key]['state'].get(pid, {}).items():
                loaded = wrapper.optimizer.state[parameter][kind]
                assert torch.equal(loaded[:10], old[:10]) if module is net[1] and name == 'fc.weight' else torch.equal(loaded, old)
    assert torch.equal(known, checkpoint['model']['1.fc.weight'][:10])
    assert torch.count_nonzero(head_momentum[10:]) == 0
    state_hashes = {name: tensor_hash(value) for name, value in net.state_dict().items()}
    momentum_hashes = {f'{i}/{name}': tensor_hash(wrapper.optimizer.state[p]['momentum_buffer'])
                      for i, (module, wrapper) in enumerate(zip(modules, wrappers))
                      for name, p in module.named_parameters() if 'momentum_buffer' in wrapper.optimizer.state.get(p, {})}
    fingerprint = dict(model=state_hashes, momentum=momentum_hashes,
                       discriminator={n: tensor_hash(v) for n, v in discriminator.state_dict().items()},
                       rng_torch=tensor_hash(torch.get_rng_state()), optimizer_steps=[w.global_step for w in wrappers],
                       grl_steps=discriminator.grl.global_step)
    if previous is not None:
        assert fingerprint == previous
    previous = fingerprint
    row = dict(arm=arm, structure_coefficient=coefficient, components=len(selected),
               known_weights_sha256=tensor_hash(known), known_momentum_sha256=tensor_hash(head_momentum[:10]),
               all_unchanged_momentum_preserved=True, unknown_momentum_zero=True,
               fingerprint=fingerprint, migration=audits)
    rows.append(row)
    del net, discriminator, wrappers, modules
report = dict(arms=rows, matching_shared_states=True, checkpoint_epoch=4, optimizer_steps_executed=0,
              source_relation_bank_sha256=tensor_hash(checkpoint['source_relation_bank']),
              target_relation_bank_sha256=tensor_hash(checkpoint['target_relation_bank']),
              virtual_sha256=tensor_hash(checkpoint['virtual_templates']),
              relation_mixture_means=checkpoint['relation_mixture'].means_.tolist(),
              exact_uninterrupted_resume=False, training_replay_verified=False,
              policy='Saved banks/mixture/virtual preserved; candidate head replaced, known/nonhead momentum retained, unknown momentum zero in both arms')
with Path('/content/matched-real-handoff-v1.json').open('x') as stream:
    json.dump(report, stream, indent=2, allow_nan=False)
print('MATCHED_REAL_HANDOFF_PASS', json.dumps({k: v for k, v in report.items() if k != 'arms'}), flush=True)
