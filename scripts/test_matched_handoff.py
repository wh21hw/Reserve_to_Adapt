"""Synthetic CPU optimizer migration tests, not dataset training."""
import copy
import json
from pathlib import Path
import sys
import torch
from torch import nn
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, '/content')
from matched_handoff import restore_sgd

torch.manual_seed(11)
old = nn.Sequential(nn.Linear(3, 4), nn.Linear(4, 5, bias=False))
optimizer = torch.optim.SGD(old.parameters(), lr=.03, momentum=.9, nesterov=True)
old(torch.ones(2, 3)).sum().backward()
optimizer.step()  # One synthetic step establishes real momentum state.
saved = copy.deepcopy(optimizer.state_dict())
signature = [(name, tuple(p.shape)) for name, p in old.named_parameters()]
outputs = []
for _ in range(2):
    model = copy.deepcopy(old)
    known = model[1].weight[:2].detach().clone()
    model[1] = nn.Linear(4, 8, bias=False)
    with torch.no_grad():
        model[1].weight[:2].copy_(known)
    current = torch.optim.SGD(model.parameters(), lr=.03, momentum=.9, nesterov=True)
    audit = restore_sgd(current, saved, signature, model.named_parameters(), '1.weight', 2)
    for old_parameter, (name, parameter) in zip(old.parameters(), model.named_parameters()):
        buffer = current.state[parameter]['momentum_buffer']
        previous = optimizer.state[old_parameter]['momentum_buffer']
        assert torch.equal(buffer[:2], previous[:2]) if name == '1.weight' else torch.equal(buffer, previous)
    assert torch.count_nonzero(current.state[model[1].weight]['momentum_buffer'][2:]) == 0
    outputs.append(current.state_dict())
assert all(torch.equal(outputs[0]['state'][key]['momentum_buffer'], outputs[1]['state'][key]['momentum_buffer'])
           for key in outputs[0]['state'])
for malformed in ('order', 'state_kind', 'shape'):
    model = copy.deepcopy(old)
    current = torch.optim.SGD(model.parameters(), lr=.03, momentum=.9)
    bad = copy.deepcopy(saved)
    names = list(model.named_parameters())
    if malformed == 'order':
        names.reverse()
    elif malformed == 'state_kind':
        next(iter(bad['state'].values()))['exp_avg'] = torch.zeros(1)
    else:
        next(iter(bad['state'].values()))['momentum_buffer'] = torch.zeros(1)
    try:
        restore_sgd(current, bad, signature, names)
    except ValueError:
        pass
    else:
        raise AssertionError(f'Accepted malformed state: {malformed}')
print('MATCHED_HANDOFF_UNIT_PASS', json.dumps(dict(
    matching_arms=True, known_momentum_preserved=True, unknown_momentum_zero=True,
    malformed_cases_rejected=3, synthetic_optimizer_steps=1, dataset_optimizer_steps=0)))
