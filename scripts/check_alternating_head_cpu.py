"""One targeted CPU interface check for growth/shrink and SGD state retention."""
import json
import sys
sys.path.insert(0, '/content')
sys.path.insert(0, '/content/rta-legacy-l4-bridge-v1')
import torch
from networks import CLS
from alternating_konly import resize_unknown_head

torch.manual_seed(3)
torch.set_num_threads(2)
cls = CLS(4, 5, bottle_neck_dim=3)
optimizer = torch.optim.SGD(cls.parameters(), lr=.01, momentum=.9)
cls(torch.randn(8, 4))[2].square().mean().backward()
optimizer.step()
for new_K in (5, 2, 2):
    previous = cls.fc.weight
    known_weights = previous[:2].detach().clone()
    known_momentum = optimizer.state[previous]['momentum_buffer'][:2].clone()
    probs = torch.softmax(torch.randn(8, cls.fc.out_features), dim=1)
    responsibilities = torch.softmax(torch.randn(8, 2+new_K), dim=1)
    report = resize_unknown_head(cls, optimizer, 2, responsibilities, probs)
    assert cls.fc is cls.main[1][2]
    assert torch.equal(cls.fc.weight[:2], known_weights)
    assert torch.equal(optimizer.state[cls.fc.weight]['momentum_buffer'][:2], known_momentum)
    assert sum(parameter is cls.fc.weight for group in optimizer.param_groups for parameter in group['params']) == 1
    if report['changed']:
        assert previous not in optimizer.state
    else:
        assert cls.fc.weight is previous
    optimizer.zero_grad()
    logits = cls(torch.randn(8, 4))[2]
    assert logits.shape == (8, 2+new_K) and torch.isfinite(logits).all()
    logits.square().mean().backward()
    optimizer.step()
    print('HEAD_INTERFACE_PASS', json.dumps(report), flush=True)
print('ALTERNATING_HEAD_CPU_COMPLETE: interface only, not an RTA training result', flush=True)
