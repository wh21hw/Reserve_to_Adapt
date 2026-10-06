"""One real-cache head/SGD bridge check, no image forward/refitting/training."""
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import numpy as np
import torch

sys.path.insert(0,'/content')
sys.path.insert(0,'/content/rta-legacy-l4-bridge-v1')
from networks import CLS
from capacity_refresh_boundary import apply_inferred_capacity

root = Path('/content/drive/MyDrive/OSDA/runs')
checkpoint = torch.load(root/'a2w-reconciled-identity-10e-v1/argmax-last.pt',map_location='cpu')
head = CLS(2048,18)
head.load_state_dict({key[2:]:value for key,value in checkpoint['model'].items() if key.startswith('1.')},strict=True)
optimizer = torch.optim.SGD(head.parameters(),lr=.0005,momentum=.9,nesterov=True,weight_decay=5e-4)
optimizer.load_state_dict(checkpoint['optimizer_cls'])
old_known = head.fc.weight[:10].detach().clone()
old_state = optimizer.state[head.fc.weight]['momentum_buffer'].clone()
with np.load(root/'a2w-current-relation-snapshot-v1/features.npz',allow_pickle=False) as data:
    probabilities = torch.softmax(torch.from_numpy(data['target_logits']),dim=1)
with np.load(root/'a2w-temporal-capacity-v1/clusters.npz',allow_pickle=False) as data:
    assignments = data['matched_assignments']
settings = json.loads((root/'a2w-temporal-capacity-v1/summary.json').read_text())
args = SimpleNamespace(shared_classes=10,all_classes=18)
report = apply_inferred_capacity(head,optimizer,args,assignments,
    settings['current']['matched_K'],probabilities,apply=True)
assert (args.all_classes,head.fc.out_features) == (14,14)
assert head.main[1][2] is head.fc
assert torch.equal(head.fc.weight[:10],old_known)
momentum = optimizer.state[head.fc.weight]['momentum_buffer']
assert torch.equal(momentum[:10],old_state[:10])
for new_row,old_row in report['row_mapping']:
    assert torch.equal(momentum[new_row],old_state[old_row])
assert sum(parameter is head.fc.weight for group in optimizer.param_groups for parameter in group['params']) == 1
print('A2W_CAPACITY_REFRESH_REAL_CACHE_INTERFACE_OK',json.dumps(report),flush=True)
