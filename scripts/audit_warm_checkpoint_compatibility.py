"""Compare old frozen-feature model with full baseline warmup; no training."""
import hashlib
import json
from pathlib import Path
import torch

torch.set_num_threads(2)
old_path = Path('/content/imp-runs/rta-space-warmup-l4-v1/a2w_seed1/last.pt')
new_path = Path('/content/imp-runs/rta-multitask-baseline-v1/office31-a2w_seed1/warmup-complete.pt')
assert hashlib.sha256(old_path.read_bytes()).hexdigest() == '0631572940759e0677632a1eadeb5bb4748912fa071e706a508df23411e03cd3'
assert hashlib.sha256(new_path.read_bytes()).hexdigest() == 'ff88bdc9d49a1a8c151baa8b4ca166549a2723195882ec37fe5056e0d91f1eb1'
old = torch.load(old_path, map_location='cpu', weights_only=False)
new = torch.load(new_path, map_location='cpu', weights_only=False)
assert old['epoch'] == new['epoch'] == 4
report = dict(old_checkpoint_sha256=hashlib.sha256(old_path.read_bytes()).hexdigest(),
              new_checkpoint_sha256=hashlib.sha256(new_path.read_bytes()).hexdigest(), epoch=4,
              actual_resume_replay_verified=False, optimizer_steps_executed=0)
for field in ('model', 'discriminator'):
    assert old[field].keys() == new[field].keys()
    different = [key for key in old[field] if not torch.equal(old[field][key], new[field][key])]
    report[field] = dict(equal_tensor_state=not different, different_keys=different,
                        max_absolute_difference=max(float((old[field][key].double() - new[field][key].double()).abs().max())
                                                    for key in old[field]))
report['old_frozen_features_model_compatible'] = report['model']['equal_tensor_state']
report['interpretation'] = ('Model states identical: old frozen features match this warm model; not a resume trajectory proof'
                             if report['old_frozen_features_model_compatible'] else
                             'Model states differ: regenerate frozen features from this fixed warm checkpoint before new IMP integration')
with Path('/content/warm-checkpoint-compatibility-v1.json').open('x') as stream:
    json.dump(report, stream, indent=2, allow_nan=False)
print('WARM_CHECKPOINT_COMPATIBILITY', json.dumps(report), flush=True)
