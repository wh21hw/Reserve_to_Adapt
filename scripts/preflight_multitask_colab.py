"""Fresh-process A->W actual entry data-loader/loss wiring check, no SGD step."""
import ast
import hashlib
import json
from pathlib import Path
import sys
import torch

code = Path('/content/rta_multitask_baseline_v1')
sys.path.insert(0, str(code))
assert torch.cuda.is_available() and 'L4' in torch.cuda.get_device_name()
torch.set_num_threads(2)
sys.argv = ['main.py', '--task', 'office31-a2w',
            '--source', '/content/amazon_0-9_train_all.txt',
            '--target', '/content/webcam_0-9_20-30_test.txt',
            '--data_dir', '/content/osda-datasets', '--virtual-clusters', '20',
            '--log_dir', '/content/imp-runs/multitask-preflight-v1', '--name', 'seed1']
tree = ast.parse((code / 'main.py').read_text())
# Execute the actual setup and loader definitions, stop BEFORE model/cluster/train.
prefix = []
for node in tree.body:
    if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'all_centroids' for t in node.targets):
        break
    prefix.append(node)
else:
    raise AssertionError('Missing expected setup boundary')
namespace = {'__name__': '__preflight__', '__file__': str(code / 'main.py')}
exec(compile(ast.Module(body=prefix, type_ignores=[]), str(code / 'main.py'), 'exec'), namespace)
import networks
assert Path(networks.__file__).resolve() == code / 'networks.py'
source_images, source_labels = next(iter(namespace['source_train']))
target_images, target_sentinel = next(iter(namespace['target_train']))
evaluation_images, evaluation_labels = next(iter(namespace['target_test']))
assert source_images.shape == target_images.shape == (64, 3, 224, 224)
assert source_labels.shape == (64, 12)
assert target_sentinel.shape == (64, 11) and torch.all(target_sentinel.argmax(1) == 10)
assert evaluation_labels.shape[1] == 31
weights_path = '/content/osda-datasets/resnet50-19c8e357.pth'
model = torch.nn.Sequential(networks.ResNetFc(model_path=weights_path), networks.CLS(2048, 12)).cuda().train()
discriminator = networks.LargeAdversarialNetwork(256).cuda().train()
source_images, target_images, source_labels = source_images.cuda(), target_images.cuda(), source_labels.cuda()
_, source_features, source_logits, source_probs = model(source_images)
_, target_features, target_logits, target_probs = model(target_images)
# Fixed random virtual directions only test shapes/gradient wiring, NOT learned templates.
virtual = torch.randn(10, 256, device='cuda')
truth = source_labels.argmax(1)
virtual_probs = model[1].virt_forward(virtual, source_features, source_logits, truth)
ce_fn, ent_fn, bce_fn = (namespace[name] for name in ('CrossEntropyLoss', 'EntropyLoss', 'BCELossForMultiClassification'))
ce = ce_fn(source_labels, source_probs)
virtual_ce = ce_fn(torch.cat([source_labels, torch.zeros(64, 10, device='cuda')], 1), virtual_probs)
entropy = ent_fn(target_probs, instance_level_weight=torch.ones(64, device='cuda'))
source_domain, target_domain = discriminator(source_features), discriminator(target_features)
adv = bce_fn(torch.ones_like(source_domain), source_domain)
adv += bce_fn(torch.ones_like(target_domain), 1 - target_domain,
              instance_level_weight=torch.ones(64, device='cuda'))
unknown_ce = torch.nn.functional.cross_entropy(target_logits[:16],
                                               target_logits[:16, 10:].argmax(1).detach() + 10)
loss = ce + .01 * virtual_ce + .3 * adv + entropy + unknown_ce
loss.backward()
assert torch.isfinite(loss)
assert all(p.grad is None or torch.isfinite(p.grad).all() for module in (model, discriminator) for p in module.parameters())
report = dict(task='office31-a2w', batch_size=64, gpu=torch.cuda.get_device_name(),
              loss=float(loss.detach()), optimizer_steps=0,
              training_target_labels_constant=True, evaluation_raw_labels_preserved=True,
              virtual_templates='fixed random wiring test, not learned clustering',
              gate='fixed ones wiring test, not GMM selection',
              unknown_selection='first16 wiring test, not full epoch selection',
              entry_sha256=hashlib.sha256((code / 'main.py').read_bytes()).hexdigest(),
              peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(), finite_gradients=True)
with Path('/content/multitask-preflight-v1.json').open('x') as stream:
    json.dump(report, stream, indent=2, allow_nan=False)
print('MULTITASK_PREFLIGHT_PASS', json.dumps(report), flush=True)
