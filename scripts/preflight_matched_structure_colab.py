"""Real train-mode image batch, restored warm state, gradients only: no SGD."""
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision import transforms

code = Path('/content/rta_multitask_baseline_v1')
sys.path.insert(0, str(code))
import networks
from utilities import CrossEntropyLoss, BCELossForMultiClassification
sys.path.insert(0, '/content')
from prototype_structure import dirichlet_log_prior, geometric_log_teacher, conditional_structure_kl
from hierarchical_unknown import marginalize_unknown, semantic_entropy
assert Path(networks.__file__).resolve() == code / 'networks.py'
assert 'L4' in torch.cuda.get_device_name()
for name, digest in {
    'prototype_structure.py': '4f7ed0fa3bce8f2a2cea88d964917c3f69e333ee6b5257aa5e218deb04f964d8',
    'hierarchical_unknown.py': '58e9a31e674926c76aa52a8f13799a575d9dcc41a86da6d3a03141494fd76fc7',
}.items():
    assert hashlib.sha256((Path('/content') / name).read_bytes()).hexdigest() == digest
for name, digest in {
    'main.py': '683c35e76a89ad1588f35dea83e3ecbc6f9a8af5bb3250e9e3de1cbfc3c7a148',
    'utilities.py': '6ba276bb2d0076a4d64cde811496d2dfb71b330c8e1660add02d8fbd42e636e7',
}.items():
    assert hashlib.sha256((code / name).read_bytes()).hexdigest() == digest
torch.set_num_threads(2)
torch.manual_seed(101)
torch.cuda.manual_seed_all(101)
path = Path('/content/imp-runs/rta-multitask-baseline-v1/office31-a2w_seed1/warmup-complete.pt')
assert hashlib.sha256(path.read_bytes()).hexdigest() == 'ff88bdc9d49a1a8c151baa8b4ca166549a2723195882ec37fe5056e0d91f1eb1'
checkpoint = torch.load(path, map_location='cpu', weights_only=False)
feature_path = Path('/content/imp-runs/multitask-a2w-seed1-warm-features-v1/features.npz')
assert hashlib.sha256(feature_path.read_bytes()).hexdigest() == 'a9d1a40aeb863b72aafefaed965d22d8eb3b3099516aee117f1897bdb4fb5459'
proposal_path = Path('/content/imp-runs/matched-warm-candidates-v1/original.npz')
assert hashlib.sha256(proposal_path.read_bytes()).hexdigest() == '3d4b9fe2833e89f97863db3f718bcf00dd4899d1221c80393e1668c44883721b'
features = np.load(feature_path, allow_pickle=False)
proposal = np.load(proposal_path, allow_pickle=False)
assert 'target_labels' not in features.files
source = torch.from_numpy(features['source'])
labels = torch.from_numpy(features['source_labels'])
anchors = torch.stack([source[labels == c].mean(0) for c in range(10)])
variance = float((source - anchors[labels]).square().mean().clamp_min(1e-8))
mass = proposal['responsibilities'].sum(0)[10:]
indices = sorted(np.flatnonzero(mass >= 5), key=lambda index: (-mass[index], index))
centers = torch.from_numpy(proposal['candidates'][indices]).cuda()
prior = dirichlet_log_prior(torch.as_tensor(mass[indices], device='cuda'), concentration=1.)
net = torch.nn.Sequential(networks.ResNetFc(model_path='/content/osda-datasets/resnet50-19c8e357.pth'),
                          networks.CLS(2048, 12)).cuda()
net.load_state_dict(checkpoint['model'], strict=True)
known_w = net[1].fc.weight[:10].detach().clone()
net[1].fc = torch.nn.Linear(256, 10 + len(indices), bias=False).cuda()
with torch.no_grad():
    net[1].fc.weight.copy_(torch.cat([known_w, F.normalize(centers, dim=1) * known_w.norm(dim=1).mean()]))
# main.1.2 aliases fc: update both, unlike merely replacing one attribute.
net[1].main[1][2] = net[1].fc
discriminator = networks.LargeAdversarialNetwork(256).cuda()
discriminator.load_state_dict(checkpoint['discriminator'], strict=True)
discriminator.grl.global_step = checkpoint['grl_steps']
net.train()
discriminator.train()
transform = transforms.Compose([transforms.Resize((256, 256)), transforms.RandomCrop(224),
                                transforms.RandomHorizontalFlip(), transforms.ToTensor()])
source_rows = [line.rsplit(None, 1) for line in Path('/content/amazon_0-9_train_all.txt').read_text().splitlines() if line.strip()]
target_names = [line.rsplit(None, 1)[0] for line in Path('/content/webcam_0-9_20-30_test.txt').read_text().splitlines() if line.strip()]
random = np.random.default_rng(101)
si = random.permutation(len(source_rows))[:64]
ti = random.permutation(len(target_names))[:64]
def images(names):
    result = []
    for name in names:
        with Image.open(Path('/content/osda-datasets') / name) as image:
            result.append(transform(image.convert('RGB')))
    return torch.stack(result).cuda()
source_images = images([source_rows[i][0] for i in si])
target_images = images([target_names[i] for i in ti])
truth = torch.tensor([int(source_rows[i][1]) for i in si], device='cuda')
teacher = geometric_log_teacher(torch.from_numpy(features['target'][ti]).cuda(), centers, prior, variance)
structure_gate = 1 - torch.from_numpy(proposal['known_compatibility'][ti]).cuda()
torch.cuda.reset_peak_memory_stats()
_, sf, sl, _ = net(source_images)
raw_target, tf, tl, _ = net(target_images)
sg, _ = marginalize_unknown(sl, 10, prior)
tg, conditional = marginalize_unknown(tl, 10, prior)
bank = checkpoint['source_relation_bank'].cuda()
pseudo = tl[:, :10].argmax(1)
score = F.kl_div(tl[:, :10].log_softmax(1), bank[pseudo], reduction='none').sum(1).detach()
assert torch.isfinite(score).all()
mixture = checkpoint['relation_mixture']
groups = mixture.predict(score.cpu().numpy()[:, None])
known_group = int(mixture.means_.argmin())
prob = mixture.predict_proba(score.cpu().numpy()[:, None])[:, known_group]
weight = torch.as_tensor(prob > .8, dtype=sf.dtype, device='cuda')
selected = torch.as_tensor(np.flatnonzero(groups != known_group), device='cuda')
if len(selected) > 16:
    selected = score.argsort()[-16:]
unknown_ce = tl.sum() * 0
if len(selected) > 1:
    unknown_logits = net[1](raw_target[selected])[2]
    unknown_group, _ = marginalize_unknown(unknown_logits, 10, prior)
    unknown_ce = F.cross_entropy(unknown_group, torch.full((len(selected),), 10, dtype=torch.long, device='cuda'))
virtual = checkpoint['virtual_templates'].cuda()
virtual_probs = net[1].virt_forward(virtual, sf, sg, truth)
virtual_labels = torch.cat([F.one_hot(truth, 11).float(), sf.new_zeros(64, len(virtual))], 1)
source_ce = F.cross_entropy(sg, truth)
virtual_ce = CrossEntropyLoss(virtual_labels, virtual_probs)
source_domain, target_domain = discriminator(sf), discriminator(tf)
adv = BCELossForMultiClassification(torch.ones_like(source_domain), source_domain)
adv += BCELossForMultiClassification(torch.ones_like(target_domain), 1 - target_domain, instance_level_weight=weight)
entropy = semantic_entropy(tg, weight)
base = source_ce + .01 * virtual_ce + .3 * adv + entropy + unknown_ce
structure = conditional_structure_kl(conditional, teacher, structure_gate)
parameters = [p for module in (net, discriminator) for p in module.parameters() if p.requires_grad]
base_grads = torch.autograd.grad(base, parameters, retain_graph=True, allow_unused=True)
structure_grads = torch.autograd.grad(structure, parameters, allow_unused=True)
for gradients in (base_grads, structure_grads):
    assert all(g is None or torch.isfinite(g).all() for g in gradients)
def norm(gradients):
    return float(sum((g.double().square().sum() for g in gradients if g is not None)).sqrt())
head_index = next(i for i, p in enumerate(parameters) if p is net[1].fc.weight)
assert torch.count_nonzero(structure_grads[head_index][:10]) == 0
report = dict(batch_size=64, gpu=torch.cuda.get_device_name(), components=len(indices),
              checkpoint_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
              loss_base=float(base.detach()), loss_structure=float(structure.detach()),
              base_gradient_norm=norm(base_grads), structure_gradient_norm=norm(structure_grads),
              base_head_gradient_norm=float(base_grads[head_index].norm()),
              structure_head_gradient_norm=float(structure_grads[head_index].norm()),
              structure_known_head_gradient_zero=True, finite_gradients=True, optimizer_steps=0,
              unknown_selected=int(len(selected)), known_gate_samples=int(weight.sum()),
              frozen_structure_gate_mass=float(structure_gate.sum()),
              peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(), target_labels_used=False,
              caveat='Single random train-mode augmented batch; hierarchical group control, not original flat RTA loss. Fixed center-crop teacher/gate. No optimizer/state handoff replay or full training verified.')
with Path('/content/matched-structure-realbatch-v1.json').open('x') as stream:
    json.dump(report, stream, indent=2, allow_nan=False)
print('MATCHED_STRUCTURE_REALBATCH_PASS', json.dumps(report), flush=True)
