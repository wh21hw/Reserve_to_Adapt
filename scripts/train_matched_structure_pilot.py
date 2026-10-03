"""ONE predeclared matched warm4->6 arm. No label-selected capacity or tuning."""
import argparse
import hashlib
import json
import random
from pathlib import Path
import sys
import time
import warnings
import faiss
import numpy as np
from PIL import Image
from scipy.optimize import linear_sum_assignment
from sklearn.mixture import BayesianGaussianMixture
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

code = Path('/content/rta_multitask_baseline_v1')
sys.path.insert(0, str(code))
import networks
from domain_bus import DomainBus
from utilities import OptimWithSheduler, inverseDecaySheduler, CrossEntropyLoss, BCELossForMultiClassification
from task_protocol import OFFICE31_A2W, macro_open_set_metrics
sys.path.insert(0, '/content')
from matched_handoff import restore_sgd, restore_rng
from prototype_structure import dirichlet_log_prior, geometric_log_teacher, conditional_structure_kl
from hierarchical_unknown import marginalize_unknown, semantic_entropy

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def tensor_sha(value):
    return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()

class TrainImages(Dataset):
    def __init__(self, names, labels, transform):
        self.names, self.labels, self.transform = names, labels, transform
    def __len__(self):
        return len(self.names)
    def __getitem__(self, index):
        with Image.open(Path('/content/osda-datasets') / self.names[index]) as image:
            tensor = self.transform(image.convert('RGB'))
        # Target label here is a constant sentinel, never a semantic label.
        return tensor, torch.tensor([self.labels[index], index])

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--arm', choices=('structure_off', 'structure_on'), required=True)
    args = parser.parse_args()
    coefficient = 0. if args.arm == 'structure_off' else .1
    assert Path(networks.__file__).resolve() == code / 'networks.py'
    assert torch.cuda.is_available() and 'L4' in torch.cuda.get_device_name()
    torch.set_num_threads(2)
    modules = {'matched_handoff.py': '789727cc8b498aa39aead54446f01bc8969c9f79935b4e68014b370abd87cbd8',
               'prototype_structure.py': '4f7ed0fa3bce8f2a2cea88d964917c3f69e333ee6b5257aa5e218deb04f964d8',
               'hierarchical_unknown.py': '58e9a31e674926c76aa52a8f13799a575d9dcc41a86da6d3a03141494fd76fc7'}
    for filename, digest in modules.items():
        assert sha(Path('/content') / filename) == digest
    assert sha(code / 'main.py') == '683c35e76a89ad1588f35dea83e3ecbc6f9a8af5bb3250e9e3de1cbfc3c7a148'
    assert sha(code / 'utilities.py') == '6ba276bb2d0076a4d64cde811496d2dfb71b330c8e1660add02d8fbd42e636e7'
    assert json.loads(Path('/content/matched-structure-realbatch-v1.json').read_text())['finite_gradients']
    handoff = json.loads(Path('/content/matched-real-handoff-v1.json').read_text())
    assert handoff['matching_shared_states']
    output = Path('/content/imp-runs/matched-structure-pilot-v1') / args.arm
    assert not output.exists(), 'Refusing overwrite'
    checkpoint_path = Path('/content/imp-runs/rta-multitask-baseline-v1/office31-a2w_seed1/warmup-complete.pt')
    features_path = Path('/content/imp-runs/multitask-a2w-seed1-warm-features-v1/features.npz')
    proposal_path = Path('/content/imp-runs/matched-warm-candidates-v1/original.npz')
    inputs = {str(checkpoint_path): 'ff88bdc9d49a1a8c151baa8b4ca166549a2723195882ec37fe5056e0d91f1eb1',
              str(features_path): 'a9d1a40aeb863b72aafefaed965d22d8eb3b3099516aee117f1897bdb4fb5459',
              str(proposal_path): '3d4b9fe2833e89f97863db3f718bcf00dd4899d1221c80393e1668c44883721b'}
    for path, digest in inputs.items():
        assert sha(path) == digest
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    data = np.load(features_path, allow_pickle=False)
    assert 'target_labels' not in data.files
    proposal = np.load(proposal_path, allow_pickle=False)
    counts = proposal['responsibilities'].sum(0)[10:]
    selected = sorted(np.flatnonzero(counts >= 5), key=lambda i: (-counts[i], i))
    # Construct on the same CPU surface as the audited fingerprint, then transfer.
    # Do not weaken exact state checks for CUDA reduction roundoff.
    cpu_centers = torch.from_numpy(proposal['candidates'][selected])
    centers = cpu_centers.cuda()
    prior = dirichlet_log_prior(torch.from_numpy(counts[selected]), concentration=1.).cuda()
    net = torch.nn.Sequential(networks.ResNetFc(model_path='/content/osda-datasets/resnet50-19c8e357.pth'), networks.CLS(2048, 12)).cuda()
    net.load_state_dict(checkpoint['model'], strict=True)
    signatures = [[(n, tuple(p.shape)) for n, p in module.named_parameters()] for module in net]
    known = net[1].fc.weight[:10].detach().clone()
    cpu_known = checkpoint['model']['1.fc.weight'][:10]
    cpu_weights = torch.cat([cpu_known, F.normalize(cpu_centers, dim=1) * cpu_known.norm(dim=1).mean()])
    net[1].fc.weight = torch.nn.Parameter(cpu_weights.cuda())
    net[1].fc.out_features = 10 + len(selected)
    net[1].register_buffer('unknown_log_weights', prior)
    assert net[1].main[1][2] is net[1].fc
    discriminator = networks.LargeAdversarialNetwork(256).cuda()
    discriminator.load_state_dict(checkpoint['discriminator'], strict=True)
    signatures.append([(n, tuple(p.shape)) for n, p in discriminator.named_parameters()])
    train_modules = (net[0], net[1], discriminator)
    scheduler = lambda step, initial_lr: inverseDecaySheduler(step, initial_lr, gamma=10, power=.75, max_iter=10000)
    wrappers = [OptimWithSheduler(torch.optim.SGD(module.parameters(), lr=lr, momentum=.9, nesterov=True, weight_decay=5e-4), scheduler)
                for module, lr in zip(train_modules, (5e-5, 5e-4, 5e-4))]
    for i, (module, wrapper, key) in enumerate(zip(train_modules, wrappers, ('optimizer_feature', 'optimizer_cls', 'optimizer_discriminator'))):
        restore_sgd(wrapper.optimizer, checkpoint[key], signatures[i], module.named_parameters(),
                    replaced_head='fc.weight' if i == 1 else None, known_rows=10 if i == 1 else None)
        wrapper.global_step = checkpoint['optimizer_steps'][i]
    discriminator.grl.global_step = checkpoint['grl_steps']
    reference = next(row for row in handoff['arms'] if row['arm'] == args.arm)['fingerprint']
    assert {n: tensor_sha(v) for n, v in net.state_dict().items()} == reference['model']
    assert {f'{i}/{n}': tensor_sha(wrapper.optimizer.state[p]['momentum_buffer'])
            for i, (module, wrapper) in enumerate(zip(train_modules, wrappers))
            for n, p in module.named_parameters() if 'momentum_buffer' in wrapper.optimizer.state.get(p, {})} == reference['momentum']
    source_features = torch.from_numpy(data['source'])
    source_labels = torch.from_numpy(data['source_labels'])
    source_centers = torch.stack([source_features[source_labels == c].mean(0) for c in range(10)])
    variance = float((source_features - source_centers[source_labels]).square().mean().clamp_min(1e-8))
    teacher = geometric_log_teacher(torch.from_numpy(data['target']).cuda(), centers, prior, variance)
    structure_gate = 1 - torch.from_numpy(proposal['known_compatibility']).cuda()
    bank = checkpoint['source_relation_bank'].cuda().clone()
    virtual = checkpoint['virtual_templates'].cuda().clone()
    mixture = checkpoint['relation_mixture']
    source_rows = [line.rsplit(None, 1) for line in Path('/content/amazon_0-9_train_all.txt').read_text().splitlines() if line.strip()]
    target_names = [line.rsplit(None, 1)[0] for line in Path('/content/webcam_0-9_20-30_test.txt').read_text().splitlines() if line.strip()]
    train_transform = transforms.Compose([transforms.Resize((256, 256)), transforms.RandomCrop(224), transforms.RandomHorizontalFlip(), transforms.ToTensor()])
    test_transform = transforms.Compose([transforms.Resize((256, 256)), transforms.CenterCrop(224), transforms.ToTensor()])
    source_loader = DataLoader(TrainImages([r[0] for r in source_rows], [int(r[1]) for r in source_rows], train_transform), batch_size=64, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
    target_loader = DataLoader(TrainImages(target_names, [10] * len(target_names), train_transform), batch_size=64, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
    assert len(source_loader) == 14 and len(target_loader) == 8
    restore_rng(checkpoint)
    assert tensor_sha(torch.get_rng_state()) == reference['rng_torch']
    output.mkdir(parents=True)
    config = dict(arm=args.arm, coefficient=coefficient, warm_epoch=4, final_epoch=6, seed=1,
                  capacity=len(selected), support_min=5, selected_indices=[int(i) for i in selected],
                  prior=prior.cpu().tolist(), variance=variance, input_sha256=inputs,
                  worker_sha256=sha(__file__), module_sha256=modules, batch_size=64,
                  policy=handoff['policy'], target_labels_used_for_training=False,
                  scope='Two-epoch stability pilot, not complete-budget performance evidence')
    (output / 'config.json').write_text(json.dumps(config, indent=2, allow_nan=False))
    net.train()
    discriminator.train()
    history = []
    for epoch in (4, 5):
        recorded = {name: [] for name in ('sf', 'tf', 'source_truth', 'source_probs', 'score')}
        loss_rows, order_hash = [], hashlib.sha256()
        for batch, ((source_images, source_meta), (target_images, target_meta)) in enumerate(DomainBus([source_loader, target_loader])):
            source_images, target_images = source_images.cuda(), target_images.cuda()
            truth = source_meta[:, 0].cuda()
            target_index = target_meta[:, 1].cuda()
            assert torch.all(target_meta[:, 0] == 10)
            order_hash.update(source_meta.numpy().tobytes())
            order_hash.update(target_meta.numpy().tobytes())
            _, sf, sl, _ = net(source_images)
            raw_target, tf, tl, _ = net(target_images)
            sg, _ = marginalize_unknown(sl, 10, prior)
            tg, conditional = marginalize_unknown(tl, 10, prior)
            score = F.kl_div(tl[:, :10].log_softmax(1), bank[tl[:, :10].argmax(1)], reduction='none').sum(1).detach()
            assert torch.isfinite(score).all()
            groups = mixture.predict(score.cpu().numpy()[:, None])
            known_group = int(mixture.means_.argmin())
            prob = mixture.predict_proba(score.cpu().numpy()[:, None])[:, known_group]
            weight = torch.as_tensor(prob > .8, device='cuda', dtype=sf.dtype)
            chosen = torch.as_tensor(np.flatnonzero(groups != known_group), device='cuda')
            if len(chosen) > 16:
                chosen = score.argsort()[-16:]
            unknown_ce = tl.sum() * 0
            if len(chosen) > 1:
                unknown_group, _ = marginalize_unknown(net[1](raw_target[chosen])[2], 10, prior)
                unknown_ce = F.cross_entropy(unknown_group, torch.full((len(chosen),), 10, device='cuda', dtype=torch.long))
            source_ce = F.cross_entropy(sg, truth)
            virtual_probs = net[1].virt_forward(virtual, sf, sg, truth)
            virtual_ce = CrossEntropyLoss(torch.cat([F.one_hot(truth, 11).float(), sf.new_zeros(64, len(virtual))], 1), virtual_probs)
            sd, td = discriminator(sf), discriminator(tf)
            adv = BCELossForMultiClassification(torch.ones_like(sd), sd)
            adv += BCELossForMultiClassification(torch.ones_like(td), 1 - td, instance_level_weight=weight)
            entropy = semantic_entropy(tg, weight)
            structure = conditional_structure_kl(conditional, teacher[target_index], structure_gate[target_index])
            loss = source_ce + .01 * virtual_ce + .3 * adv + entropy + unknown_ce + coefficient * structure
            for wrapper in wrappers:
                wrapper.zero_grad()
            assert torch.isfinite(loss), 'Nonfinite loss; no optimizer step'
            loss.backward()
            assert all(p.grad is None or torch.isfinite(p.grad).all() for module in train_modules for p in module.parameters()), 'Nonfinite gradient; no optimizer step'
            for wrapper in wrappers:
                wrapper.step()
            row = dict(epoch=epoch + 1, batch=batch + 1, total=float(loss.detach()), structure=float(structure.detach()),
                       source_ce=float(source_ce.detach()), virtual=float(virtual_ce.detach()), entropy=float(entropy.detach()),
                       unknown_ce=float(unknown_ce.detach()), adversarial=float(adv.detach()), unknown_selected=len(chosen))
            loss_rows.append(row)
            with (output / 'batches.jsonl').open('a') as stream:
                stream.write(json.dumps(row, allow_nan=False) + '\n')
            for key, value in [('sf', sf), ('tf', tf), ('source_truth', truth), ('source_probs', sl[:, :10].softmax(1)), ('score', score)]:
                recorded[key].append(value.detach().cpu().numpy())
            if batch in (0, 13):
                print('MATCHED_PILOT_BATCH', json.dumps(row), flush=True)
        assert len(loss_rows) == 14
        values = {key: np.concatenate(value) for key, value in recorded.items()}
        source_centers = []
        for c in range(10):
            mask = values['source_truth'] == c
            assert mask.any(), 'Source class absent; stop instead of producing NaN'
            bank[c].copy_(torch.from_numpy(values['source_probs'][mask].mean(0)).cuda())
            source_centers.append(values['sf'][mask].mean(0))
        kmeans = faiss.Kmeans(256, 20, niter=800, verbose=False, min_points_per_centroid=1, gpu=False)
        kmeans.train(values['tf'])
        _, matched = linear_sum_assignment(np.linalg.norm(np.stack(source_centers)[:, None] - kmeans.centroids[None], axis=-1))
        virtual = torch.from_numpy(kmeans.centroids[[i for i in range(20) if i not in matched]]).cuda()
        mixture = BayesianGaussianMixture(n_components=4, max_iter=800).fit(values['score'][:, None])
        assert torch.isfinite(bank).all() and torch.isfinite(virtual).all() and np.isfinite(mixture.means_).all()
        # Evaluation labels are first parsed here, after the full epoch; never select a checkpoint.
        evaluation_labels = [int(line.rsplit(None, 1)[1]) for line in Path('/content/webcam_0-9_20-30_test.txt').read_text().splitlines() if line.strip()]
        net.eval()
        predictions = []
        with torch.no_grad():
            for start in range(0, len(target_names), 64):
                images = []
                for name in target_names[start:start + 64]:
                    with Image.open(Path('/content/osda-datasets') / name) as image:
                        images.append(test_transform(image.convert('RGB')))
                group, _ = marginalize_unknown(net(torch.stack(images).cuda())[2], 10, prior)
                predictions.extend(group.argmax(1).cpu().tolist())
        metrics = macro_open_set_metrics(OFFICE31_A2W, evaluation_labels, predictions)
        net.train()
        report = dict(epoch=epoch + 1, arm=args.arm, coefficient=coefficient, metrics=metrics,
                      batch_count=14, batch_order_sha256=order_hash.hexdigest(),
                      loss_means={key: float(np.mean([r[key] for r in loss_rows])) for key in ('total', 'structure', 'source_ce', 'virtual', 'entropy', 'unknown_ce', 'adversarial')},
                      optimizer_steps=[w.global_step for w in wrappers], grl_steps=discriminator.grl.global_step,
                      mixture_converged=bool(mixture.converged_))
        history.append(report)
        with (output / 'history.jsonl').open('a') as stream:
            stream.write(json.dumps(report, allow_nan=False) + '\n')
        state = dict(model=net.state_dict(), discriminator=discriminator.state_dict(), epoch=epoch + 1,
                     optimizer_feature=wrappers[0].optimizer.state_dict(), optimizer_cls=wrappers[1].optimizer.state_dict(),
                     optimizer_discriminator=wrappers[2].optimizer.state_dict(), optimizer_steps=report['optimizer_steps'],
                     grl_steps=report['grl_steps'], source_relation_bank=bank,
                     target_relation_bank=checkpoint['target_relation_bank'], virtual_templates=virtual, relation_mixture=mixture,
                     rng_python=random.getstate(), rng_numpy=np.random.get_state(), rng_torch=torch.get_rng_state(),
                     rng_cuda=torch.cuda.get_rng_state_all(), teacher=teacher, structure_gate=structure_gate, config=config)
        torch.save(state, output / 'last.pt')
        print('MATCHED_PILOT_EPOCH', json.dumps(report), flush=True)
    assert [r['epoch'] for r in history] == [5, 6]
    assert wrappers[0].global_step == 84 and discriminator.grl.global_step == 168
    summary = dict(arm=args.arm, config=config, final=history[-1], complete_epochs=[5, 6],
                   actual_new_optimizer_steps=28, checkpoint_sha256=sha(output / 'last.pt'),
                   target_label_checkpoint_selection=False, exact_resume=False, complete_budget=False)
    (output / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False))
    print('MATCHED_PILOT_COMPLETE', json.dumps(summary), flush=True)

if __name__ == '__main__':
    main()
