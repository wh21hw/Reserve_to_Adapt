"""Audit full seed1 and independently evaluate fixed-last; no training or tuning."""
import hashlib
import json
import math
from pathlib import Path
import sys
import zipfile
import numpy as np
from PIL import Image
import torch
from torchvision import transforms

code = Path('/content/rta_multitask_baseline_v1')
sys.path.insert(0, str(code))
import networks
from task_protocol import OFFICE31_A2W, macro_open_set_metrics
assert Path(networks.__file__).resolve() == code / 'networks.py'
run = Path('/content/imp-runs/rta-multitask-baseline-v1/office31-a2w_seed1')
launch = json.loads(Path('/content/rta-multitask-a2w-seed1-launch-v1.json').read_text())
config = json.loads((run / 'config.json').read_text())
assert launch['epochs'] == 70 and launch['seed'] == config['seed'] == 1
assert config['shared_classes'] == 10 and config['all_classes'] == 12 and config['virtual_clusters'] == 20
for name, digest in launch['code_sha256'].items():
    assert hashlib.sha256((code / name).read_bytes()).hexdigest() == digest
history = [json.loads(line) for line in (run / 'history.jsonl').read_text().splitlines()]
assert [row['epoch'] for row in history] == list(range(1, 71))
for row in history:
    assert row['seed'] == 1
    assert all(math.isfinite(row[key]) and 0 <= row[key] <= 1
               for key in ('OS', 'OS_star', 'unknown', 'HOS'))
last = torch.load(run / 'last.pt', map_location='cpu', weights_only=False)
warm = torch.load(run / 'warmup-complete.pt', map_location='cpu', weights_only=False)
best = torch.load(run / 'best.pt', map_location='cpu', weights_only=False)
assert last['epoch'] == 70 and warm['epoch'] == 4
assert last['metrics'] == history[-1]
resume_keys = ('source_relation_bank', 'target_relation_bank', 'virtual_templates',
               'relation_mixture', 'optimizer_steps', 'grl_steps', 'rng_python',
               'rng_numpy', 'rng_torch', 'rng_cuda')
for checkpoint, steps in ((warm, 56), (last, 980)):
    assert all(key in checkpoint for key in resume_keys)
    assert checkpoint['optimizer_steps'] == [steps] * 3
    assert checkpoint['grl_steps'] == steps * 2
    for state_key in ('model', 'discriminator'):
        assert all(torch.isfinite(tensor).all() for tensor in checkpoint[state_key].values())
    assert checkpoint['source_relation_bank'].shape == (10, 10)
    assert checkpoint['virtual_templates'].shape == (10, 256)
    assert torch.isfinite(checkpoint['source_relation_bank']).all()
    assert torch.isfinite(checkpoint['virtual_templates']).all()
oracle_index = max(range(70), key=lambda index: history[index]['HOS'])
assert best['epoch'] == oracle_index == history[-1]['best']['epoch']
assert best['HOS'] == history[oracle_index]['HOS']
torch.set_num_threads(2)
model = torch.nn.Sequential(networks.ResNetFc(model_path='/content/osda-datasets/resnet50-19c8e357.pth'),
                            networks.CLS(2048, 12)).cuda().eval()
model.load_state_dict(last['model'], strict=True)
transform = transforms.Compose([transforms.Resize((256, 256)), transforms.CenterCrop(224), transforms.ToTensor()])
rows = [line.rsplit(None, 1) for line in Path(config['target']).read_text().splitlines() if line.strip()]
raw_labels = [int(row[1]) for row in rows]
logits = []
with torch.no_grad():
    for start in range(0, len(rows), 64):
        images = []
        for name, _ in rows[start:start + 64]:
            with Image.open(Path(config['data_dir']) / name) as image:
                images.append(transform(image.convert('RGB')))
        logits.append(model(torch.stack(images).cuda())[2].cpu())
logits = torch.cat(logits)
assert logits.shape == (564, 12) and torch.isfinite(logits).all()
predictions = logits.argmax(1).clamp_max(10).tolist()
recomputed = macro_open_set_metrics(OFFICE31_A2W, raw_labels, predictions)
for key, logged in (('OS_star', 'OS_star'), ('UNK', 'unknown'), ('HOS', 'HOS')):
    assert abs(recomputed[key] - history[-1][logged]) < 1e-12
np.savez_compressed(run / 'fixed-final-evaluation.npz', raw_logits=logits.numpy(),
                    raw_evaluation_labels=np.array(raw_labels), semantic_predictions=np.array(predictions))
report = dict(task='office31-a2w', seed=1, epochs_verified=70, completed_seed1=True,
              completed_three_seed_matrix=False, fixed_final=history[-1],
              oracle_best=history[oracle_index], oracle_best_uses_target_labels=True,
              independent_fixed_final_metrics=recomputed, resume_state_fields_verified=list(resume_keys),
              checkpoint_sha256={name: hashlib.sha256((run / name).read_bytes()).hexdigest()
                                 for name in ('last.pt', 'warmup-complete.pt', 'best.pt')},
              gpu=torch.cuda.get_device_name(), loss_protocol='published code, not paper equation reimplementation',
              target_labels_used_for_evaluation_only=True, optimizer_steps_executed_by_collector=0)
with (run / 'audit-summary.json').open('x') as stream:
    json.dump(report, stream, indent=2, allow_nan=False)
archives = []
definitions = [('results', [path for path in run.iterdir() if path.is_file() and path.suffix != '.pt']),
               ('warmup', [run / 'warmup-complete.pt']), ('checkpoints', [run / 'last.pt', run / 'best.pt'])]
for label, paths in definitions:
    archive_path = Path('/content') / f'rta-multitask-a2w-seed1-{label}-v1.zip'
    with zipfile.ZipFile(archive_path, 'x', zipfile.ZIP_DEFLATED) as archive:
        for path in paths:
            archive.write(path, path.name)
        if label == 'results':
            for path in code.glob('*.py'):
                archive.write(path, 'code/' + path.name)
            archive.write('/content/rta-multitask-a2w-seed1-console-v1.log', 'console.log')
            archive.write('/content/rta-multitask-a2w-seed1-launch-v1.json', 'launch.json')
    assert archive_path.stat().st_size <= 500 * 1024**2
    item = dict(file=archive_path.name, bytes=archive_path.stat().st_size,
                sha256=hashlib.sha256(archive_path.read_bytes()).hexdigest())
    archives.append(item)
    print('BASELINE_ARCHIVE', json.dumps(item), flush=True)
with Path('/content/rta-multitask-a2w-seed1-archives-v1.json').open('x') as stream:
    json.dump(archives, stream, indent=2)
print('BASELINE_SEED1_AUDIT_PASS', json.dumps(report), flush=True)
