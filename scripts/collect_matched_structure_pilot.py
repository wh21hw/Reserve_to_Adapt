"""Independent single-arm final checkpoint audit/evaluation; never train."""
import argparse
import hashlib
import json
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
sys.path.insert(0, '/content')
from hierarchical_unknown import marginalize_unknown
from matched_pilot_audit import validate_logs

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--arm', choices=('structure_off', 'structure_on'), required=True)
    args = parser.parse_args()
    assert Path(networks.__file__).resolve() == code / 'networks.py'
    run = Path('/content/imp-runs/matched-structure-pilot-v1') / args.arm
    config = json.loads((run / 'config.json').read_text())
    summary = json.loads((run / 'summary.json').read_text())
    history = [json.loads(line) for line in (run / 'history.jsonl').read_text().splitlines()]
    batches = [json.loads(line) for line in (run / 'batches.jsonl').read_text().splitlines()]
    audit = validate_logs(config, history, batches, summary)
    assert config['arm'] == args.arm
    assert digest(run / 'last.pt') == summary['checkpoint_sha256']
    for filename, expected in config['input_sha256'].items():
        assert digest(filename) == expected
    for filename, expected in config['module_sha256'].items():
        assert digest(Path('/content') / filename) == expected
    checkpoint = torch.load(run / 'last.pt', map_location='cpu', weights_only=False)
    assert checkpoint['epoch'] == 6 and checkpoint['config'] == config
    assert checkpoint['optimizer_steps'] == [84] * 3 and checkpoint['grl_steps'] == 168
    for field in ('model', 'discriminator'):
        assert all(torch.isfinite(value).all() for value in checkpoint[field].values())
    for field in ('source_relation_bank', 'target_relation_bank', 'virtual_templates', 'teacher', 'structure_gate'):
        assert torch.isfinite(checkpoint[field]).all()
    for field in ('optimizer_feature', 'optimizer_cls', 'optimizer_discriminator'):
        assert all(torch.isfinite(value).all() for state in checkpoint[field]['state'].values() for value in state.values())
    assert checkpoint['teacher'].shape == (564, config['capacity'])
    assert checkpoint['structure_gate'].shape == (564,)
    assert np.isfinite(checkpoint['relation_mixture'].means_).all()
    torch.set_num_threads(2)
    model = torch.nn.Sequential(networks.ResNetFc(model_path='/content/osda-datasets/resnet50-19c8e357.pth'),
                                networks.CLS(2048, 10 + config['capacity'])).cuda()
    prior = torch.tensor(config['prior'], dtype=torch.float32, device='cuda')
    model[1].register_buffer('unknown_log_weights', prior)
    model.load_state_dict(checkpoint['model'], strict=True)
    assert torch.equal(model[1].unknown_log_weights, prior)
    model.eval()
    transform = transforms.Compose([transforms.Resize((256, 256)), transforms.CenterCrop(224), transforms.ToTensor()])
    rows = [line.rsplit(None, 1) for line in Path('/content/webcam_0-9_20-30_test.txt').read_text().splitlines() if line.strip()]
    predictions, logits = [], []
    with torch.no_grad():
        for start in range(0, len(rows), 64):
            images = []
            for name, _ in rows[start:start + 64]:
                with Image.open(Path('/content/osda-datasets') / name) as image:
                    images.append(transform(image.convert('RGB')))
            raw = model(torch.stack(images).cuda())[2]
            semantic, _ = marginalize_unknown(raw, 10, model[1].unknown_log_weights)
            logits.append(raw.cpu().numpy())
            predictions.extend(semantic.argmax(1).cpu().tolist())
    labels = [int(row[1]) for row in rows]
    recomputed = macro_open_set_metrics(OFFICE31_A2W, labels, predictions)
    for key in ('OS_star', 'UNK', 'HOS'):
        assert abs(recomputed[key] - history[-1]['metrics'][key]) < 1e-12
    np.savez_compressed(run / 'independent-final-evaluation.npz', logits=np.concatenate(logits),
                        evaluation_only_target_labels=np.array(labels), predictions=np.array(predictions))
    audit.update(independent_checkpoint_evaluation_verified=True, independent_final_metrics=recomputed,
                 checkpoint_sha256=summary['checkpoint_sha256'], collector_optimizer_steps=0,
                 complete_budget=False, target_labels_used_for_evaluation_only=True)
    with (run / 'independent-audit.json').open('x') as stream:
        json.dump(audit, stream, indent=2, allow_nan=False)
    for label, files in [('results', [p for p in run.iterdir() if p.is_file() and p.suffix != '.pt']),
                         ('checkpoint', [run / 'last.pt'])]:
        archive_path = Path('/content') / f'matched-structure-pilot-v1-{args.arm}-{label}.zip'
        with zipfile.ZipFile(archive_path, 'x', zipfile.ZIP_DEFLATED) as archive:
            for path in files:
                archive.write(path, path.name)
        assert archive_path.stat().st_size < 500 * 1024**2
        print('MATCHED_COLLECT_ARCHIVE', json.dumps(dict(file=archive_path.name,
              bytes=archive_path.stat().st_size, sha256=digest(archive_path))), flush=True)
    print('MATCHED_COLLECT_PASS', json.dumps(audit), flush=True)

if __name__ == '__main__':
    main()
