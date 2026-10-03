"""Fresh full-budget final checkpoint audit; no SGD or target-based selection."""
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
from matched_full_audit import validate_full_logs
from matched_full_budget import build, PILOT_SHA256

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def finite(value):
    if isinstance(value, torch.Tensor):
        assert torch.isfinite(value).all()
    elif isinstance(value, np.ndarray):
        assert np.isfinite(value).all()
    elif isinstance(value, dict):
        for child in value.values(): finite(child)
    elif isinstance(value, (list, tuple)):
        for child in value: finite(child)
    elif isinstance(value, (float, np.floating)):
        assert np.isfinite(value)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--arm', choices=('structure_off','structure_on'), required=True)
    args = parser.parse_args()
    assert Path(networks.__file__).resolve() == code / 'networks.py'
    assert torch.cuda.is_available() and 'L4' in torch.cuda.get_device_name()
    run = Path('/content/imp-runs/matched-structure-full-a2w-seed1-v1') / args.arm
    config = json.loads((run / 'config.json').read_text())
    summary = json.loads((run / 'summary.json').read_text())
    history = [json.loads(line) for line in (run / 'history.jsonl').read_text().splitlines()]
    batches = [json.loads(line) for line in (run / 'batches.jsonl').read_text().splitlines()]
    audit = validate_full_logs(config, history, batches, summary)
    assert config['arm'] == args.arm
    for path, expected in (config['input_sha256'] | config['data_list_sha256']).items():
        assert digest(path) == expected
    for filename, expected in config['module_sha256'].items():
        assert digest(Path('/content') / filename) == expected
    assert digest('/content/train_matched_full_a2w_seed1.py') == config['worker_sha256']
    assert digest('/content/matched_full_budget.py') == 'e764da6a387160c5c98d906c1e9aed0dbd1afafb2e70458a443fe072c80852e3'
    assert digest(code / 'main.py') == '683c35e76a89ad1588f35dea83e3ecbc6f9a8af5bb3250e9e3de1cbfc3c7a148'
    assert digest(code / 'utilities.py') == '6ba276bb2d0076a4d64cde811496d2dfb71b330c8e1660add02d8fbd42e636e7'
    assert config['base_pilot_sha256'] == PILOT_SHA256
    source = Path('/content/train_matched_structure_pilot_fixed_v2.py').read_bytes().decode()
    assert hashlib.sha256(build(source).encode()).hexdigest() == config['generated_training_sha256']
    assert digest(run / 'last.pt') == summary['checkpoint_sha256']
    checkpoint = torch.load(run / 'last.pt', map_location='cpu', weights_only=False)
    assert checkpoint['epoch'] == 70 and checkpoint['config'] == config
    assert checkpoint['optimizer_steps'] == [980]*3 and checkpoint['grl_steps'] == 1960
    for name in ('model','discriminator','optimizer_feature','optimizer_cls','optimizer_discriminator',
                 'source_relation_bank','target_relation_bank','virtual_templates','teacher','structure_gate'):
        finite(checkpoint[name])
    assert checkpoint['teacher'].shape == (564, config['capacity'])
    assert checkpoint['structure_gate'].shape == (564,)
    assert checkpoint['source_relation_bank'].shape == (10,10)
    assert checkpoint['virtual_templates'].shape == (10,256)
    assert (checkpoint['structure_gate'] >= 0).all() and (checkpoint['structure_gate'] <= 1).all()
    assert torch.allclose(checkpoint['teacher'].exp().sum(1), torch.ones(564), atol=1e-5)
    mixture = checkpoint['relation_mixture']
    assert mixture.n_components == 2
    for name in ('means_','weights_','covariances_','precisions_','precisions_cholesky_'):
        finite(getattr(mixture,name))
    warm = torch.load('/content/imp-runs/rta-multitask-baseline-v1/office31-a2w_seed1/warmup-complete.pt',
                      map_location='cpu', weights_only=False)
    assert warm['epoch'] == 4 and warm['optimizer_steps'] == [56]*3 and warm['grl_steps'] == 112
    torch.set_num_threads(2)
    model = torch.nn.Sequential(networks.ResNetFc(model_path='/content/osda-datasets/resnet50-19c8e357.pth'),
                                networks.CLS(2048,10+config['capacity'])).cuda()
    prior = torch.tensor(config['prior'],dtype=torch.float32,device='cuda')
    model[1].register_buffer('unknown_log_weights',prior)
    model.load_state_dict(checkpoint['model'], strict=True)
    assert torch.equal(model[1].unknown_log_weights,prior)
    model.eval()
    transform = transforms.Compose([transforms.Resize((256,256)), transforms.CenterCrop(224), transforms.ToTensor()])
    rows = [line.rsplit(None,1) for line in Path('/content/webcam_0-9_20-30_test.txt').read_text().splitlines() if line.strip()]
    predictions, logits = [], []
    with torch.no_grad():
        for start in range(0,len(rows),64):
            images = []
            for name,_ in rows[start:start+64]:
                with Image.open(Path('/content/osda-datasets') / name) as image:
                    images.append(transform(image.convert('RGB')))
            raw = model(torch.stack(images).cuda())[2]
            semantic,_ = marginalize_unknown(raw,10,model[1].unknown_log_weights)
            logits.append(raw.cpu().numpy())
            predictions.extend(semantic.argmax(1).cpu().tolist())
    labels = [int(row[1]) for row in rows]
    recomputed = macro_open_set_metrics(OFFICE31_A2W,labels,predictions)
    for key in ('OS_star','UNK','HOS'):
        assert abs(recomputed[key]-history[-1]['metrics'][key]) < 1e-12
    np.savez_compressed(run/'independent-final-evaluation.npz',logits=np.concatenate(logits),
                        evaluation_only_target_labels=np.array(labels),predictions=np.array(predictions))
    audit.update(independent_checkpoint_evaluation_verified=True,independent_final_metrics=recomputed,
                 checkpoint_sha256=summary['checkpoint_sha256'],collector_optimizer_steps=0,
                 complete_budget=True, warm_checkpoint_epoch_verified=4, final_epoch_verified=70,
                 target_label_checkpoint_selection=False,collector_sha256=digest(__file__))
    with (run/'independent-audit.json').open('x') as stream:
        json.dump(audit,stream,indent=2,allow_nan=False)
    manifest = []
    for label,files in [('results',[p for p in run.iterdir() if p.is_file() and p.suffix!='.pt']),('checkpoint',[run/'last.pt'])]:
        path = Path('/content')/f'matched-full-a2w-seed1-v1-{args.arm}-{label}.zip'
        with zipfile.ZipFile(path,'x',zipfile.ZIP_DEFLATED) as archive:
            for file in files: archive.write(file,file.name)
        assert path.stat().st_size < 500*1024**2
        manifest.append(dict(file=path.name,bytes=path.stat().st_size,sha256=digest(path)))
    with (run/'archive-manifest.json').open('x') as stream:
        json.dump(manifest,stream,indent=2)
    print('FULL_COLLECT_PASS',json.dumps(dict(audit=audit,archives=manifest)),flush=True)

if __name__=='__main__': main()
