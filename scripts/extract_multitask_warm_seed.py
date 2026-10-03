"""Export seed2/3 own fixed warm4 inputs; execute only with GPU idle."""
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
sys.path.insert(0,str(code))
import networks

def digest(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed',type=int,choices=(2,3),required=True)
    args = parser.parse_args()
    assert Path(networks.__file__).resolve()==code/'networks.py'
    assert torch.cuda.is_available() and 'L4' in torch.cuda.get_device_name()
    expected = {2:'e16e895e1bdaec17f14dd4eda7cc81f8b05ad1c1c6d76145bbd3088aa1d2fea0',
                3:'f74f473b8d5ae58bac64916db8dc5ad153ba2b54c5e4688227c460a96956152b'}
    run = Path(f'/content/imp-runs/rta-multitask-baseline-v1/office31-a2w_seed{args.seed}')
    checkpoint_path = run/'warmup-complete.pt'
    assert digest(checkpoint_path)==expected[args.seed]
    audit = json.loads((run/'audit-summary.json').read_text())
    assert audit['epochs_verified']==70 and audit['completed_seed'] and audit['seed']==args.seed
    config = json.loads((run/'config.json').read_text())
    launch = json.loads(Path(f'/content/rta-multitask-a2w-seed{args.seed}-launch-v1.json').read_text())
    assert launch['seed']==args.seed and launch['epochs']==70
    for name,sha in launch['code_sha256'].items(): assert digest(code/name)==sha
    output = Path(f'/content/imp-runs/multitask-a2w-seed{args.seed}-warm-features-v1')
    assert not output.exists(),'Refusing overwrite'
    checkpoint = torch.load(checkpoint_path,map_location='cpu',weights_only=False)
    assert checkpoint['epoch']==4 and checkpoint['optimizer_steps']==[56]*3 and checkpoint['grl_steps']==112
    torch.set_num_threads(2)
    net = torch.nn.Sequential(networks.ResNetFc(model_path='/content/osda-datasets/resnet50-19c8e357.pth'),
                              networks.CLS(2048,12)).cuda().eval()
    net.load_state_dict(checkpoint['model'],strict=True)
    transform = transforms.Compose([transforms.Resize((256,256)),transforms.CenterCrop(224),transforms.ToTensor()])
    @torch.no_grad()
    def extract(path):
        # Only filenames: target labels are not part of representation inputs.
        names = [line.rsplit(None,1)[0] for line in Path(path).read_text().splitlines() if line.strip()]
        features,logits=[],[]
        for start in range(0,len(names),64):
            images=[]
            for name in names[start:start+64]:
                with Image.open(Path(config['data_dir'])/name) as image:
                    images.append(transform(image.convert('RGB')))
            _,feature,logit,_=net(torch.stack(images).cuda())
            features.append(feature.cpu().numpy()); logits.append(logit.cpu().numpy())
        return np.concatenate(features),np.concatenate(logits)
    source,source_logits=extract(config['source'])
    target,target_logits=extract(config['target'])
    assert source.shape==(958,256) and target.shape==(564,256)
    assert all(np.isfinite(v).all() for v in (source,target,source_logits,target_logits))
    labels=lambda path: np.array([int(line.rsplit(None,1)[1]) for line in Path(path).read_text().splitlines() if line.strip()])
    source_labels=labels(config['source'])
    assert np.array_equal(np.unique(source_labels),np.arange(10))
    output.mkdir(parents=True)
    np.savez_compressed(output/'features.npz',source=source,target=target,source_labels=source_labels,
                        source_logits=source_logits,target_logits=target_logits)
    np.savez_compressed(output/'evaluation-only.npz',target_labels=labels(config['target']))
    manifest=dict(task='office31-a2w',seed=args.seed,epoch=4,checkpoint_sha256=expected[args.seed],
                  target_labels_in_features=False,optimizer_steps_executed=0,
                  selection='fixed warmup-complete; target-selected best excluded',code_sha256=launch['code_sha256'],
                  list_sha256={config[k]:digest(config[k]) for k in ('source','target')},
                  files={name:digest(output/name) for name in ('features.npz','evaluation-only.npz')})
    with (output/'manifest.json').open('x') as stream: json.dump(manifest,stream,indent=2,allow_nan=False)
    path=Path(f'/content/multitask-a2w-seed{args.seed}-warm-features-v1.zip')
    with zipfile.ZipFile(path,'x',zipfile.ZIP_DEFLATED) as archive:
        for file in output.iterdir(): archive.write(file,file.name)
        archive.write(__file__,'export-worker.py')
    print('OWN_SEED_WARM_FEATURE_EXPORT_PASS',json.dumps(dict(manifest=manifest,archive=path.name,sha256=digest(path))),flush=True)

if __name__=='__main__': main()
