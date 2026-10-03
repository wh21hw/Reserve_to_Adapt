"""Fixed-final semantic/latent occupancy and collapse audit. Fresh process only."""
import hashlib
import json
from pathlib import Path
import sys
import zipfile
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

code=Path('/content/rta-hierarchical-pilot-v1')
sys.path.insert(0,str(code))
import networks
assert Path(networks.__file__).resolve()==code/'networks.py'
from networks import ResNetFc,CLS
sys.path.insert(0,'/content')
from source_warmup_features import Images

root=Path('/content/imp-runs/rta-hierarchical-pilot-v1')
paths=[str(Path('/content/osda-datasets')/line.rsplit(' ',1)[0]) for line in
    Path('/content/webcam_0-9_20-30_test.txt').read_text().splitlines() if line.strip()]
transform=transforms.Compose([transforms.Resize((256,256)),transforms.CenterCrop(224),transforms.ToTensor()])
old=json.loads(Path('/content/imp-runs/rta-capacity-pilot-v1/summary.json').read_text())['arms']
torch.set_num_threads(2)
rows=[]
for index,arm in enumerate(('original2','proto2','proto18')):
    run=root/arm/'a2w_seed1'
    assert json.loads((root/arm/'process-status.json').read_text())['exit_code']==0
    history=[json.loads(line) for line in (run/'history.jsonl').read_text().splitlines()]
    assert [row['epoch'] for row in history]==[5,6]
    checkpoint_path=run/'last.pt'
    checkpoint=torch.load(checkpoint_path,map_location='cpu',weights_only=False)
    assert checkpoint['epoch']==6
    count=18 if arm=='proto18' else 2
    net=torch.nn.Sequential(ResNetFc(model_path='/content/osda-datasets/resnet50-19c8e357.pth'),CLS(2048,10+count)).cuda().eval()
    net.load_state_dict(checkpoint['model'],strict=True)
    raw_rows=[]; semantic_rows=[]; conditional_rows=[]
    with torch.no_grad():
        for images in DataLoader(Images(paths,transform),batch_size=64,shuffle=False,num_workers=0):
            raw=net(images.cuda())[2]
            semantic,conditional=net[1].group_forward(raw)
            raw_rows.append(raw.cpu()); semantic_rows.append(semantic.cpu()); conditional_rows.append(conditional.cpu())
    raw=torch.cat(raw_rows); semantic=torch.cat(semantic_rows); conditional=torch.cat(conditional_rows)
    assert semantic.shape==(564,11) and all(torch.isfinite(x).all() for x in (raw,semantic,conditional))
    assert torch.allclose(conditional.exp().sum(1),torch.ones(564),atol=1e-6,rtol=0)
    unknown=semantic.argmax(1)==10
    occupancy=torch.bincount(conditional[unknown].argmax(1),minlength=count) if unknown.any() else torch.zeros(count,dtype=torch.long)
    directions=torch.nn.functional.normalize(net[1].fc.weight[10:].detach().cpu(),dim=1)
    cosine=(directions@directions.T)[torch.triu(torch.ones(count,count,dtype=torch.bool),diagonal=1)]
    if arm=='original2':
        warm=torch.load('/content/imp-runs/rta-space-warmup-l4-v1/a2w_seed1/last.pt',map_location='cpu',weights_only=False)
        initial_directions=torch.nn.functional.normalize(warm['model']['1.fc.weight'][10:],dim=1)
        del warm
    else:
        proposal=np.load('/content/imp-runs/farthest-constrained-frozen-v1/gate-seed1-original.npz',allow_pickle=False)
        initial_directions=torch.nn.functional.normalize(torch.from_numpy(proposal['candidates'][old[index]['handoff']['head']['selected_indices']]),dim=1)
    initial_cosine=(initial_directions@initial_directions.T)[torch.triu(torch.ones(count,count,dtype=torch.bool),diagonal=1)]
    handoff=json.loads((run/'handoff.json').read_text())
    for key in ('known_weights_sha256','known_momentum_sha256','source_relation_bank_sha256','virtual_sha256','gmm_means','optimizer_steps','grl_steps'):
        assert handoff[key]==old[index]['handoff'][key]
    row=dict(arm=arm,history=history,final=history[-1],handoff=handoff,
        checkpoint_sha256=hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
        occupancy=dict(semantic_unknown_count=int(unknown.sum()),latent_hard_counts_among_predicted_unknown=occupancy.tolist(),
            occupied_latent_slots=int((occupancy>0).sum()),conditional_soft_mass_all_target=conditional.exp().sum(0).tolist(),
            conditional_soft_mass_predicted_unknown=conditional[unknown].exp().sum(0).tolist(),
            mean_semantic_unknown_probability=float(semantic.softmax(1)[:,10].mean()),
            pairwise_head_direction_cosine=dict(mean=float(cosine.mean()),max=float(cosine.max()),min=float(cosine.min())),
            initial_pairwise_head_direction_cosine=dict(mean=float(initial_cosine.mean()),max=float(initial_cosine.max()),min=float(initial_cosine.min())),
            mean_initial_final_direction_cosine=float((initial_directions*directions).sum(1).mean()),
            target_labels_read=False),raw_flat_head_fixed_final=old[index]['final'])
    rows.append(row)
    np.savez_compressed(root/arm/'final-target-predictions.npz',raw_logits=raw.numpy(),semantic_logits=semantic.numpy(),log_conditional=conditional.numpy())
    print('HIERARCHICAL_ARM_RESULTS',json.dumps(row),flush=True)
    del net,checkpoint,images,raw,semantic,conditional,raw_rows,semantic_rows,conditional_rows
    torch.cuda.empty_cache()
with (root/'summary.json').open('x') as stream:
    json.dump(dict(arms=rows,scope='Office31 A2W seed1, two adaptation epochs only',complete_full_baseline=False,
        semantic_unknown_count_inferred=False,selection='Fixed final primary, oracle-best diagnostic only'),stream,indent=2,allow_nan=False)
archive=Path('/content/rta-hierarchical-pilot-v1-results.zip')
with zipfile.ZipFile(archive,'x',zipfile.ZIP_DEFLATED) as bundle:
    for path in root.rglob('*'):
        if path.is_file() and path.suffix!='.pt':
            bundle.write(path,str(path.relative_to(root)))
    for path in code.glob('*.py'):
        bundle.write(path,'code/'+path.name)
    for name in ('hierarchical-unit-v1.json','hierarchical-preflight-v1.json','hierarchical-preflight-console-v1.log',
                 'preflight_hierarchical_worker_v1.py','collect_hierarchical_worker_v1.py','run_hierarchical_pilot_colab.py'):
        bundle.write(Path('/content')/name,name)
print('HIERARCHICAL_RESULTS_ARCHIVE',archive.stat().st_size,hashlib.sha256(archive.read_bytes()).hexdigest(),flush=True)
for arm in ('original2','proto2','proto18'):
    archive=Path('/content')/f'rta-hierarchical-pilot-v1-{arm}-checkpoint.zip'
    with zipfile.ZipFile(archive,'x',zipfile.ZIP_STORED) as bundle:
        bundle.write(root/arm/'a2w_seed1/last.pt','last.pt')
        bundle.write(root/arm/'a2w_seed1/handoff.json','handoff.json')
        bundle.write(root/arm/'audit.json','audit.json')
    print('HIERARCHICAL_CHECKPOINT_ARCHIVE',arm,archive.stat().st_size,hashlib.sha256(archive.read_bytes()).hexdigest(),flush=True)
