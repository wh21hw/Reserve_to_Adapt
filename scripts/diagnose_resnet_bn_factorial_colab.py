"""Source-only 2x2 frozen intervention on ResNet weights and BN buffers."""
import argparse
import json
import sys
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.metrics import roc_auc_score,average_precision_score

sys.path.insert(0,'/content/rta-legacy-l4-bridge-v1')
from networks import ResNetFc,CLS
from train_source_prior_konly import Images

ROOT=Path('/content/imp-runs/resnet-bn-factorial-v1')
OLD=Path('/content/imp-runs/source-leaveclass-officehome-v1/seed1')
DATA=Path('/content/osda-officehome-pr2rw-v1')
ARMS=[('original_weights_original_bn',False,False),('tuned_weights_original_bn',True,False),
      ('original_weights_tuned_bn',False,True),('tuned_weights_tuned_bn',True,True)]
def records():
    return [r.rsplit(None,1) for r in (DATA/'product_0-24_train_all.txt').read_text().splitlines() if r.strip()]
def is_bn_buffer(key):
    return key.endswith(('running_mean','running_var','num_batches_tracked'))

def extract(smoke):
    torch.set_num_threads(2)
    state=torch.load(str(OLD/'source/source-final.pt'),map_location='cpu')['model']
    original=ResNetFc(model_path='/content/osda-datasets/resnet50-19c8e357.pth').state_dict()
    tuned={key[2:]:value for key,value in state.items() if key.startswith('0.')}
    if set(original)!=set(tuned):raise ValueError('Encoder state keys mismatch')
    rows=records();names=[row[0] for row in rows]
    if not smoke:
        ROOT.mkdir(parents=True,exist_ok=False)
    fixed=transforms.Compose([transforms.Resize((256,256)),transforms.CenterCrop(224),transforms.ToTensor()])
    loader=DataLoader(Images(names[:8] if smoke else names,DATA,fixed),batch_size=64,shuffle=False,num_workers=4,pin_memory=True)
    nets={};chunks={}
    for name,use_weights,use_bn in ARMS:
        net=torch.nn.Sequential(ResNetFc(model_path='/content/osda-datasets/resnet50-19c8e357.pth'),CLS(2048,20))
        net.load_state_dict(state)
        encoder={key:(tuned if (use_bn if is_bn_buffer(key) else use_weights) else original)[key] for key in original}
        net[0].load_state_dict(encoder)
        nets[name]=net.cuda().eval();chunks[name]={'backbone':[],'bottleneck':[],'prediction':[]}
    with torch.no_grad():
        for step,images in enumerate(loader):
            images=images.cuda()
            for name,net in nets.items():
                raw=net[0](images);values=net[1](raw)
                back=torch.nn.functional.normalize(raw,dim=1,eps=1e-8)
                if back.shape[1]!=2048 or values[1].shape[1]!=256 or not torch.isfinite(back).all() or not torch.isfinite(values[1]).all():
                    raise RuntimeError('Invalid intervention features')
                chunks[name]['backbone'].append(back.cpu().numpy())
                chunks[name]['bottleneck'].append(values[1].cpu().numpy())
                chunks[name]['prediction'].append(values[2].argmax(1).cpu().numpy())
            if step%8==0:print('BN_FACTORIAL_EXTRACT',step*64+len(images),len(names[:8] if smoke else names),flush=True)
    if smoke:
        print('BN_FACTORIAL_CHECK_OK',list(nets),flush=True);return
    for name,parts in chunks.items():
        np.savez_compressed(ROOT/(name+'.npz'),**{key:np.concatenate(value) for key,value in parts.items()})
    config=dict(source_only=True,real_target_used=False,trained=False,source_checkpoint=str(OLD/'source/source-final.pt'),
        arms=ARMS,bn_intervention='Only running_mean, running_var, num_batches_tracked of encoder. BN affine parameters belong to weight factor.',
        classifier='Same source-final C20 head, weights and BN for every arm; not retrained for intervention',
        torch=torch.__version__,gpu=torch.cuda.get_device_name(0),images=len(rows),
        preprocessing='Original RTA Resize256x256/CenterCrop224; encoder internal ImageNet normalize; L2 features')
    (ROOT/'config.json').write_text(json.dumps(config,indent=2));print('BN_FACTORIAL_EXTRACT_DONE',flush=True)

def score():
    output=ROOT/'comparison.json'
    if output.exists():raise FileExistsError('Preserve previous scoring')
    y=np.array([int(row[1]) for row in records()]);known=[c for c in range(25) if c not in range(10,15)]
    mapping={c:i for i,c in enumerate(known)};rng=np.random.RandomState(2026);train,calibration,test=[],[],[]
    for c in range(25):
        ids=np.flatnonzero(y==c);rng.shuffle(ids);cut=max(1,int(.7*len(ids)));middle=cut+(len(ids)-cut)//2
        train.extend(ids[:cut]);calibration.extend(ids[cut:middle]);test.extend(ids[middle:])
    train,calibration,test=map(np.asarray,(train,calibration,test));tr=train[np.isin(y[train],known)];cr=calibration[np.isin(y[calibration],known)]
    hidden=np.isin(y[test],range(10,15));kt=test[~hidden]
    report=dict(source_only=True,target_used=False,trained=False,arms={},
        caveat='Frozen local intervention on one checkpoint, not trained BN-freeze ablation or RTA performance. Head fixed to source-tuned head. Posthoc pressure block; known test images were source supervised.')
    for name,_,_ in ARMS:
        cache=np.load(ROOT/(name+'.npz'));arm=dict(known_fixed_head_accuracy=float((cache['prediction'][kt]==np.array([mapping[int(c)] for c in y[kt]])).mean()),scores={})
        for key in ['backbone','bottleneck']:
            x=cache[key].astype(np.float64)
            if len(x)!=len(y):raise ValueError('Source count mismatch')
            anchors=np.stack([x[tr][y[tr]==c].mean(0) for c in known])
            def distance(ids):return ((x[ids,None,:]-anchors[None,:,:])**2).sum(2).min(1)
            scores=distance(test);threshold=float(np.quantile(distance(cr),.99));chosen=scores>threshold
            arm['scores'][key]=dict(AUROC=float(roc_auc_score(hidden,scores)),AP=float(average_precision_score(hidden,scores)),threshold99=threshold,
                known_false_count=int(chosen[~hidden].sum()),hidden_selected_count=int(chosen[hidden].sum()),
                known_false_candidate_rate=float(chosen[~hidden].mean()),hidden_recall=float(chosen[hidden].mean()))
        report['arms'][name]=arm
    output.write_text(json.dumps(report,indent=2,allow_nan=False));print('BN_FACTORIAL_DIAGNOSIS',json.dumps(report),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['smoke','extract','score']);a=p.parse_args()
    if a.stage=='score':score()
    else:extract(a.stage=='smoke')
