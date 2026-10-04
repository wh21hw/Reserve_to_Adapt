"""Compare original ImageNet ResNet and existing source-tuned features, source only."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from torchvision import models,transforms
from sklearn.metrics import roc_auc_score,average_precision_score

ROOT=Path('/content/imp-runs/resnet-pretraining-diagnostic-v1')
DATA=Path('/content/osda-officehome-pr2rw-v1')
PRIOR=Path('/content/imp-runs/source-leaveclass-officehome-v1/seed1')
def records():
    return [r.rsplit(None,1) for r in (DATA/'product_0-24_train_all.txt').read_text().splitlines() if r.strip()]

def extract(smoke):
    torch.set_num_threads(2)
    net=models.resnet50(weights=None)
    # Existing official torchvision checkpoint uses the old tar format.
    # Explicit legacy loading is limited to this already-used trusted weight file.
    net.load_state_dict(torch.load('/content/osda-datasets/resnet50-19c8e357.pth',map_location='cpu',weights_only=False))
    net.fc=torch.nn.Identity();net.eval().requires_grad_(False).cuda()
    transform=transforms.Compose([transforms.Resize((256,256)),transforms.CenterCrop(224),transforms.ToTensor(),
        transforms.Normalize((.485,.456,.406),(.229,.224,.225))])
    rows=records();chosen=rows[:8] if smoke else rows
    if not smoke and (ROOT/'features.npz').exists():raise FileExistsError('Preserve previous extraction')
    chunks=[]
    with torch.inference_mode():
        for start in range(0,len(chosen),64):
            batch=[]
            for name,_ in chosen[start:start+64]:
                with Image.open(DATA/name) as im:batch.append(transform(im.convert('RGB')))
            x=torch.nn.functional.normalize(net(torch.stack(batch).cuda()),dim=1)
            if x.shape[1]!=2048 or not torch.isfinite(x).all():raise RuntimeError('Invalid ResNet feature')
            chunks.append(x.cpu().numpy())
            if smoke or start%320==0:print('EXTRACT',min(start+64,len(chosen)),len(chosen),flush=True)
    x=np.concatenate(chunks)
    if smoke:print('SMOKE_OK',x.shape,flush=True);return
    ROOT.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(ROOT/'features.npz',features=x,labels=np.array([int(r[1]) for r in rows]))
    (ROOT/'config.json').write_text(json.dumps(dict(model='resnet50',weights='existing RTA resnet50-19c8e357.pth',
        trained=False,feature_dim=2048,source_only=True,torch=torch.__version__,gpu=torch.cuda.get_device_name(0),
        preprocess='Resize256x256, CenterCrop224, ImageNet normalize, L2 feature normalize'),indent=2))
    print('EXTRACTION_DONE',x.shape,flush=True)

def diagnose():
    output=ROOT/'comparison.json'
    if output.exists():raise FileExistsError('Preserve previous report')
    y=np.array([int(r[1]) for r in records()]);rng=np.random.RandomState(2026)
    train,calibration,test=[],[],[]
    for c in range(25):
        ids=np.flatnonzero(y==c);rng.shuffle(ids);cut=max(1,int(.7*len(ids)));middle=cut+(len(ids)-cut)//2
        train.extend(ids[:cut]);calibration.extend(ids[cut:middle]);test.extend(ids[middle:])
    train,calibration,test=map(np.asarray,(train,calibration,test));known=[c for c in range(25) if c not in range(10,15)]
    tr=train[np.isin(y[train],known)];cr=calibration[np.isin(y[calibration],known)];hidden=np.isin(y[test],range(10,15))
    arms=[('imagenet_resnet50',ROOT/'features.npz','features'),('source_tuned_resnet50_backbone',PRIOR/'backbone-features.npz','target'),
        ('source_tuned_resnet50_bottleneck',PRIOR/'source/features.npz','target'),
        ('external_dinov2_diagnostic',Path('/content/imp-runs/dinov2-source-diagnostic-v1/features.npz'),'features')]
    report=dict(source_only=True,target_used=False,formal_RTA_changed=False,arms={},
        caveat='Frozen source-only posthoc stress-block diagnosis. Known test images participated in source tuning; native Torch extraction for ImageNet/DINO vs cached legacy source features. DINO uses extra pretraining and distinct preprocessing. Not RTA improvement or causal proof.')
    for name,path,key in arms:
        x=np.load(path)[key].astype(np.float64)
        if len(x)!=len(y) or not np.isfinite(x).all():raise ValueError('Feature mismatch')
        anchors=np.stack([x[tr][y[tr]==c].mean(0) for c in known])
        def distance(ids):return ((x[ids,None,:]-anchors[None,:,:])**2).sum(2).min(1)
        scores=distance(test);threshold=float(np.quantile(distance(cr),.99));selected=scores>threshold
        report['arms'][name]=dict(feature_dim=x.shape[1],AUROC=float(roc_auc_score(hidden,scores)),AP=float(average_precision_score(hidden,scores)),
            threshold99=threshold,known_false_candidate_rate=float(selected[~hidden].mean()),hidden_recall=float(selected[hidden].mean()),
            known_false_count=int(selected[~hidden].sum()),hidden_selected_count=int(selected[hidden].sum()),
            per_hidden_class_recall={str(c):float(selected[y[test]==c].mean()) for c in range(10,15)})
    output.write_text(json.dumps(report,indent=2,allow_nan=False));print('RESNET_PRETRAINING_DIAGNOSIS',json.dumps(report),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['smoke','extract','diagnose']);a=p.parse_args()
    if a.stage=='diagnose':diagnose()
    else:extract(a.stage=='smoke')
