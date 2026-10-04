"""Frozen external encoder diagnosis only; never supplies K to formal RTA."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from sklearn.metrics import roc_auc_score, average_precision_score

ROOT = Path('/content/imp-runs/dinov2-source-diagnostic-v1')
DATA = Path('/content/osda-officehome-pr2rw-v1')
PRIOR = Path('/content/imp-runs/source-leaveclass-officehome-v1/seed1/source/features.npz')

def rows():
    return [r.rsplit(None, 1) for r in (DATA/'product_0-24_train_all.txt').read_text().splitlines() if r.strip()]

def extract(smoke):
    torch.set_num_threads(2)
    model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14', trust_repo=True)
    model.eval().requires_grad_(False).cuda()
    transform = transforms.Compose([
        transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224), transforms.ToTensor(),
        transforms.Normalize((.485,.456,.406),(.229,.224,.225))])
    records = rows()
    output = ROOT/'features.npz'
    if not smoke and output.exists():
        raise FileExistsError('Preserve previous extraction')
    chosen = records[:8] if smoke else records
    features = []
    with torch.inference_mode():
        for start in range(0,len(chosen),32):
            batch=[]
            for name,_ in chosen[start:start+32]:
                with Image.open(DATA/name) as im:
                    batch.append(transform(im.convert('RGB')))
            z=torch.nn.functional.normalize(model(torch.stack(batch).cuda()),dim=1)
            if z.shape[1] != 384 or not torch.isfinite(z).all():
                raise RuntimeError('Invalid DINO feature')
            features.append(z.cpu().numpy())
            if smoke or start % 320 == 0:
                print('EXTRACT',min(start+32,len(chosen)),len(chosen),flush=True)
    x=np.concatenate(features)
    if smoke:
        print('SMOKE_OK',x.shape,flush=True)
        return
    ROOT.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(output,features=x,labels=np.array([int(r[1]) for r in records]),paths=np.array([r[0] for r in records]))
    config=dict(model='dinov2_vits14',frozen=True,pretraining='LVD-142M',
                use='source-only diagnostic, not RTA encoder or formal K estimator',
                images=len(records),feature_dim=384,batch_size=32,gpu=torch.cuda.get_device_name(0),
                torch=torch.__version__,preprocess='Resize256 bicubic, CenterCrop224, ImageNet normalize, L2 feature normalize')
    (ROOT/'config.json').write_text(json.dumps(config,indent=2))
    print('EXTRACTION_DONE',json.dumps(config),flush=True)

def diagnose():
    output=ROOT/'comparison.json'
    if output.exists():
        raise FileExistsError('Preserve previous diagnosis')
    cache=np.load(ROOT/'features.npz')
    y=cache['labels']; expected=np.array([int(r[1]) for r in rows()])
    if not np.array_equal(y,expected):
        raise ValueError('Source order mismatch')
    rng=np.random.RandomState(2026)
    train,calibration,test=[],[],[]
    for c in range(25):
        ids=np.flatnonzero(y==c); rng.shuffle(ids)
        cut=max(1,int(.7*len(ids))); middle=cut+(len(ids)-cut)//2
        train.extend(ids[:cut]); calibration.extend(ids[cut:middle]); test.extend(ids[middle:])
    train,calibration,test=map(np.asarray,(train,calibration,test))
    known=[c for c in range(25) if c not in range(10,15)]
    tr=train[np.isin(y[train],known)]; cr=calibration[np.isin(y[calibration],known)]
    hidden=np.isin(y[test],range(10,15))
    report=dict(source_only=True,real_target_used=False,formal_RTA_changed=False,
        hidden_classes=list(range(10,15)),test_count=len(test),known_count=int((~hidden).sum()),hidden_count=int(hidden.sum()),
        caveat='Posthoc stress block; ResNet known test images participated in supervision; DINO has extra pretraining and distinct preprocessing. This is not a fair RTA improvement comparison.',arms={})
    for name,x in [('resnet50_bottleneck',np.load(PRIOR)['target']),('dinov2_vits14',cache['features'])]:
        x=x.astype(np.float64)
        if len(x)!=len(y):
            raise ValueError('Feature row mismatch')
        anchors=np.stack([x[tr][y[tr]==c].mean(0) for c in known])
        def distance(ids):
            return ((x[ids,None,:]-anchors[None,:,:])**2).sum(2).min(1)
        score=distance(test); threshold=float(np.quantile(distance(cr),.99)); selected=score>threshold
        report['arms'][name]=dict(feature_dim=x.shape[1],AUROC=float(roc_auc_score(hidden,score)),
            average_precision=float(average_precision_score(hidden,score)),source99_threshold=threshold,
            known_false_candidate_rate=float(selected[~hidden].mean()),hidden_recall=float(selected[hidden].mean()),
            per_hidden_class_recall={str(c):float(selected[y[test]==c].mean()) for c in range(10,15)},
            known_quantiles=np.quantile(score[~hidden],[.1,.5,.9,.99]).tolist(),hidden_quantiles=np.quantile(score[hidden],[.1,.5,.9,.99]).tolist())
    output.write_text(json.dumps(report,indent=2,allow_nan=False))
    print('DINO_SOURCE_DIAGNOSIS',json.dumps(report),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('stage',choices=['smoke','extract','diagnose']); a=p.parse_args()
    if a.stage=='diagnose': diagnose()
    else: extract(a.stage=='smoke')
