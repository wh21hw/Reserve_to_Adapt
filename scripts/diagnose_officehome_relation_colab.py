"""Rebuilt frozen final gate probe; checkpoints omit historical bank/mixture."""
import json
from pathlib import Path
import sys
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.mixture import BayesianGaussianMixture
sys.path.insert(0,'/content/rta-legacy-l4-bridge-v1')
sys.path.insert(0,'/content')
from networks import ResNetFc,CLS
from alternating_konly import _FrozenImages

root=Path('/content/imp-runs/officehome-capacity-10e-v1')
output=root/'relation-gate-diagnostic.json'
if output.exists():
    raise FileExistsError('Preserve diagnostic')
torch.set_num_threads(2)
rows=[x.rsplit(None,1) for x in Path('/content/osda-officehome-pr2rw-v1/real_world_0-64_test.txt').read_text().splitlines() if x.strip()]
y=np.array([int(x[1]) for x in rows])
loader=DataLoader(_FrozenImages([x[0] for x in rows],'/content/osda-officehome-pr2rw-v1'),batch_size=64,shuffle=False,num_workers=4)
source_rows=[x.rsplit(None,1) for x in Path('/content/osda-officehome-pr2rw-v1/product_0-24_train_all.txt').read_text().splitlines() if x.strip()]
source_labels=np.array([int(x[1]) for x in source_rows])
source_loader=DataLoader(_FrozenImages([x[0] for x in source_rows],'/content/osda-officehome-pr2rw-v1'),batch_size=64,shuffle=False,num_workers=4)
report=dict(scope='Rebuilt final eval-center-crop source bank and four-component Bayesian GMM; not saved/historical gate or actual augmented minibatches',
            labels='Only diagnostic scoring, no threshold/K/model updates',arms={})
for arm,k in [('fixed4',4),('estimated',1)]:
    state=torch.load(str(root/arm/'officehome-pr2rw_seed1/last.pt'),map_location='cpu')
    if state['epoch']!=10:
        raise ValueError('Expected final10')
    extractor=ResNetFc(model_path='/content/osda-datasets/resnet50-19c8e357.pth')
    net=torch.nn.Sequential(extractor,CLS(extractor.output_num(),25+k,bottle_neck_dim=256)).cuda().eval()
    net.load_state_dict(state['model'])
    source_probs=[]
    with torch.no_grad():
        for images in source_loader:
            logits=net(images.cuda())[2]
            source_probs.append(logits[:,:25].softmax(1).cpu())
    source_probs=torch.cat(source_probs)
    bank=torch.stack([source_probs[source_labels==c].mean(0) for c in range(25)]).cuda()
    scores=[]; predictions=[]
    with torch.no_grad():
        for images in loader:
            _,_,logits,p=net(images.cuda())
            predicted=p[:,:25].argmax(1)
            # Same KL direction as published gate; stable log-softmax.
            score=torch.nn.functional.kl_div(logits[:,:25].log_softmax(1),bank[predicted],reduction='none').sum(1)
            scores.append(score.cpu().numpy());predictions.append(p.argmax(1).cpu().numpy())
    score=np.concatenate(scores);prediction=np.concatenate(predictions)
    if not np.isfinite(score).all():
        raise ValueError('Nonfinite relation score')
    mixture=BayesianGaussianMixture(n_components=4,max_iter=800,random_state=2026).fit(score[:,None])
    known_component=int(np.argmin(mixture.means_))
    component=mixture.predict(score[:,None]);eligible=component!=known_component
    # Reproduce the epoch<=10 global batch-top16 fallback with one fixed
    # label-independent shuffle. No sweep/selection over shuffle seeds.
    permutation=np.random.RandomState(2026).permutation(len(y))
    selected=np.zeros(len(y),dtype=bool)
    for start in range(0,len(y)-63,64):
        idx=permutation[start:start+64];candidate=idx[eligible[idx]]
        if len(candidate)>16:
            candidate=idx[np.argsort(score[idx])[-16:]]
        if len(candidate)>1:
            selected[candidate]=True
    row=dict(K=k,mixture_means=mixture.means_.ravel().tolist(),
        selected=int(selected.sum()),selected_true_known=int((selected&(y<25)).sum()),
        selected_true_unknown=int((selected&(y>=25)).sum()),
        selected_known_fraction=float((y[selected]<25).mean()),
        known_selection_rate=float(selected[y<25].mean()),unknown_selection_rate=float(selected[y>=25].mean()),
        known_rejected_selected=int((selected&(y<25)&(prediction>=25)).sum()),
        eligible_known_rate=float(eligible[y<25].mean()),eligible_unknown_rate=float(eligible[y>=25].mean()),
        relation_quantiles={group:np.quantile(score[mask],[.1,.5,.9]).tolist() for group,mask in [('known',y<25),('unknown',y>=25)]})
    report['arms'][arm]=row
    np.savez_compressed(root/(arm+'-final-relation-snapshot.npz'),truth=y,score=score,prediction=prediction,selected=selected)
    print('RELATION_DIAGNOSTIC_ARM',arm,json.dumps(row),flush=True)
    del net,state
    torch.cuda.empty_cache()
output.write_text(json.dumps(report,indent=2,allow_nan=False))
print('RELATION_DIAGNOSTIC_COMPLETE',flush=True)
