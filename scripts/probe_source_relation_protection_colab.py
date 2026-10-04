"""Fixed source99 protection probe against cached target selections, no SGD."""
import json
from pathlib import Path
import sys
import numpy as np
import torch
from torch.utils.data import DataLoader
sys.path.insert(0,'/content/rta-legacy-l4-bridge-v1')
sys.path.insert(0,'/content')
from networks import ResNetFc,CLS
from alternating_konly import _FrozenImages

root=Path('/content/imp-runs/officehome-capacity-10e-v1')
output=root/'source99-protection-probe.json'
if output.exists():
    raise FileExistsError('Preserve previous probe')
torch.set_num_threads(2)
rows=[x.rsplit(None,1) for x in Path('/content/osda-officehome-pr2rw-v1/product_0-24_train_all.txt').read_text().splitlines() if x.strip()]
labels=np.array([int(x[1]) for x in rows])
loader=DataLoader(_FrozenImages([x[0] for x in rows],'/content/osda-officehome-pr2rw-v1'),batch_size=64,shuffle=False,num_workers=4)
report=dict(rule='Keep existing unknown candidates only when relation score > source99 threshold',
    quantile=.99,target_used_for_calibration=False,training=False,
    caveat='In-sample source calibration, final center-crop snapshot; no held-out guarantee or historical gate reconstruction',arms={})
for arm,k in [('fixed4',4),('estimated',1)]:
    state=torch.load(str(root/arm/'officehome-pr2rw_seed1/last.pt'),map_location='cpu')
    extractor=ResNetFc(model_path='/content/osda-datasets/resnet50-19c8e357.pth')
    net=torch.nn.Sequential(extractor,CLS(extractor.output_num(),25+k,bottle_neck_dim=256)).cuda().eval()
    net.load_state_dict(state['model'])
    probs=[]
    with torch.no_grad():
        for images in loader:
            probs.append(net(images.cuda())[2][:,:25].softmax(1).cpu())
    p=torch.cat(probs).double()
    bank=torch.stack([p[labels==c].mean(0) for c in range(25)])
    score=torch.nn.functional.kl_div(p.clamp_min(1e-12).log(),bank[p.argmax(1)],reduction='none').sum(1).numpy()
    threshold=max(float(np.quantile(score,.99)),1e-8)
    np.savez_compressed(root/(arm+'-source-relation-calibration.npz'),scores=score,threshold=threshold)
    cached=np.load(root/(arm+'-final-relation-snapshot.npz'))
    original=cached['selected']; kept=original&(cached['score']>threshold)
    # Target labels are accessed only after the rule and threshold are fixed.
    truth=cached['truth']; removed=original&~kept
    report['arms'][arm]=dict(K=k,source_threshold=threshold,source_samples=len(labels),
        original_candidates=int(original.sum()),kept=int(kept.sum()),
        removed_known=int((removed&(truth<25)).sum()),removed_unknown=int((removed&(truth>=25)).sum()),
        kept_known=int((kept&(truth<25)).sum()),kept_unknown=int((kept&(truth>=25)).sum()))
    print('SOURCE_PROTECTION_PROBE',arm,json.dumps(report['arms'][arm]),flush=True)
    del net,state
    torch.cuda.empty_cache()
output.write_text(json.dumps(report,indent=2,allow_nan=False))
print('SOURCE_PROTECTION_PROBE_COMPLETE',flush=True)
