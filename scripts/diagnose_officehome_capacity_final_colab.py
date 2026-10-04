"""One final inference per arm: error decomposition, no training or tuning."""
import json
from pathlib import Path
import sys
import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, '/content/rta-legacy-l4-bridge-v1')
sys.path.insert(0, '/content')
from networks import ResNetFc, CLS
from alternating_konly import _FrozenImages

torch.set_num_threads(2)
root = Path('/content/imp-runs/officehome-capacity-10e-v1')
destination = root/'final-error-diagnostic.json'
if destination.exists():
    raise FileExistsError('Preserve existing diagnostic')
rows = [x.rsplit(None, 1) for x in Path('/content/osda-officehome-pr2rw-v1/real_world_0-64_test.txt').read_text().splitlines() if x.strip()]
truth = np.array([int(x[1]) for x in rows])
loader = DataLoader(_FrozenImages([x[0] for x in rows], '/content/osda-officehome-pr2rw-v1'),
                    batch_size=64, shuffle=False, num_workers=4, pin_memory=True)
if set(truth.tolist()) != set(range(65)):
    raise ValueError('Unexpected evaluation split')

def macro(mask, ids):
    return float(np.mean([mask[truth == c].mean() for c in ids])*100)

report = dict(task='OfficeHome Pr->Rw',epoch=10,seed=1,
    scope='Final inference only; labels for error explanation, never estimation or tuning',arms={})
for arm, k in [('fixed4',4),('estimated',1)]:
    state = torch.load(str(root/arm/'officehome-pr2rw_seed1/last.pt'),map_location='cpu')
    if state['epoch'] != 10:
        raise ValueError('Expected final epoch10')
    extractor = ResNetFc(model_name='resnet50',model_path='/content/osda-datasets/resnet50-19c8e357.pth')
    net = torch.nn.Sequential(extractor, CLS(extractor.output_num(),25+k,bottle_neck_dim=256)).cuda().eval()
    net.load_state_dict(state['model'])
    probabilities=[]
    with torch.no_grad():
        for images in loader:
            probabilities.append(net(images.cuda())[3].cpu().numpy())
    p=np.concatenate(probabilities)
    if p.shape != (len(truth),25+k) or not np.isfinite(p).all():
        raise ValueError('Invalid probabilities')
    prediction=p.argmax(1)
    known_correct=macro(prediction==truth,range(25))
    known_rejected=macro(prediction>=25,range(25))
    known_confused=macro((prediction<25)&(prediction!=truth),range(25))
    unk=macro(prediction>=25,range(25,65))
    maximum_known=p[:,:25].max(1)
    maximum_unknown=p[:,25:].max(1)
    total_unknown=p[:,25:].sum(1)
    split=(total_unknown>maximum_known)&(maximum_unknown<=maximum_known)
    report['arms'][arm]=dict(K=k,samples=len(truth),OS_star=known_correct,UNK=unk,
        HOS=2*known_correct*unk/(known_correct+unk),
        known_rejected_macro=known_rejected,known_confused_macro=known_confused,
        known_only_argmax_accuracy=macro(p[:,:25].argmax(1)==truth,range(25)),
        unknown_slot_winners=[int((prediction==25+j).sum()) for j in range(k)],
        split_probability_true_known=int((split&(truth<25)).sum()),
        split_probability_true_unknown=int((split&(truth>=25)).sum()),
        unknown_total_mean=dict(known=float(total_unknown[truth<25].mean()),unknown=float(total_unknown[truth>=25].mean())),
        training_final=state['metrics'])
    print('FINAL_DIAGNOSTIC_ARM',arm,json.dumps(report['arms'][arm]),flush=True)
    del net,state
    torch.cuda.empty_cache()
destination.write_text(json.dumps(report,indent=2,allow_nan=False))
print('FINAL_DIAGNOSTIC_COMPLETE',str(destination),flush=True)
