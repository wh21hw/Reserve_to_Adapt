"""One targeted final-checkpoint inference ablation, no threshold tuning/SGD."""
import json
from pathlib import Path
import sys
import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0,'/content/rta-legacy-l4-bridge-v1')
sys.path.insert(0,'/content')
from networks import ResNetFc, CLS
from alternating_konly import _FrozenImages

torch.set_num_threads(2)
rows = [line.rsplit(None,1) for line in Path('/content/webcam_0-9_20-30_test.txt').read_text().splitlines()]
truth = np.asarray([int(row[1]) for row in rows])
names = [row[0] for row in rows]
loader = DataLoader(_FrozenImages(names,'/content/osda-datasets/'),batch_size=64,
                    shuffle=False,num_workers=4,pin_memory=True)
known_ids, unknown_ids = list(range(10)),list(range(20,31))
if set(truth.tolist()) != set(known_ids+unknown_ids):
    raise ValueError('Unexpected evaluation label split')

def metric(prediction):
    known = np.mean([np.mean(prediction[truth==label]==label) for label in known_ids])*100
    unknown = np.mean([np.mean(prediction[truth==label]>=10) for label in unknown_ids])*100
    return dict(OS_star=float(known),unknown=float(unknown),
                HOS=float(2*known*unknown/(known+unknown)) if known+unknown else 0.)

report = dict(task='Office31 A->W',checkpoints='final epoch10 only',K=7,rows=[],
              rule='unknown sum probability competes with each individual known class; no threshold',
              scope='target labels only used to score fixed preregistered rules; no model update',
              caveat='grouped argmax optimal only for calibrated coarse-label posterior and ordinary accuracy, not guaranteed macro-HOS optimum')
for name,folder in [('self-label',Path('/content/imp-runs/fusion-capacity-diagnostic-v1/K7/rta/a2w_seed1')),
                    ('prototype-teacher',Path('/content/imp-runs/fusion-unknown-teacher-v1/rta/a2w_seed1'))]:
    checkpoint = torch.load(str(folder/'last.pt'),map_location='cpu')
    if checkpoint['epoch'] != 10:
        raise ValueError('Final checkpoint epoch mismatch')
    extractor = ResNetFc(model_name='resnet50',model_path='/content/osda-datasets/resnet50-19c8e357.pth')
    net = torch.nn.Sequential(extractor,CLS(extractor.output_num(),17,bottle_neck_dim=256)).cuda()
    net.load_state_dict(checkpoint['model'])
    net.eval()
    probabilities = []
    with torch.no_grad():
        for images in loader:
            probabilities.append(net(images.cuda())[3].cpu())
    p = torch.cat(probabilities)
    flat = p.argmax(1).numpy()
    coarse = torch.cat([p[:,:10],p[:,10:].sum(1,keepdim=True)],1).argmax(1).numpy()
    switched = (flat<10)&(coarse>=10)
    report['rows'].append(dict(method=name,original=metric(flat),marginal=metric(coarse),
        switched_known_to_unknown=int(switched.sum()),
        switched_true_known=int((switched&(truth<10)).sum()),
        switched_true_unknown=int((switched&(truth>=10)).sum()),
        training_final=checkpoint['metrics'],evaluation_samples=len(truth)))
    del net,checkpoint
    torch.cuda.empty_cache()
destination = Path('/content/imp-runs/fusion-unknown-teacher-v1/marginal-decode.json')
destination.write_text(json.dumps(report,indent=2,allow_nan=False))
print(json.dumps(report,indent=2),flush=True)
