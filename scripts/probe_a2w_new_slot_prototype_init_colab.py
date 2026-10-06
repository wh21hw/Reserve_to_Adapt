"""One in-memory head prototype counterfactual; no forward, training or truth."""
import argparse
import json
from pathlib import Path
import shutil
import numpy as np
import torch

torch.set_num_threads(2)
parser = argparse.ArgumentParser()
parser.add_argument('--head-state',required=True,help='Small saved final70 fc/BN subset, not a new model')
args = parser.parse_args()
run = Path('/content/drive/MyDrive/OSDA/runs/a2w-capacity-refresh-k2-70e-v1/refresh/office31-a2w_seed3')
cache = Path('/content/drive/MyDrive/OSDA/runs/a2w-capacity-refresh-k2-70e-final-v1')
output = Path('/content/imp-runs/a2w-new-slot-init-proxy-v1')
durable = Path('/content/drive/MyDrive/OSDA/runs/a2w-new-slot-init-proxy-v1')
if output.exists() or durable.exists():
    raise FileExistsError('Preserve existing single-proxy result')
manifest = json.loads((cache/'manifest.json').read_text())
if (not manifest['complete'] or manifest['target_labels_used'] or
        (manifest['epoch'],manifest['C'],manifest['K']) != (70,10,5)):
    raise ValueError('Require declared final70 K5 cache')
with np.load(cache/'features.npz',allow_pickle=False) as data:
    features = {split:torch.from_numpy(data[split].copy()) for split in ('source','target')}
    cached_logits = {split:torch.from_numpy(data[split+'_logits'].copy()) for split in features}
    paths = data['target_paths'].copy()
with np.load(run/'capacity-after-010/snapshot.npz',allow_pickle=False) as data:
    if not np.array_equal(paths,data['target_paths']):
        raise ValueError('Candidate membership sample order differs')
    assignments = torch.from_numpy(data['matched_assignments'].copy())
checkpoint = torch.load(args.head_state,map_location='cpu')
state = checkpoint['model']
if checkpoint['epoch'] != 70 or state['1.fc.weight'].shape != (15,256):
    raise ValueError('Actual checkpoint must be final70 C10+K5')
weight = state['1.fc.weight'].clone()
bn = '1.main.1.0.'
head_features,errors = {},{}
with torch.no_grad():
    for split,feature in features.items():
        value = (feature-state[bn+'running_mean'])/torch.sqrt(state[bn+'running_var']+1e-5)
        value = value*state[bn+'weight']+state[bn+'bias']
        head_features[split] = torch.where(value>=0,value,.2*value)
        reconstructed = head_features[split]@weight.T
        errors[split] = float((reconstructed-cached_logits[split]).abs().max())
        if not torch.allclose(reconstructed,cached_logits[split],atol=1e-4,rtol=1e-5):
            raise RuntimeError('Head-space reconstruction mismatch; do not test wrong-space initializer')
    changed_weight = weight.clone()
    norm = weight[:12].norm(dim=1).mean()
    supports = []
    for row in (12,13,14):
        members = head_features['target'][assignments==row]
        if not len(members):
            raise RuntimeError('New candidate has no members')
        center = members.mean(0)
        if not torch.isfinite(center).all() or center.norm()<=1e-8:
            raise RuntimeError('Undefined prototype direction')
        changed_weight[row] = center/center.norm()*norm
        supports.append(dict(row=row,members=len(members)))
    if not torch.equal(changed_weight[:12],weight[:12]):
        raise RuntimeError('Known or retained unknown rows changed')
    statistics = {}
    for split,feature in head_features.items():
        cases = {}
        for name,w in (('original',weight),('prototype',changed_weight)):
            logits = feature@w.T
            prediction = logits.argmax(1)
            unknown_only = logits[:,10:].argmax(1)+10
            cases[name] = dict(rows=len(feature),
                full_head_counts=torch.bincount(prediction,minlength=15).tolist(),
                unknown_only_counts=torch.bincount(unknown_only-10,minlength=5).tolist(),
                unknown_predictions=int((prediction>=10).sum()),
                new_row_predictions=int((prediction>=12).sum()),
                new_rows_with_full_winners=int(sum(bool((prediction==row).any()) for row in (12,13,14))))
        original = (feature@weight.T).argmax(1)
        changed = (feature@changed_weight.T).argmax(1)
        statistics[split] = dict(cases=cases,
            known_to_unknown=int(((original<10)&(changed>=10)).sum()),
            unknown_to_known=int(((original>=10)&(changed<10)).sum()),
            any_prediction_changes=int((original!=changed).sum()))
source = statistics['source']['cases']
increase = 100*(source['prototype']['unknown_predictions']-source['original']['unknown_predictions'])/source['original']['rows']
metric = statistics['target']['cases']['prototype']['new_rows_with_full_winners']
report = dict(epoch=70,C=10,K=5,assignment_epoch=10,representation_epoch=70,
    stale_membership=True,prototype_space='Actual eval BN->LeakyReLU fc input',
    norm_rule='Mean norm of unchanged known10+retained unknown2 rows',norm=float(norm),
    supports=supports,reconstruction_max_abs_error=errors,statistics=statistics,
    metric_new_rows_used=metric,source_rejection_increase_pp=increase,
    source_guard_at_most_1pp=increase<=1.,candidate_has_functional_activation=metric>=1,
    target_labels_used=False,new_training=False,new_image_forward=False,K_refitted=False,
    checkpoint_modified=False,caveat='Frozen final-state initialization stress test, '
    'not epoch10 training, target accuracy or complete OSDA improvement. Candidate '
    'memberships from epoch10 are deliberately held fixed and may be stale.')
output.mkdir(parents=True)
(output/'summary.json').write_text(json.dumps(report,indent=2,allow_nan=False))
durable.mkdir(parents=True)
shutil.copyfile(output/'summary.json',durable/'summary.json')
print('PROTOTYPE_INITIALIZATION_PROXY_COMPLETE',json.dumps(report),flush=True)
