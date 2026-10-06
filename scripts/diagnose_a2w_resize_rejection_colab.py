"""Reconstruct immediate pruning effect from saved pre-resize logits only.

No image forward, checkpoint evaluation, target truth, training or new K fit.
With known rows unchanged and unknown rows only pruned/permuted, unknown max
cannot increase. This says nothing about the subsequent SGD-trained model.
"""
import json
from pathlib import Path
import shutil
import numpy as np

root = Path('/content/imp-runs/a2w-capacity-refresh-20e-v1')
folder = root/'refresh/office31-a2w_seed3/capacity-after-010'
report = json.loads((folder/'estimate.json').read_text())
head = report['head']
if not report['applied'] or report['target_labels_used']:
    raise ValueError('Require completed label-blind capacity update')
destination = root/'resize-rejection-diagnostic.json'
if destination.exists():
    raise FileExistsError('Keep existing diagnostic')
result = dict(target_labels_used=False,new_training=False,new_forward=False,
    checkpoint_reevaluated=False,old_K=head['old_K'],new_K=head['new_K'],
    interpretation='Immediate pruning only; not final OSDA accuracy, probability-aggregation prediction, '
                   'or proof of post-update training outcome')
if head['new_K'] > head['old_K']:
    result.update(available=False,reason='Growth has random new weights; saved old logits insufficient')
else:
    with np.load(folder/'snapshot.npz',allow_pickle=False) as data:
        logits = data['target_logits']
    C = report['settings']['C']
    if logits.shape[1] != C+head['old_K']:
        raise ValueError('Cache not from the pre-update classifier')
    if head['changed']:
        mapping = sorted(head['row_mapping'])
        if [new for new,old in mapping] != list(range(C,C+head['new_K'])):
            raise ValueError('Cannot reconstruct all new unknown rows')
        if not head['known_rows_preserved'] or head['new_random_rows'] != 0:
            raise ValueError('This is not a pure prune/permutation')
        columns = list(range(C))+[old for new,old in mapping]
    else:
        columns = list(range(logits.shape[1]))
    before = logits.argmax(1) >= C
    after = logits[:,columns].argmax(1) >= C
    if ((~before)&after).any():
        raise RuntimeError('Contradiction to pure head-pruning reconstruction')
    result.update(available=True,rows=len(logits),before_unknown=int(before.sum()),
        immediately_after_unknown=int(after.sum()),unknown_to_known=int((before&~after).sum()),
        known_to_unknown=int((~before&after).sum()),unchanged=int((before==after).sum()))
destination.write_text(json.dumps(result,indent=2,allow_nan=False))
saved = Path('/content/drive/MyDrive/OSDA/runs/a2w-capacity-refresh-20e-v1')
if not (saved/'refresh/office31-a2w_seed3/last.pt').is_file():
    raise RuntimeError('Preserve completed training before this post-run report')
shutil.copyfile(destination,saved/destination.name)
print('A2W_RESIZE_REJECTION_DIAGNOSTIC',json.dumps(result),flush=True)
