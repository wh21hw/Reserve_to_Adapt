"""One label-blind estimator check on the currently available original57 cache.

No new RTA training, target truth, threshold scan or repeated model forward.
This is not a same-budget counterfactual to the source3/frozenBN20 study.
"""
import json
from pathlib import Path
import shutil
import sys
import numpy as np

sys.path.insert(0,'/content')
from source_precision_capacity import estimate_source_cost_capacity
from prototype_identity_reconciliation import reconcile_known_identities

cache = Path('/content/drive/MyDrive/OSDA/runs/a2w-reference-available57-v1')
manifest = json.loads((cache/'manifest.json').read_text())
if not manifest['complete'] or manifest['target_labels_used'] or (manifest['epoch'],manifest['C'],manifest['K']) != (57,10,2):
    raise ValueError('Require the explicitly declared available57/K2 cache, not claimed final70')
manifest.update(selection='Only currently available last.pt snapshot at epoch57; never oracle best',
    reference_training_budget=70,snapshot_is_training_final=False,
    provenance_note='Archived and mounted logs reach70; currently mounted last.pt metadata is57. Cause not established.')
for file in (cache/'manifest.json',Path('/content/imp-runs/a2w-reference-available57-v1/manifest.json')):
    file.write_text(json.dumps(manifest,indent=2,allow_nan=False))
output = Path('/content/imp-runs/a2w-reference-available57-capacity-v1')
if output.exists():
    raise FileExistsError('Preserve existing inference')
with np.load(cache/'features.npz',allow_pickle=False) as data:
    source,labels,target,logits,paths = (data[key] for key in
        ('source','source_labels','target','target_logits','target_paths'))
result,settings = estimate_source_cost_capacity(source,labels,target,proposal_block_size=64)
occupied = len(np.unique(result['assignments'][result['assignments'] >= 0]))
matched = reconcile_known_identities(result['assignments'],logits[:,:10],10) if occupied >= 10 else None
inferred = result['assignments'] if matched is None else matched['assignments']
model_unknown = logits.argmax(1) >= 10
prototype_unknown = inferred >= 10
prototype_known = (inferred >= 0)&(inferred < 10)
report = dict(task='Office31 A->W',checkpoint=manifest['checkpoint'],epoch=57,head_K=2,
    raw_K=result['K'],matched_K=None if matched is None else matched['K'],
    occupied_clusters=occupied,noise_count=result['noise_count'],counts=result['counts'].tolist(),
    settings=settings,target_labels_used=False,new_training=False,new_forward=False,
    agreement_with_model=dict(rows=len(target),model_unknown_rows=int(model_unknown.sum()),
        prototype_unknown_rows=int(prototype_unknown.sum()),
        prototype_known_but_model_unknown=int((prototype_known&model_unknown).sum()),
        prototype_unknown_but_model_known=int((prototype_unknown&~model_unknown).sum()),
        noise_rows=int((inferred < 0).sum())),
    assignment_basis='raw source-anchor IDs' if matched is None else 'known-identity reconciliation',
    caveat='Estimator sanity on the currently available original epoch57 weights. The last.pt file '
           'does not contain expected epoch70; do not call this final70 or choose an oracle best. '
           'Not a pure K/BN/prefix causal comparison, semantic truth or accuracy reevaluation. No target-fitted parameters.')
if matched is not None:
    report['cluster_to_identity'] = matched['cluster_to_identity']
output.mkdir(parents=True)
arrays = dict(centers=result['centers'],assignments=result['assignments'],target_paths=paths)
if matched is not None:
    arrays['matched_assignments'] = inferred
np.savez_compressed(output/'clusters.npz',**arrays)
(output/'summary.json').write_text(json.dumps(report,indent=2,allow_nan=False))
saved = Path('/content/drive/MyDrive/OSDA/runs/a2w-reference-available57-capacity-v1')
saved.mkdir(exist_ok=False)
for filename in ('clusters.npz','summary.json'):
    shutil.copyfile(output/filename,saved/filename)
print('A2W_REFERENCE_AVAILABLE57_CAPACITY_COMPLETE',json.dumps({key:value for key,value in report.items()
    if key not in ('settings','cluster_to_identity')}),flush=True)
