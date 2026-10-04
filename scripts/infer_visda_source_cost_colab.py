"""Full-data frozen-rule capacity inference; no target labels or model evaluation."""
import json
from pathlib import Path
import sys
import time
import numpy as np

# Isolated, previously checked blocked implementation; do not accidentally use
# an older /content module from the completed OfficeHome experiments.
sys.path.insert(0,'/content/visda-capacity-code-v1')
from source_precision_capacity import estimate_source_cost_capacity

root=Path('/content/imp-runs/visda-frozenbn-capacity-10e-v1')
output=root/'capacity.json';details=root/'capacity.npz'
if output.exists() or details.exists():raise FileExistsError('Preserve existing capacity result')
summary=json.loads((root/'source/summary.json').read_text())
config=json.loads((root/'source/config.json').read_text())
if not summary['complete'] or summary['epochs']!=3 or not config['freeze_backbone_bn']:
    raise ValueError('Expected completed frozen-BN source3 prior')
with np.load(root/'source/features.npz') as f:
    source=f['source'];labels=f['source_labels'];target=f['target']
if source.shape!=(79765,256) or target.shape!=(55388,256) or labels.shape!=(79765,):
    raise ValueError('Expected full VisDA source/target features; no sampling')
if set(np.unique(labels).tolist())!=set(range(6)):
    raise ValueError('Expected six remapped known source classes')
start=time.time()
print('VISDA_CAPACITY_START',json.dumps(dict(source_shape=list(source.shape),target_shape=list(target.shape),
    block_size=256,target_labels_used=False,implementation='7ece9d34 blocked full proposals',
    caveat='Memory bounded; proposal time remains quadratic')),flush=True)
result,settings=estimate_source_cost_capacity(source,labels,target,proposal_block_size=256)
report=dict(task='VisDA Synthetic->Real',seed=1,C=6,K=result['K'],converged=result['converged'],
    settings=settings,counts=result['counts'].tolist(),noise_count=result['noise_count'],
    history=result['history'],seconds=time.time()-start,target_labels_used=False,
    semantic_unknown_count=None,source_prior='ResNet50 C6 source CE3, frozen encoder BN',
    implementation='Blocked exact candidates, no subsampling')
np.savez_compressed(details,centers=result['centers'],assignments=result['assignments'])
output.write_text(json.dumps(report,indent=2,allow_nan=False))
print('VISDA_CAPACITY_COMPLETE',json.dumps(report),flush=True)
