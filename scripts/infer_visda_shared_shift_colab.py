"""Independent CPU full-data inference; never changes the running RTA pilot."""
import json
from pathlib import Path
import sys
import time
import numpy as np

sys.path.insert(0, '/content/visda-shared-shift-code-v1')
from source_precision_capacity import estimate_source_cost_capacity

root = Path('/content/imp-runs/visda-frozenbn-capacity-10e-v1')
# Uses frozen source-stage arrays, independent of current RTA epoch or scores.
summary = json.loads((root/'source/summary.json').read_text())
config = json.loads((root/'source/config.json').read_text())
if not summary['complete'] or summary['epochs'] != 3 or not config['freeze_backbone_bn']:
    raise ValueError('Expected shared frozen-BN source3')
out = root/'shared-domain-shift-v1'
if out.exists():
    raise FileExistsError('Preserve previous inference, including failures')
with np.load(root/'source/features.npz') as f:
    source, labels, target = f['source'], f['source_labels'], f['target']
if source.shape != (79765,256) or target.shape != (55388,256) or set(np.unique(labels)) != set(range(6)):
    raise ValueError('Expected full VisDA arrays and six known classes')
out.mkdir()
protocol = dict(task='VisDA Synthetic->Real', seed=1, C=6, version='shared-domain-shift-v1',
    source_prior='Same frozen-BN ResNet50 C6 source CE3 as paired K-only pilot',
    changed_factor='One shared latent translation in source-anchor prior',
    shift_precision_rule='tau=source reference_samples R; frozen by source-only proxy',
    scheduling='Independent CPU inference with two BLAS threads; paired GPU RTA unchanged',
    source_calibration_unchanged=True, proposal_block_size=256,
    target_labels_used=False, RTA_started=False, semantic_unknown_count=None)
(out/'protocol.json').write_text(json.dumps(protocol,indent=2))
start = time.time()
print('VISDA_SHARED_SHIFT_START', json.dumps(protocol), flush=True)
try:
    result, settings = estimate_source_cost_capacity(source, labels, target,
        proposal_block_size=256, shared_domain_shift=True)
except Exception as exc:
    (out/'failure.json').write_text(json.dumps(dict(error_type=type(exc).__name__,error=str(exc),
        seconds=time.time()-start, protocol=protocol),indent=2))
    raise
report = dict(protocol=protocol, C=6, K=result['K'], converged=result['converged'],
    settings=settings, counts=result['counts'].tolist(), noise_count=result['noise_count'],
    history=result['history'], seconds=time.time()-start,
    shared_shift_norm=float(np.linalg.norm(result['shared_shift'])),
    caveat='Capacity structure, not proved semantic count; one common translation may miss real domain changes. No automatic RTA run, K flooring, or target-label tuning.')
np.savez_compressed(out/'capacity.npz',centers=result['centers'],assignments=result['assignments'],
    shared_shift=np.asarray(result['shared_shift']))
(out/'capacity.json').write_text(json.dumps(report,indent=2,allow_nan=False))
print('VISDA_SHARED_SHIFT_COMPLETE',json.dumps(report),flush=True)
