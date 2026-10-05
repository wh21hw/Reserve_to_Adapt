"""One cached mechanism probe, not new training or target-based tuning."""
import json
from pathlib import Path
import sys
import numpy as np

sys.path.insert(0, '/content/known-guard-v1')
from reliable_known_guard import infer_protected_known

root = Path('/content/imp-runs/visda-frozenbn-capacity-10e-v1/fixed2-final10-geometry-v1')
output = root / 'source-calibrated-guard-probe.json'
if output.exists():
    raise FileExistsError('No repeat/threshold sweep')
known = [1, 2, 3, 6, 10, 11]
data = Path('/content/osda-visda-syn2real-v1')
source_raw = np.array([int(r.rsplit(None, 1)[1]) for r in
                      (data/'source-known-6.txt').read_text().splitlines() if r.strip()])
labels = np.array([known.index(int(v)) for v in source_raw])
with np.load(str(root/'features.npz'), allow_pickle=False) as cache:
    guard = infer_protected_known(cache['source'], labels, cache['source_logits'],
                                 cache['target'], cache['target_logits'])
# Freeze inferred mask before accessing truth for post-hoc evaluation.
mask, identity = guard['protected_known'], guard['known_identity']
target_raw = np.array([int(r.rsplit(None, 1)[1]) for r in
                      (data/'target-real-12.txt').read_text().splitlines() if r.strip()])
known_mask = np.isin(target_raw, known)
if len(target_raw) != len(mask):
    raise ValueError('Diagnostic label rows mismatch')
correct = np.zeros(len(mask), dtype=bool)
for c, raw in enumerate(known):
    correct |= (target_raw == raw) & (identity == c)
report = dict(stage='Fixed final10 failure-cache probe, not a deployable warm-stage result',
    target_labels_used_for_guard=False, target_labels_used_for_posthoc_diagnostics=True,
    quantiles=dict(radius=.99, confidence=.01),
    radii=guard['radii'].tolist(), confidence=guard['confidence'].tolist(),
    protected_count=int(mask.sum()), protected_known_count=int((mask & known_mask).sum()),
    protected_unknown_count=int((mask & ~known_mask).sum()),
    known_coverage=float(mask[known_mask].mean()),
    unknown_false_protection=float(mask[~known_mask].mean()),
    known_identity_precision=float(correct[mask & known_mask].mean()) if (mask & known_mask).any() else None,
    caveat='Predeclared source quantiles, no target tuning. Final collapsed features may not support sufficient known coverage. No RTA benefit established.')
output.write_text(json.dumps(report, indent=2, allow_nan=False))
print('KNOWN_GUARD_CACHE_PROBE', json.dumps(report), flush=True)
