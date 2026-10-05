"""Post-fit source proxy composition; cached last arm, no refit or target labels."""
import json
from pathlib import Path
import numpy as np
from scipy.optimize import linear_sum_assignment

root = Path('/content/imp-runs/source-frozenbn-block0to4-v1')
output = root/'shared-shift-proxy-composition-v1.json'
payload = root/'shared-shift-translated-mixed-cache-v1.npz'
if output.exists() or payload.exists():
    raise FileExistsError('Preserve diagnostic/cache')
expected = json.loads((root/'shared-shift-probe-v1.json').read_text())['arms']['translated/mixed/shared_shift']
if r['K'] != 3 or r['centers'].shape != (23,256) or len(test) != 231 or len(hidden) != 51:
    raise ValueError('Cached last arm differs from declared experiment')
assignments = r['assignments']
if len(assignments) != 282 or not np.isclose((assignments[:231] >= 20).mean(),expected['known_false_candidate_rate']):
    raise ValueError('Cached assignments differ from reported last arm')
raw_labels = np.concatenate([labels[test]+5, y[hidden]])
if not np.isclose((assignments[231:] >= 20).mean(),expected['hidden_candidate_recall']):
    raise ValueError('Cached hidden recall differs')
np.savez_compressed(payload,assignments=assignments,raw_labels=raw_labels,
    centers=r['centers'],shared_shift=np.asarray(r['shared_shift']))
table = np.array([[(raw_labels[assignments == slot] == cls).sum() for cls in range(5)] for slot in range(20,23)])
matched_rows, matched_cols = linear_sum_assignment(-table)
matches = int(table[matched_rows, matched_cols].sum())
report = dict(source_only=True, real_target_used=False, arm='translated/mixed/shared_shift',
    fitting_unchanged=True, K=3, hidden_semantic_classes=5,
    clusters=[dict(slot=slot,support=int((assignments == slot).sum()),
        known_members=int(((assignments[:231] == slot)).sum()),
        hidden_counts=table[slot-20].tolist()) for slot in range(20,23)],
    hidden_per_class=[dict(raw_class=cls,total=int((raw_labels[231:] == cls).sum()),
        candidate_count=int(((raw_labels[231:] == cls)&(assignments[231:] >= 20)).sum())) for cls in range(5)],
    one_to_one_matching=[dict(slot=20+int(row),raw_class=int(col),count=int(table[row,col])) for row,col in zip(matched_rows,matched_cols)],
    one_to_one_correct=matches, hidden_rows=51,
    one_to_one_accuracy_over_all_hidden=matches/51.,
    caveat='Post-fit source proxy labels only. Candidate membership recall does not imply recovering five semantic classes. Matching is diagnosis, never passed to training.')
output.write_text(json.dumps(report,indent=2,allow_nan=False))
print('SHARED_SHIFT_COMPOSITION',json.dumps(report),flush=True)
