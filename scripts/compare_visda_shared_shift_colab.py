"""Post-fit composition comparison; labels never enter estimators or RTA."""
import json
from pathlib import Path
import numpy as np
from scipy.optimize import linear_sum_assignment

root = Path('/content/imp-runs/visda-frozenbn-capacity-10e-v1')
out = root/'shared-domain-shift-comparison.json'
if out.exists():
    raise FileExistsError('Preserve completed comparison')
data = Path('/content/osda-visda-syn2real-v1')
rows = [line.rsplit(None,1) for line in (data/'target-real-12.txt').read_text().splitlines()]
if [row[0] for row in rows] != (data/'target-unlabeled-paths.txt').read_text().splitlines():
    raise ValueError('Target feature order differs')
raw = np.array([int(row[1]) for row in rows])
config = json.loads((root/'source/config.json').read_text())
known_ids = config['original_known_ids']
known = np.isin(raw,known_ids)
unknown_ids = sorted(set(raw.tolist())-set(known_ids))
mapped = np.array([known_ids.index(int(v)) if v in known_ids else -1 for v in raw])
report = dict(task='VisDA Synthetic->Real', target_labels_used_for_diagnostic=True,
    target_labels_used_for_inference_or_training=False, known_rows=int(known.sum()),
    unknown_rows=int((~known).sum()), arms={},
    caveat='Same frozen arrays, post-fit diagnostics only. Unknown matching is oracle assessment, not training labels or evidence of classification gains. No fitting, target threshold selection or RTA reevaluation.')
for name,directory in [('original',root),('shared_shift',root/'shared-domain-shift-v1')]:
    capacity = json.loads((directory/'capacity.json').read_text())
    if not capacity['converged'] or capacity['C'] != 6 or capacity['settings']['target_labels_used']:
        raise ValueError('Expected completed unlabeled C6 capacity')
    with np.load(directory/'capacity.npz') as f:
        ids, centers = f['assignments'], f['centers']
    k = capacity['K']
    if len(ids) != len(raw) or len(centers) != 6+k:
        raise ValueError('Shape/count differs')
    table = np.array([[(raw[ids == slot] == cls).sum() for cls in unknown_ids] for slot in range(6,6+k)],dtype=np.int64).reshape(k,len(unknown_ids))
    matched = 0
    if k:
        rr,cc = linear_sum_assignment(-table)
        matched = int(table[rr,cc].sum())
    row = dict(K=k, noise_count=int((ids < 0).sum()),
        known_false_candidate_rate=float((ids[known] >= 6).mean()),
        known_preserved_identity_rate=float((ids[known] == mapped[known]).mean()),
        unknown_candidate_recall=float((ids[~known] >= 6).mean()),
        unknown_absorbed_by_known_rate=float(((ids[~known] >= 0)&(ids[~known] < 6)).mean()),
        unknown_one_to_one_correct=matched,
        unknown_one_to_one_accuracy_over_all_unknown=matched/int((~known).sum()),
        candidates=[dict(slot=slot,support=int((ids == slot).sum()),
            known_members=int(((ids == slot)&known).sum()),
            unknown_semantic_counts=table[slot-6].tolist()) for slot in range(6,6+k)],
        unknown_semantic_ids=unknown_ids, settings=capacity['settings'])
    report['arms'][name] = row
metrics=('known_false_candidate_rate','known_preserved_identity_rate','unknown_candidate_recall',
    'unknown_absorbed_by_known_rate','unknown_one_to_one_accuracy_over_all_unknown')
report['shared_minus_original_pp'] = {m:100*(report['arms']['shared_shift'][m]-report['arms']['original'][m]) for m in metrics}
out.write_text(json.dumps(report,indent=2,allow_nan=False))
print('VISDA_SHARED_SHIFT_COMPARISON',json.dumps(report),flush=True)
