"""Assess one existing frozen IMP structure before proposing slot supervision.

No training, no parameter search. Target truth is read only after assignments.
"""
import json
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from alternating_konly import estimate_current_K

torch.set_num_threads(2)
prior = Path('/content/imp-runs/fusion-imp-rta-v1/seed1/source')
destination = Path('/content/initial-imp-binding-a2w-v1.json')
if destination.exists():
    raise FileExistsError('Preserve previous diagnostic')
features = np.load(prior / 'features.npz')
source = torch.from_numpy(features['source']).float()
target = torch.from_numpy(features['target']).float()
source_labels = torch.from_numpy(features['source_labels']).long()
result, settings = estimate_current_K(source, source_labels, target, 10)
assignment = result['responsibilities'].argmax(1).cpu().numpy()
k = int(result['candidate_count'])
if k != 7:
    raise ValueError('Existing initial K7 not restored; do not alter settings')
# Ordered filenames are the same list used by the frozen source extraction.
rows = [r.rsplit(None, 1) for r in Path('/content/webcam_0-9_20-30_test.txt').read_text().splitlines() if r.strip()]
truth = np.array([int(r[1]) for r in rows])
if len(truth) != len(assignment) or set(np.unique(truth)) != set(range(10)) | set(range(20, 31)):
    raise ValueError('Unexpected target list or label protocol')
unknown = truth >= 20
candidate = assignment >= 10
components = []
contingency = np.zeros((k, 11), dtype=int)
for slot in range(k):
    mask = assignment == 10 + slot
    values, counts = np.unique(truth[mask], return_counts=True)
    for c in range(20, 31):
        contingency[slot, c - 20] = int((mask & (truth == c)).sum())
    components.append(dict(slot=slot, support=int(mask.sum()), known=int((mask & ~unknown).sum()),
                           unknown=int((mask & unknown).sum()),
                           composition={str(int(v)): int(n) for v, n in zip(values, counts)}))
r, c = linear_sum_assignment(-contingency)
matched = int(contingency[r, c].sum())
report = dict(task='Office31 A->W', seed=1, source_epochs=3, K=k, settings=settings,
              scope='Frozen initial geometry attribution only; not training performance or tuning',
              target_labels_used_for_fit=False, total=len(truth), known_total=int((~unknown).sum()),
              unknown_total=int(unknown.sum()), candidate_total=int(candidate.sum()),
              candidate_unknown_recall=float(candidate[unknown].mean()),
              candidate_precision=float(unknown[candidate].mean()) if candidate.any() else None,
              known_false_candidate_rate=float(candidate[~unknown].mean()),
              unknown_absorbed_by_known=int((unknown & ~candidate).sum()),
              known_identity_accuracy=float((assignment[~unknown] == truth[~unknown]).mean()),
              candidate_semantic_purity=float(contingency.max(1).sum()/contingency.sum()) if contingency.sum() else None,
              one_to_one_unknown_coverage=matched/int(unknown.sum()),
              components=components,
              per_unknown_class_coverage={str(cls):float(candidate[truth == cls].mean()) for cls in range(20,31)},
              caveat='Purity can be inflated by tiny clusters; label diagnostics never define K, thresholds or mapping for training')
destination.write_text(json.dumps(report, indent=2, allow_nan=False))
print('INITIAL_IMP_BINDING', json.dumps(report), flush=True)
