"""Post-fit label composition diagnostic; never feeds labels back into training."""
import json
from pathlib import Path
import numpy as np

root = Path('/content/imp-runs/visda-frozenbn-capacity-10e-v1')
data = Path('/content/osda-visda-syn2real-v1')
output = root/'capacity-composition-diagnostic.json'
if output.exists():
    raise FileExistsError('Preserve existing diagnostic')
capacity = json.loads((root/'capacity.json').read_text())
config = json.loads((root/'source/config.json').read_text())
rows = [line.rsplit(None, 1) for line in (data/'target-real-12.txt').read_text().splitlines()]
paths = (data/'target-unlabeled-paths.txt').read_text().splitlines()
if [row[0] for row in rows] != paths:
    raise ValueError('Evaluation and frozen-feature path order differ')
labels = np.array([int(row[1]) for row in rows])
with np.load(root/'capacity.npz') as f:
    assignment = f['assignments']
if assignment.shape != labels.shape or not capacity['converged']:
    raise ValueError('Expected completed assignments matching evaluation rows')
c = capacity['C']
known_ids = config['original_known_ids']
novel = assignment >= c
noise = assignment < 0
known = np.isin(labels, known_ids)
def summary(mask):
    return dict(samples=int(mask.sum()), assigned_new=int((mask & novel).sum()),
                assigned_known=int((mask & (assignment >= 0) & ~novel).sum()),
                noise=int((mask & noise).sum()), new_fraction=float(novel[mask].mean()))
clusters = []
for cluster in range(c+capacity['K']):
    mask = assignment == cluster
    composition = {str(raw): int((mask & (labels == raw)).sum()) for raw in np.unique(labels)}
    clusters.append(dict(cluster=cluster, role='source-known' if cluster < c else 'candidate-new',
                         samples=int(mask.sum()), known_samples=int((mask & known).sum()),
                         unknown_samples=int((mask & ~known).sum()), raw_class_counts=composition))
report = dict(task=capacity['task'], C=c, K=capacity['K'], known_raw_ids=known_ids,
              known=summary(known), unknown=summary(~known),
              by_raw_class={str(raw): summary(labels == raw) for raw in np.unique(labels)},
              clusters=clusters, target_labels_used_for_diagnostic=True,
              target_labels_used_for_capacity_or_training=False,
              caveat='Post-fit diagnosis only; no threshold selection, cluster removal, or training changes')
output.write_text(json.dumps(report, indent=2, allow_nan=False))
print('VISDA_CAPACITY_DIAGNOSTIC', json.dumps(dict(known=report['known'], unknown=report['unknown'],
    by_raw_class=report['by_raw_class'], output=str(output))), flush=True)
