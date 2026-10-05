"""Current source3 A2W structure probe: cached CPU only, no rule search.

Fit the already declared source-calibrated birth-cost rule WITHOUT target truth.
Persist the label-free artifact before a separate posthoc composition report.
"""
import json
from pathlib import Path
import shutil
import sys
import time
import numpy as np

sys.path.insert(0, '/content/visda-capacity-code-v1')
from source_precision_capacity import estimate_source_cost_capacity

root = Path('/content/imp-runs/a2w-unknown-ce-10e-v1')
output = root/'imp-structure-probe-v1'
if output.exists():
    raise FileExistsError('Preserve previous inference; no automatic retry')
summary = json.loads((root/'source/summary.json').read_text())
config = json.loads((root/'source/config.json').read_text())
if (not summary['complete'] or config['seed'] != 3 or config['epochs'] != 3
        or config['known_classes'] != 10 or not config['freeze_backbone_bn']):
    raise ValueError('Require the shared prior from the completed A2W pair')
with np.load(root/'source/features.npz', allow_pickle=False) as features:
    source, labels, target = features['source'], features['source_labels'], features['target']
if source.shape != (958,256) or target.shape != (564,256):
    raise ValueError('Unexpected full cached feature shapes')
paths = [line.rsplit(None,1)[0] for line in
         Path('/content/osda-office31-a2w-v1/webcam_0-9_20-30_test.txt').read_text().splitlines() if line.strip()]
start = time.time()
result, settings = estimate_source_cost_capacity(source, labels, target, proposal_block_size=64)
output.mkdir()
artifact = dict(task='Office31 A->W', seed=3, source_epochs=3,
    K=result['K'], counts=result['counts'].tolist(), noise_count=result['noise_count'],
    converged=result['converged'], settings=settings, seconds=time.time()-start,
    target_labels_used=False, purpose='Structure diagnosis only; not used by completed CE arms',
    semantic_unknown_count=None)
(output/'capacity.json').write_text(json.dumps(artifact, indent=2, allow_nan=False))
np.savez_compressed(output/'clusters.npz', centers=result['centers'],
    assignments=result['assignments'], target_paths=np.asarray(paths))

# This block evaluates the already persisted artifact. No labels are fed back
# into the estimator, thresholds, artifact or an experimental training run.
truth = np.asarray([int(line.rsplit(None,1)[1]) for line in
    Path('/content/osda-office31-a2w-v1/webcam_0-9_20-30_test.txt').read_text().splitlines() if line.strip()])
known = truth < 10
ids = result['assignments']
candidate = ids >= 10
assigned_known = (ids >= 0) & (ids < 10)
report = dict(task=artifact['task'], K=artifact['K'], new_gpu_training=False,
    new_image_forward=False, threshold_search=False, target_labels_posthoc_only=True,
    known_candidate_rate=float(candidate[known].mean()),
    unknown_candidate_rate=float(candidate[~known].mean()),
    unknown_absorbed_by_known_rate=float(assigned_known[~known].mean()),
    known_assigned_known_rate=float(assigned_known[known].mean()),
    known_identity_accuracy=float((ids[known] == truth[known]).mean()),
    noise_known=int(((ids == -1)&known).sum()), noise_unknown=int(((ids == -1)&~known).sum()),
    clusters=[], interpretation='These are capacities, not certified semantic classes. No target-driven deletion or forced K.')
for cluster in np.unique(ids):
    members = ids == cluster
    semantic, counts = np.unique(truth[members], return_counts=True)
    report['clusters'].append(dict(cluster=int(cluster), members=int(members.sum()),
        known=int((members&known).sum()), unknown=int((members&~known).sum()),
        semantics={str(int(label)):int(count) for label,count in zip(semantic,counts)}))
(output/'posthoc-composition.json').write_text(json.dumps(report, indent=2, allow_nan=False))
drive = Path('/content/drive/MyDrive/OSDA/runs/a2w-unknown-ce-10e-v1/imp-structure-probe-v1')
drive.mkdir(exist_ok=False)
for path in output.iterdir():
    shutil.copyfile(path, drive/path.name)
print('A2W_IMP_STRUCTURE_COMPLETE', json.dumps({key:value for key,value in report.items() if key!='clusters'}), flush=True)
print('CANDIDATE_COMPOSITION', json.dumps([row for row in report['clusters'] if row['cluster'] >= 10]), flush=True)
