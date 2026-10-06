"""Post-training label-exposure bound from fixed artifacts and usage logs.

No model forward or rule search; truth is used only after training completes.
Slot counts include fallback, so subtract ALL fallback for a conservative bound.
"""
import json
from pathlib import Path
import numpy as np

root = Path('/content/imp-runs/a2w-reconciled-identity-10e-v1')
destination = root/'posthoc-identity-risk.json'
if destination.exists():
    raise FileExistsError('Preserve earlier diagnosis')
run = root/'identity/office31-a2w_seed3'
history = [json.loads(line) for line in (run/'history.jsonl').read_text().splitlines()]
if [row['epoch'] for row in history] != list(range(1,11)):
    raise ValueError('Require completed training before inspecting target truth')
usage = [json.loads(line) for line in (run/'cluster-identity-history.jsonl').read_text().splitlines()]
with np.load('/content/imp-runs/a2w-unknown-ce-10e-v1/identity-reconciliation-probe-v1/clusters.npz',allow_pickle=False) as data:
    ids, paths = data['assignments'],data['target_paths'].tolist()
rows = [line.rsplit(None,1) for line in Path('/content/osda-office31-a2w-v1/webcam_0-9_20-30_test.txt').read_text().splitlines() if line.strip()]
if [row[0] for row in rows] != paths:
    raise ValueError('Sample order differs')
truth = np.asarray([int(row[1]) for row in rows])
details = []
pure_known = []
for cluster in range(10,18):
    members = ids == cluster
    known = int((truth[members]<10).sum())
    if known == int(members.sum()) and known > 0:
        pure_known.append(cluster)
    details.append(dict(cluster=cluster,members=int(members.sum()),known=known,
        unknown=int(members.sum())-known))
bound, fallback, selected = 0,0,0
exposure = np.zeros(8,dtype=np.int64)
for row in usage:
    selected += row['selected']
    fallback += row['fallback']
    exposure += np.asarray(row['pseudo_slot_counts'])
    if row['cluster_to_slot'] is None:
        continue
    slots = [row['cluster_to_slot'][cluster-10]-10 for cluster in pure_known]
    bound += max(0,sum(row['pseudo_slot_counts'][slot] for slot in slots)-row['fallback'])
report = dict(task='Office31 A->W',target_labels_posthoc_only=True,
    new_model_forward=False,new_training=False,configuration_changed=False,
    candidate_composition=details,pure_known_candidate_clusters=pure_known,
    unknown_label_exposures=selected,original_argmax_fallback_exposures=fallback,
    total_slot_label_exposures=exposure.tolist(),
    guaranteed_known_as_unknown_cluster_label_exposures_lower_bound=bound,
    lower_bound_fraction=bound/selected if selected else None,
    method='For pure-known candidate cluster slots, subtract all fallback labels per epoch; sum positive remaining counts',
    interpretation='Repeated training exposures, not unique images. Mixed clusters not counted. No comparable argmax-arm trace, so cannot claim increased selection error versus control. Both arms retain the same original candidate-selection mechanism.')
destination.write_text(json.dumps(report,indent=2,allow_nan=False))
print('A2W_IDENTITY_POSTHOC_RISK',json.dumps(report),flush=True)
