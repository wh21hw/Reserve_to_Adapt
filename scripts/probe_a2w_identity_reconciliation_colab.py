"""One label-blind matching candidate on cached features; no GPU or training."""
import json
from pathlib import Path
import shutil
import sys
import numpy as np
import torch

sys.path.insert(0, '/content')
sys.path.insert(0, '/content/rta-legacy-l4-bridge-v1')
from prototype_identity_reconciliation import reconcile_known_identities
from networks import CLS

torch.set_num_threads(2)
root = Path('/content/imp-runs/a2w-unknown-ce-10e-v1')
output = root/'identity-reconciliation-probe-v1'
if output.exists():
    raise FileExistsError('Preserve prior probe')
with np.load(root/'source/features.npz', allow_pickle=False) as data:
    target = data['target']
with np.load(root/'imp-structure-probe-v1/clusters.npz', allow_pickle=False) as data:
    ids, paths = data['assignments'], data['target_paths']
checkpoint = torch.load(root/'source/source-final.pt', map_location='cpu')
head = CLS(2048,10)
head.load_state_dict({key[2:]:value for key,value in checkpoint['model'].items() if key.startswith('1.')}, strict=True)
head.eval()
with torch.no_grad():
    # Features are already normalized bottleneck values. Preserve actual head
    # BN -> LeakyReLU -> fc -> temperature, not a bare fc multiplication.
    logits = (head.main[1](torch.from_numpy(target)) / head.temp).numpy()
result = reconcile_known_identities(ids, logits, 10)
output.mkdir()
np.savez_compressed(output/'clusters.npz', assignments=result['assignments'], target_paths=paths)
settings = {key:value for key,value in result.items() if key!='assignments'}
(output/'settings.json').write_text(json.dumps(settings, indent=2, allow_nan=False))

# Evaluation is separate and after persisting the complete label-free result.
truth = np.asarray([int(line.rsplit(None,1)[1]) for line in
    Path('/content/osda-office31-a2w-v1/webcam_0-9_20-30_test.txt').read_text().splitlines() if line.strip()])
known = truth < 10
assigned = result['assignments']
report = dict(K=result['K'], occupied_clusters=result['occupied_clusters'],
    known_candidate_rate=float((assigned[known]>=10).mean()),
    unknown_candidate_rate=float((assigned[~known]>=10).mean()),
    unknown_absorbed_by_known_rate=float(((assigned[~known]>=0)&(assigned[~known]<10)).mean()),
    known_identity_accuracy=float((assigned[known]==truth[known]).mean()),
    cluster_to_identity=result['cluster_to_identity'],
    new_training=False, new_image_forward=False, target_labels_posthoc_only=True,
    caveat='Single candidate diagnostic, not certified semantics, training improvement, or target-tuned K; all-known matching can misassign unknown/mixed clusters')
(output/'posthoc-composition.json').write_text(json.dumps(report, indent=2, allow_nan=False))
drive = Path('/content/drive/MyDrive/OSDA/runs/a2w-unknown-ce-10e-v1/identity-reconciliation-probe-v1')
drive.mkdir(exist_ok=False)
for path in output.iterdir():
    shutil.copyfile(path,drive/path.name)
print('A2W_IDENTITY_RECONCILIATION_COMPLETE', json.dumps(report), flush=True)
