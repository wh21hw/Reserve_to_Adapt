"""Offline attribution of fixed candidates. Labels never feed inference or selection."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--inference', default='/content/imp-runs/source-anchored-stability-v1/original.npz')
parser.add_argument('--labels', default='/content/imp-runs/source-warmup-l4-seed1-v2/evaluation-only.npz')
parser.add_argument('--output', default='/content/source-anchored-attribution-v1.json')
args = parser.parse_args()
output = Path(args.output)
if output.exists():
    raise RuntimeError('Refusing overwrite')
inference_path, label_path = Path(args.inference), Path(args.labels)
inference = np.load(inference_path, allow_pickle=False)
labels = np.load(label_path, allow_pickle=False)['target_labels']
probabilities = inference['responsibilities']
assert len(probabilities) == len(labels) and np.isfinite(probabilities).all()
assert set(np.unique(labels)) <= set(range(10)) | set(range(20, 31))
unknown = labels >= 20
assigned = probabilities.argmax(1)
candidate = assigned >= 10
tp, fp, fn = int((candidate & unknown).sum()), int((candidate & ~unknown).sum()), int((~candidate & unknown).sum())
components = []
for index in range(10, probabilities.shape[1]):
    mask = assigned == index
    values, counts = np.unique(labels[mask], return_counts=True)
    components.append(dict(index=int(index-10), hard_samples=int(mask.sum()),
        soft_mass=float(probabilities[:, index].sum()),
        known_samples=int((mask & ~unknown).sum()), unknown_samples=int((mask & unknown).sum()),
        semantic_composition={str(int(label)): int(count) for label, count in zip(values, counts)}))
report = dict(stage='offline attribution; not a capacity selection rule or OSDA score',
    inference_sha256=hashlib.sha256(inference_path.read_bytes()).hexdigest(),
    labels_sha256=hashlib.sha256(label_path.read_bytes()).hexdigest(),
    total_samples=len(labels), known_samples=int((~unknown).sum()), unknown_samples=int(unknown.sum()),
    candidate_precision=tp/(tp+fp) if tp+fp else None,
    candidate_unknown_recall=tp/(tp+fn) if tp+fn else None,
    known_false_candidate_rate=fp/int((~unknown).sum()),
    components=components)
output.write_text(json.dumps(report, indent=2, allow_nan=False))
print(json.dumps(report, indent=2))
