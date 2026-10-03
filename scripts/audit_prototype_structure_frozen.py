"""Teacher/prior audit on fixed real features; no target labels or SGD steps."""
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import torch

sys.path.insert(0, '/content')
from prototype_structure import dirichlet_log_prior, geometric_log_teacher

torch.set_num_threads(2)
feature_path = Path('/content/imp-runs/rta-space-warmup-l4-features-v1/features.npz')
proposal_path = Path('/content/imp-runs/farthest-constrained-frozen-v1/gate-seed1-original.npz')
assert hashlib.sha256(feature_path.read_bytes()).hexdigest() == '75cabbddd74d812dd9780b95248e1cc7dc8147d82ac336659162a2f642241823'
assert hashlib.sha256(proposal_path.read_bytes()).hexdigest() == '5bfd8dd4e51d0f7a818fa240b526ed942c3f61ebb63fca48f3bc4a4145aff9c2'
features = np.load(feature_path, allow_pickle=False)
proposal = np.load(proposal_path, allow_pickle=False)
assert 'target_labels' not in features.files
source = torch.from_numpy(features['source']).double()
target = torch.from_numpy(features['target']).double()
labels = torch.from_numpy(features['source_labels']).long()
source_centers = torch.stack([source[labels == index].mean(0) for index in range(10)])
variance = float((source - source_centers[labels]).square().mean().clamp_min(1e-8))
counts = proposal['responsibilities'].sum(0)[10:]
selected = np.array(sorted(np.flatnonzero(counts >= 5), key=lambda index: (-counts[index], index)))
assert len(selected) == 18
log_prior = dirichlet_log_prior(torch.as_tensor(counts[selected], dtype=torch.float64), concentration=1.)
centers = torch.as_tensor(proposal['candidates'][selected], dtype=torch.float64)
teacher = geometric_log_teacher(target, centers, log_prior, variance)
probabilities = teacher.exp()
assert torch.isfinite(teacher).all()
assert torch.allclose(probabilities.sum(1), torch.ones(len(target), dtype=torch.float64))
entropy = -(probabilities * teacher).sum(1)
confidence = probabilities.max(1).values
report = dict(target_samples=len(target), candidate_components=18,
              semantic_unknown_count_inferred=False, source_calibrated_variance=variance,
              selected_proposal_indices=selected.tolist(), concentration=1.,
              frozen_expected_counts=counts[selected].tolist(), prior_weights=log_prior.exp().tolist(),
              teacher_mean_entropy=float(entropy.mean()),
              teacher_confidence_quantiles=torch.quantile(confidence, torch.tensor([0., .5, .9, 1.], dtype=torch.float64)).tolist(),
              teacher_hard_occupancy=torch.bincount(teacher.argmax(1), minlength=18).tolist(),
              target_labels_read=False, optimizer_steps=0,
              interpretation='All-target conditional geometry diagnostic; not unknown selection, class recovery or performance evidence')
with Path('/content/prototype-structure-frozen-v1.json').open('x') as stream:
    json.dump(report, stream, indent=2, allow_nan=False)
print('PROTOTYPE_STRUCTURE_FROZEN_PASS', json.dumps(report), flush=True)
