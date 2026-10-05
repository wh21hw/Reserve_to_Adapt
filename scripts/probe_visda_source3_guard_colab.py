"""Early guard probe using cached bottlenecks and CPU classifier layers only.

No image/ResNet forward, training, K fit, or target-based threshold selection.
"""
import json
from pathlib import Path
import sys
import numpy as np
import torch

sys.path.insert(0, '/content/known-guard-v1')
sys.path.insert(0, '/content/rta-legacy-l4-bridge-v1')
from reliable_known_guard import infer_protected_known
from networks import CLS

root = Path('/content/imp-runs/visda-frozenbn-capacity-10e-v1')
output = root / 'source3-known-guard-v1'
if output.exists():
    raise FileExistsError('Preserve previous probe; no sweep')
torch.set_num_threads(2)
checkpoint = torch.load(str(root/'source/source-final.pt'), map_location='cpu')
if checkpoint['config']['epochs'] != 3 or checkpoint['config']['known_classes'] != 6:
    raise ValueError('Require shared source3 C6 prior')
head = CLS(2048, 6)
head.load_state_dict({key[2:]: value for key, value in checkpoint['model'].items()
                      if key.startswith('1.')}, strict=True)
head.eval()


@torch.no_grad()
def cached_logits(features):
    # CLS.forward returns normalized bottleneck at index1, before head BN and
    # LeakyReLU. features.npz stores exactly this tensor: do NOT apply fc alone.
    values = []
    for start in range(0, len(features), 2048):
        feature = torch.from_numpy(features[start:start+2048])
        logits = head.main[1](feature) / head.temp
        if not torch.isfinite(logits).all():
            raise ValueError('Nonfinite cached head logits')
        values.append(logits.numpy())
    return np.concatenate(values)


with np.load(str(root/'source/features.npz'), allow_pickle=False) as cache:
    source, target, labels = cache['source'], cache['target'], cache['source_labels']
source_logits, target_logits = cached_logits(source), cached_logits(target)
guard = infer_protected_known(source, labels, source_logits, target, target_logits)
mask, identity = guard['protected_known'], guard['known_identity']
known = [1, 2, 3, 6, 10, 11]
rows = [row.rsplit(None, 1) for row in
        Path('/content/osda-visda-syn2real-v1/target-real-12.txt').read_text().splitlines() if row.strip()]
paths = np.array([r[0] for r in rows])
if len(paths) != len(mask):
    raise ValueError('Target ordering mismatch')
output.mkdir()
# Only label-free inferred information is eligible as a future training artifact.
np.savez_compressed(str(output/'guard.npz'), protected_known=mask,
                    known_identity=identity, target_paths=paths)
# Post-hoc semantic checks after guard construction; not passed back to guard.
target_raw = np.array([int(r[1]) for r in rows])
known_mask = np.isin(target_raw, known)
correct = np.zeros(len(mask), dtype=bool)
for c, raw in enumerate(known):
    correct |= (target_raw == raw) & (identity == c)
report = dict(stage='Source3 early cached-feature probe, not an RTA gain',
    CPU_head='Exact saved normalized bottleneck -> head BN -> LeakyReLU -> fc / temperature',
    quantiles=dict(radius=.99, confidence=.01),
    calibration='In-sample source per-class quantiles; not target-calibrated or finite-sample guarantee',
    target_labels_used_for_guard=False, target_labels_used_for_posthoc_diagnostics=True,
    radii=guard['radii'].tolist(), confidence=guard['confidence'].tolist(),
    protected_count=int(mask.sum()), protected_known_count=int((mask & known_mask).sum()),
    protected_unknown_count=int((mask & ~known_mask).sum()),
    known_coverage=float(mask[known_mask].mean()),
    unknown_false_protection=float(mask[~known_mask].mean()),
    known_identity_precision=float(correct[mask & known_mask].mean()) if (mask & known_mask).any() else None,
    caveat='Candidate early intervention only. Mask was not used to train either completed arm; no prevention or causal benefit established.')
(output/'report.json').write_text(json.dumps(report, indent=2, allow_nan=False))
print('SOURCE3_KNOWN_GUARD_PROBE', json.dumps(report), flush=True)
