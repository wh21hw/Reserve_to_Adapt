"""Source-hidden-class proxy for cluster-supported known protection.

No real target data. Fixed existing split, radius/birth cost and translations;
no threshold grid, image forwards or new training. Proxy labels score only.
"""
import importlib.util
import json
from pathlib import Path
import sys
import numpy as np
import torch

sys.path.insert(0, '/content/known-guard-v1')
sys.path.insert(0, '/content/rta-legacy-l4-bridge-v1')
from reliable_known_guard import infer_protected_known, cluster_supported_known
from networks import CLS
spec = importlib.util.spec_from_file_location('guard_proxy_capacity', '/content/shared-shift-warm-code-v1/robust_capacity.py')
core = importlib.util.module_from_spec(spec)
spec.loader.exec_module(core)
root = Path('/content/imp-runs/source-frozenbn-block0to4-v1')
output = root/'cluster-known-guard-proxy-v1.json'
if output.exists():
    raise FileExistsError('Preserve source proxy')
torch.set_num_threads(2)
prior = json.loads((root/'prior-mass-probe-v1.json').read_text())
settings = prior['arms']['source_counts']['known_only']['settings']
shift = np.array(json.loads((root/'shared-shift-probe-v1.json').read_text())['translation'])
with np.load(str(root/'source-frozenbn/source/features.npz'), allow_pickle=False) as f:
    source, labels, pool = f['source'], f['source_labels'], f['target']
if source.shape != (1458, 256) or pool.shape != (1785, 256):
    raise ValueError('Expected existing C20 hidden0..4 proxy')
checkpoint = torch.load(str(root/'source-frozenbn/source/source-final.pt'), map_location='cpu')
head = CLS(2048, 20)
head.load_state_dict({k[2:]: v for k, v in checkpoint['model'].items() if k.startswith('1.')}, strict=True)
head.eval()


@torch.no_grad()
def logits(x):
    return (head.main[1](torch.from_numpy(np.asarray(x, dtype=np.float32))) / head.temp).numpy()


y = np.array([int(r.rsplit(None, 1)[1]) for r in
              Path('/content/osda-officehome-pr2rw-v1/product_0-24_train_all.txt').read_text().splitlines() if r.strip()])
rng = np.random.RandomState(2026)
train, test, hidden = [], [], []
for c in range(20):
    ids = np.flatnonzero(labels == c); rng.shuffle(ids)
    cut = max(1, int(.7 * len(ids))); middle = cut + (len(ids)-cut)//2
    train.extend(ids[:cut]); test.extend(ids[middle:])
for c in range(5):
    ids = np.flatnonzero(y == c); rng.shuffle(ids)
    cut = max(1, int(.7 * len(ids))); middle = cut + (len(ids)-cut)//2
    hidden.extend(ids[middle:])
train, test, hidden = map(np.asarray, (train, test, hidden))
anchors = np.stack([source[train][labels[train] == c].mean(0) for c in range(20)])
source_logits = logits(source[train])
report = dict(source_only=True, real_target_used=False, new_training=False,
    settings='Fixed .99 radius, .01 confidence, strict >.5 cluster support; no scan',
    caveat='Known proxy images were seen by encoder; hidden raw0..4 were not. Artificial translation, not real-domain validation.', arms={})
for scenario, offset in [('unshifted', np.zeros_like(shift)), ('translated', shift)]:
    x = np.vstack([source[test], pool[hidden]]) + offset
    guard = infer_protected_known(source[train], labels[train], source_logits, x, logits(x))
    clustering = core.fit_robust_capacity(x, anchors, settings['lambda_radius'],
        prior_strength=settings['source_prior_counts'], reference_samples=settings['reference_samples'],
        birth_penalty=settings['birth_cost'], birth_order='before_update', proposal_block_size=64)
    if not clustering['converged']:
        raise RuntimeError('Unconverged clustering; preserve, no retry')
    supported, details = cluster_supported_known(guard, clustering['assignments'])
    rows = {}
    for name, mask in [('point_agreement', guard['protected_known']), ('cluster_supported', supported)]:
        covered = mask[:len(test)]
        rows[name] = dict(known_coverage=float(covered.mean()),
            hidden_false_protection=float(mask[len(test):].mean()),
            known_identity_precision=float((guard['known_identity'][:len(test)][covered] == labels[test][covered]).mean()) if covered.any() else None,
            protected_count=int(mask.sum()))
    report['arms'][scenario] = dict(K=clustering['K'], scores=rows, cluster_support=details)
    print('CLUSTER_GUARD_PROXY_ARM', scenario, json.dumps(rows), flush=True)
output.write_text(json.dumps(report, indent=2, allow_nan=False))
print('CLUSTER_GUARD_PROXY_COMPLETE', str(output), flush=True)
