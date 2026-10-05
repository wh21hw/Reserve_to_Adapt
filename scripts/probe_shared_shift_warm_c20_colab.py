"""Declared pre-birth initialization ablation, source-only cached features."""
import importlib.util
import json
from pathlib import Path
import numpy as np

spec = importlib.util.spec_from_file_location('warm_capacity_v1', '/content/shared-shift-warm-code-v1/robust_capacity.py')
core = importlib.util.module_from_spec(spec)
spec.loader.exec_module(core)
root = Path('/content/imp-runs/source-frozenbn-block0to4-v1')
output = root/'shared-shift-warm-probe-v1.json'
if output.exists():
    raise FileExistsError('Preserve completed source proxy')
old = json.loads((root/'shared-shift-probe-v1.json').read_text())
prior = json.loads((root/'prior-mass-probe-v1.json').read_text())
settings = prior['arms']['source_counts']['known_only']['settings']
with np.load(root/'source-frozenbn/source/features.npz') as f:
    source, labels, all_source = f['source'].astype(np.float64), f['source_labels'], f['target'].astype(np.float64)
y = np.array([int(s.rsplit(None, 1)[1]) for s in Path('/content/osda-officehome-pr2rw-v1/product_0-24_train_all.txt').read_text().splitlines() if s.strip()])
rng = np.random.RandomState(2026)
train, test, hidden = [], [], []
for cls in range(20):
    ids = np.flatnonzero(labels == cls); rng.shuffle(ids)
    cut = max(1, int(.7*len(ids))); middle = cut+(len(ids)-cut)//2
    train.extend(ids[:cut]); test.extend(ids[middle:])
for cls in range(5):
    ids = np.flatnonzero(y == cls); rng.shuffle(ids)
    cut = max(1, int(.7*len(ids))); middle = cut+(len(ids)-cut)//2
    hidden.extend(ids[middle:])
train, test, hidden = map(np.asarray, (train, test, hidden))
anchors = np.stack([source[train][labels[train] == c].mean(0) for c in range(20)])
translation = np.array(old['translation'])
report = dict(source_only=True, real_target_used=False, training_changed=False,
    changed_factor='K0 known/shift optimization before allowing births; identical capped objective',
    known_rows=len(test), hidden_rows=len(hidden), settings=settings,
    baseline_reused='shared-shift-probe-v1.json', arms={},
    caveat='Known encoder-training images, artificial shift and existing hidden pressure block; not independent real-domain validation. Unknowns can bias warm start. No parameter scan or target labels.')
for scenario, offset in [('unshifted',np.zeros_like(translation)), ('translated',translation)]:
    for pool, x in [('known_only',source[test]), ('mixed',np.vstack([source[test],all_source[hidden]]))]:
        r = core.fit_robust_capacity(x+offset, anchors, settings['lambda_radius'],
            prior_strength=settings['source_prior_counts'], reference_samples=settings['reference_samples'],
            birth_penalty=settings['birth_cost'], birth_order='before_update', proposal_block_size=64,
            shared_shift_precision=settings['reference_samples'], shared_shift_warm_start=True)
        if not r['converged']:
            raise RuntimeError('Unconverged warm-start proxy; do not retry with changed budget')
        ids = r['assignments']
        row = dict(K=r['K'], converged=r['converged'], noise_count=r['noise_count'],
            known_false_candidate_rate=float((ids[:len(test)] >= 20).mean()),
            known_identity_accuracy=float((ids[:len(test)] == labels[test]).mean()),
            shift_norm=float(np.linalg.norm(r['shared_shift'])),
            shift_recovery_error=float(np.linalg.norm(np.array(r['shared_shift'])-offset)),
            warm_start_history=r['warm_start_history'], history=r['history'])
        if pool == 'mixed':
            row['hidden_candidate_recall'] = float((ids[len(test):] >= 20).mean())
        key = scenario+'/'+pool
        report['arms'][key] = dict(baseline=old['arms'][key+'/shared_shift'], warm_start=row)
        print('WARM_PROXY_ARM',key,json.dumps({k:v for k,v in row.items() if k not in ('history','warm_start_history')}),flush=True)
output.write_text(json.dumps(report,indent=2,allow_nan=False))
print('WARM_PROXY_COMPLETE',str(output),flush=True)
