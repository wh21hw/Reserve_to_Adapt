"""Synthetic relation-prior tests, including default-behavior regression."""
import hashlib
import importlib.util
import json
from pathlib import Path
import torch

path = Path('/content/source_anchored_imp_relation_v1.py')
spec = importlib.util.spec_from_file_location('constrained', path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
old_spec = importlib.util.spec_from_file_location('old', '/content/source_anchored_imp.py')
old = importlib.util.module_from_spec(old_spec)
old_spec.loader.exec_module(old)
torch.set_num_threads(2)
source = torch.zeros(1,2)
target = torch.cat([torch.tensor([[2.,0.]]).repeat(20,1), torch.tensor([[0.,4.]]).repeat(20,1)])
options = dict(threshold=1., observation_variance=.04, prior_strength=5., steps=5)
baseline = old.SourceAnchoredIMP(**options).fit(target, source)
neutral = module.SourceAnchoredIMP(**options).fit(target, source)
for field in ['known_prototypes','candidate_prototypes','responsibilities']:
    assert torch.equal(baseline[field], neutral[field])
# Relation says shifted first group remains known, second may be novel.
compatibility = torch.cat([torch.ones(20), torch.zeros(20)])
result = module.SourceAnchoredIMP(**options).fit(target, source, compatibility)
assert result['candidate_count'] == 1
assert result['responsibilities'][:20,0].eq(1).all()
assert result['responsibilities'][20:,1].eq(1).all()
assert torch.isfinite(result['responsibilities']).all()
assert torch.allclose(result['known_prototypes'][0], torch.tensor([1.6,0.]))
all_known = module.SourceAnchoredIMP(**options).fit(target, source, torch.ones(40))
assert all_known['candidate_count'] == 0 and torch.isfinite(all_known['responsibilities']).all()
invalid = False
try:
    module.SourceAnchoredIMP(**options).fit(target, source, torch.full((40,),float('nan')))
except ValueError:
    invalid = True
assert invalid
report = dict(module_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
    tests=['default bitwise regression','shifted known vs novel exact gate','all-known finite','invalid gate rejected'],
    warning='Perfect synthetic compatibility is supplied, not learned; not a real unknown classifier')
output = Path('/content/relation-constrained-unit-v1.json')
if output.exists():
    raise RuntimeError('Refusing overwrite')
output.write_text(json.dumps(report,indent=2))
print('RELATION_CONSTRAINED_UNIT_PASS',json.dumps(report),flush=True)
