"""Check order invariance, row correspondence, and default compatibility."""
import importlib.util
import hashlib
import json
from pathlib import Path
import torch

def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

path=Path('/content/source_anchored_imp_farthest_v1.py')
module=load('farthest',path)
old=load('sequential','/content/source_anchored_imp_relation_v1.py')
torch.set_num_threads(2)
source=torch.tensor([[-3.,0.],[3.,0.]])
target=torch.cat([torch.tensor([[-2.7,0.]]).repeat(20,1),torch.tensor([[3.3,0.]]).repeat(20,1),torch.tensor([[0.,4.]]).repeat(20,1)])
gate=torch.cat([torch.ones(40),torch.zeros(20)])
options=dict(threshold=1.,observation_variance=.04,prior_strength=5.,steps=5)
previous=old.SourceAnchoredIMP(**options).fit(target,source,gate)
default=module.SourceAnchoredIMP(**options).fit(target,source,gate)
assert torch.equal(previous['responsibilities'],default['responsibilities'])
baseline=module.SourceAnchoredIMP(**options).fit(target,source,gate,birth_strategy='farthest')
assert baseline['candidate_count']==1
assert baseline['responsibilities'][:20].argmax(1).eq(0).all()
assert baseline['responsibilities'][20:40].argmax(1).eq(1).all()
assert baseline['responsibilities'][40:].argmax(1).eq(2).all()
for seed in [1,2,3,4,5]:
    order=torch.randperm(len(target),generator=torch.Generator().manual_seed(seed))
    result=module.SourceAnchoredIMP(**options).fit(target[order],source,gate[order],birth_strategy='farthest')
    assert torch.equal(result['known_prototypes'],baseline['known_prototypes'])
    assert torch.equal(result['candidate_prototypes'],baseline['candidate_prototypes'])
    assert torch.equal(result['responsibilities'][order.argsort()],baseline['responsibilities'])
report=dict(module_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
    tests=['default regression','known identity/row correspondence','five permutations exact'],
    scope='synthetic geometry only; count correctness not established')
output=Path('/content/farthest-birth-unit-v1.json')
if output.exists():
    raise RuntimeError('Refusing overwrite')
output.write_text(json.dumps(report,indent=2))
print('FARTHEST_UNIT_PASS',json.dumps(report),flush=True)
