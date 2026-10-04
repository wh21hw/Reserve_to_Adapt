"""Targeted check of blocked source birth-cost calculation; core checked separately."""
import importlib.util
from pathlib import Path
import sys
import numpy as np

path=Path(sys.argv[1]);sys.path.insert(0,str(path.parent))
spec=importlib.util.spec_from_file_location('source_cost_streaming',str(path))
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
captured=[]
def capture(target,anchors,radius,**kwargs):
    captured.append((anchors.copy(),radius,kwargs))
    return dict(converged=True,K=0)
module.fit_robust_capacity=capture
rng=np.random.RandomState(19)
s=np.concatenate([rng.normal([0.,0.,0.],.1,(24,3)),rng.normal([2.,0.,0.],.1,(27,3))])
y=np.repeat([0,1],[24,27]);target=s[:3]
_,full=module.estimate_source_cost_capacity(s,y,target)
_,blocked=module.estimate_source_cost_capacity(s,y,target,proposal_block_size=3)
assert np.allclose(captured[0][0],captured[1][0],rtol=0,atol=1e-12)
for key in ['lambda_radius','birth_cost','reference_samples']:
    assert np.isclose(full[key],blocked[key],rtol=0,atol=1e-12)
assert full['source_prior_counts']==blocked['source_prior_counts']
assert captured[1][2]['proposal_block_size']==3
print('SOURCE_COST_STREAMING_CHECK_OK',dict(source_rows=len(s),cost_rows=full['birth_cost_rows'],
    same_cost=True,block_forwarded=True),flush=True)
