"""Algebra/gradient controls, no target-label read or optimizer steps."""
import hashlib
import json
from pathlib import Path
import sys
import torch

sys.path.insert(0,'/content/rta-capacity-prior-pilot-v1')
from capacity_prior import uniform_capacity_log_prior, apply_capacity_log_prior

torch.manual_seed(1)
raw = torch.randn(7,12,dtype=torch.float64,requires_grad=True)
zero = uniform_capacity_log_prior(10,2,dtype=raw.dtype)
assert torch.equal(apply_capacity_log_prior(raw,zero),raw)
repeat = torch.cat([raw[:,:10],raw[:,10:].repeat_interleave(9,dim=1)],1)
prior = uniform_capacity_log_prior(10,18,dtype=raw.dtype)
old = raw.softmax(1)
new = apply_capacity_log_prior(repeat,prior).softmax(1)
assert torch.allclose(old[:,:10],new[:,:10],atol=1e-14,rtol=1e-14)
assert torch.allclose(old[:,10:].sum(1),new[:,10:].sum(1),atol=1e-14,rtol=1e-14)
gradient_old=torch.autograd.grad(old[:,10:].sum(),raw,retain_graph=True)[0]
gradient_new=torch.autograd.grad(new[:,10:].sum(),raw,retain_graph=True)[0]
assert torch.allclose(gradient_old,gradient_new,atol=1e-14,rtol=1e-14)
uncorrected=repeat.softmax(1)[:,10:].sum(1)
assert (uncorrected > old[:,10:].sum(1)).all()
entropy_old=-(old*old.log()).sum(1)
entropy_new=-(new*new.log()).sum(1)
expected_delta=old[:,10:].sum(1)*torch.log(torch.tensor(9.,dtype=raw.dtype))
assert torch.allclose(entropy_new-entropy_old,expected_delta,atol=1e-14,rtol=1e-14)
extreme=torch.tensor([[1000.,-1000.,-1000.,1000.]],dtype=torch.float32,requires_grad=True)
extreme_probability=apply_capacity_log_prior(extreme,uniform_capacity_log_prior(2,2)).softmax(1)
extreme_probability[:,2:].sum().backward()
assert torch.isfinite(extreme_probability).all() and torch.isfinite(extreme.grad).all()
rejected=0
for count in (0,-1,True,1.5):
    try:
        uniform_capacity_log_prior(10,count)
    except ValueError:
        rejected+=1
assert rejected==4
report=dict(k2_identity=True,duplicate_group_probability_invariant=True,
    duplicate_known_probability_invariant=True,duplicate_group_gradient_invariant=True,
    finite_extreme=True,invalid_inputs_rejected=rejected,optimizer_steps=0,target_labels_read=False,
    raw_unknown_probability=old[:,10:].sum(1).tolist(),
    duplicated_uncorrected_unknown_probability=uncorrected.tolist(),
    entropy_still_capacity_dependent=True,duplicate_entropy_delta=expected_delta.tolist(),
    module_sha256=hashlib.sha256(Path('/content/rta-capacity-prior-pilot-v1/capacity_prior.py').read_bytes()).hexdigest())
with Path('/content/capacity-prior-unit-v1.json').open('x') as stream:
    json.dump(report,stream,indent=2,allow_nan=False)
print('CAPACITY_PRIOR_UNIT_PASS',json.dumps(report),flush=True)
