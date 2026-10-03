"""Semantic outputs/loss/shared-logit gradients invariant to component splitting."""
import hashlib
import json
from pathlib import Path
import sys
import torch

sys.path.insert(0,'/content/rta-hierarchical-pilot-v1')
from hierarchical_unknown import marginalize_unknown,semantic_entropy

torch.manual_seed(1)
raw=torch.randn(9,12,dtype=torch.float64,requires_grad=True)
prior=raw.new_full((2,),-torch.log(raw.new_tensor(2.)))
group,conditional=marginalize_unknown(raw,10,prior)
duplicate=torch.cat([raw[:,:10],raw[:,10:].repeat_interleave(9,1)],1)
split_prior=prior.repeat_interleave(9)-torch.log(raw.new_tensor(9.))
split,split_conditional=marginalize_unknown(duplicate,10,split_prior)
assert torch.allclose(group,split,atol=1e-14,rtol=1e-14)
assert torch.equal(group.argmax(1),split.argmax(1))
assert torch.allclose(conditional.exp().sum(1),raw.new_ones(9),atol=1e-14,rtol=0)
truth=torch.arange(9)%10
weights=torch.linspace(0,1,9,dtype=raw.dtype)
virtual=torch.randn(9,10,dtype=raw.dtype)
def losses(logits):
    return torch.stack([torch.nn.functional.cross_entropy(logits,truth),
        torch.nn.functional.cross_entropy(logits,torch.full((9,),10,dtype=torch.long)),
        semantic_entropy(logits,weights),
        torch.nn.functional.cross_entropy(torch.cat([logits,virtual],1),truth)])
old_losses=losses(group); new_losses=losses(split)
assert torch.allclose(old_losses,new_losses,atol=1e-14,rtol=1e-14)
old_gradient=torch.autograd.grad(old_losses.sum(),raw,retain_graph=True)[0]
new_gradient=torch.autograd.grad(new_losses.sum(),raw)[0]
assert torch.allclose(old_gradient,new_gradient,atol=1e-14,rtol=1e-14)
# Non-uniform component splitting also preserves total semantic evidence.
asymmetric=raw.new_tensor([.25,.75]).log()
partial=torch.cat([raw[:,:10],raw[:,10:11].repeat(1,3),raw[:,11:]],1)
partial_prior=torch.cat([asymmetric[:1]-torch.log(raw.new_tensor(3.)),asymmetric[:1]-torch.log(raw.new_tensor(3.)),asymmetric[:1]-torch.log(raw.new_tensor(3.)),asymmetric[1:]])
g1,_=marginalize_unknown(raw,10,asymmetric)
g2,_=marginalize_unknown(partial,10,partial_prior)
assert torch.allclose(g1,g2,atol=1e-14,rtol=1e-14)
extreme=torch.tensor([[1000.,-1000.,-1000.,1000.]],requires_grad=True)
g,_=marginalize_unknown(extreme,2,torch.full((2,),-torch.log(torch.tensor(2.))))
stress=torch.nn.functional.cross_entropy(g,torch.tensor([2]))+semantic_entropy(g)
stress.backward()
assert torch.isfinite(stress) and torch.isfinite(extreme.grad).all()
report=dict(semantic_logits_invariant=True,semantic_predictions_invariant=True,
    source_ce_unknown_ce_entropy_virtual_ce_invariant=True,shared_logit_gradient_invariant=True,
    nonuniform_split_invariant=True,conditional_row_sum_one=True,extreme_finite=True,
    optimizer_steps=0,target_labels_read=False,
    limitation='Independent duplicated-parameter SGD trajectories not asserted invariant',
    module_sha256=hashlib.sha256(Path('/content/rta-hierarchical-pilot-v1/hierarchical_unknown.py').read_bytes()).hexdigest())
with Path('/content/hierarchical-unit-v1.json').open('x') as stream:
    json.dump(report,stream,indent=2,allow_nan=False)
print('HIERARCHICAL_UNIT_PASS',json.dumps(report),flush=True)
