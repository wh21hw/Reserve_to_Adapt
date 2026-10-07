"""Targeted objective monotonicity, separated-cluster and known identity check."""
import torch
from dpmeans_merge import merge_candidates

x=torch.tensor([[0.,0.],[-3.,0.],[-3.05,0.],[3.,0.],[3.05,0.]])
known=torch.tensor([[0.,0.]])
candidates=x[1:].clone();labels=[0,1,2,3,4]
merged,ids,report=merge_candidates(x,known,candidates,labels,.1)
assert len(merged)==2 and ids[0]==0
assert torch.equal(known,torch.zeros(1,2))
assert all(a>b for a,b in zip(report['objective_history'],report['objective_history'][1:]))
separate,ids,report=merge_candidates(x,known,candidates,labels,1e-5)
assert len(separate)==4 and len(report['merge_events'])==0
print('MERGE_OBJECTIVE_MONOTONE_KNOWN_IDENTITY_AND_NO_FORCED_MERGE_OK')
