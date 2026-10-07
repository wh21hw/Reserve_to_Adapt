"""Targeted order/union/eligibility/actual-added-label-exposure check."""
from types import SimpleNamespace
import numpy as np
import torch
from online_imp_structure import OnlineStructure

s=OnlineStructure.__new__(OnlineStructure)
s.args=SimpleNamespace(shared_classes=2)
s.assignments=np.array([0,2,3,3,1,2]);s.reliable=np.array([True,False])
s.selection_mode='rta_only'
indices=np.arange(6);original=torch.tensor([3,0]).view(2,1,1)
assert torch.equal(s.select_unknown(indices,original),original.view(-1))
s.selection_mode='reliable_union'
union=s.select_unknown(indices,original)
assert union.tolist()==[3,0,1,5]
assert len(union.unique())==len(union)
s.use_labels=True;s.label_scope='screened';s.counts=np.zeros(2,dtype=np.int64)
s.selected=s.overridden=0
s.selected_by_sample=np.zeros(6,dtype=np.int32);s.overridden_by_sample=np.zeros(6,dtype=np.int32)
s.teacher_added_by_sample=np.zeros(6,dtype=np.int32)
labels=s.labels(indices[union.numpy()],torch.full((4,),2,dtype=torch.long))
assert labels.tolist()==[2,2,2,2]
assert s.teacher_added_by_sample.tolist()==[0,1,0,0,0,1]
assert s.teacher_added_by_sample.sum()==2
assert s.select_unknown(indices,torch.arange(6)).tolist()==list(range(6))
print('UNKNOWN_SUPPORT_ORIGINAL_ORDER_UNION_RELIABLE_ONLY_AND_ACTUAL_LABEL_EXPOSURE_OK')
