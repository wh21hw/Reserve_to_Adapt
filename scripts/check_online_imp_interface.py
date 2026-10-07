"""One targeted functional check: identity permutation, resize and labels."""
import numpy as np
import torch
from torch import nn
from online_imp_structure import transport_head, OnlineStructure, infer_structure

class Head(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc=nn.Linear(3,4,bias=False)
        self.main=nn.Sequential(nn.Identity(),nn.Sequential(nn.Identity(),nn.Identity(),self.fc))

head=Head()
optimizer=torch.optim.SGD(head.parameters(),lr=.1,momentum=.9)
original=head.fc.weight.detach().clone()
optimizer.state[head.fc.weight]['momentum_buffer']=torch.arange(12.).reshape(4,3)
previous=np.array([2,2,3,3,0,1])
current=np.array([3,3,2,2,0,1])
transport_head(head,optimizer,2,current,previous,torch.zeros(6,4))
assert torch.equal(head.fc.weight[:2],original[:2])
assert torch.equal(head.fc.weight[2],original[3])
assert torch.equal(head.fc.weight[3],original[2])
assert torch.equal(optimizer.state[head.fc.weight]['momentum_buffer'][2],torch.tensor([9.,10.,11.]))
grown=np.array([2,2,3,3,4,0])
transport_head(head,optimizer,2,grown,current,torch.zeros(6,4))
assert head.fc.out_features==5 and head.main[1][2] is head.fc
class Args:
    shared_classes=2
bridge=object.__new__(OnlineStructure)
bridge.args=Args();bridge.use_labels=True;bridge.assignments=grown
bridge.reliable=np.array([True,False,True]);bridge.counts=np.zeros(3,dtype=np.int64)
bridge.selected=bridge.overridden=0
labels=bridge.labels(np.array([0,2,4]),torch.tensor([4,4,2]))
assert labels.tolist()==[2,4,4] and bridge.overridden==2
# Actual cached features may be supplied by launcher; no targetsemantic column.
import os
if os.environ.get('ONLINE_FEATURE_CACHE'):
    with np.load(os.environ['ONLINE_FEATURE_CACHE']) as cache:
        assignments,candidates,reliable,report=infer_structure(torch.from_numpy(cache['source']).float(),
            torch.from_numpy(cache['source_labels']).long(),torch.from_numpy(cache['target']).float(),
            torch.from_numpy(cache['target_logits']).float(),10)
    assert assignments.shape==(564,) and len(candidates)==report['K']
    print('ONLINE_IMP_CURRENT_FEATURE_CHECK',report)
print('ONLINE_IMP_INTERFACE_CHECK_OK')
