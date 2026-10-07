"""One new-interface check: unknown identity scope and isolated head-init RNG."""
from types import SimpleNamespace
import numpy as np
import torch
from torch import nn
from online_imp_structure import OnlineStructure,transport_head

class Head(nn.Module):
    def __init__(self):
        super().__init__();self.fc=nn.Linear(3,4,bias=False)
        self.main=nn.Sequential(nn.Identity(),nn.Sequential(nn.Identity(),nn.Identity(),self.fc))

head=Head();opt=torch.optim.SGD(head.parameters(),lr=.1,momentum=.9)
before=torch.get_rng_state().clone()
transport_head(head,opt,2,np.array([2,2,3,3]),None,torch.zeros(4,4))
assert torch.equal(before,torch.get_rng_state())
for scope,expected in [('screened',[2,2,3]),('all_candidates',[2,3,3])]:
    bridge=object.__new__(OnlineStructure)
    bridge.args=SimpleNamespace(shared_classes=2);bridge.use_labels=True;bridge.label_scope=scope
    bridge.assignments=np.array([2,3,0]);bridge.reliable=np.array([True,False])
    bridge.counts=np.zeros(2,dtype=np.int64);bridge.selected=bridge.overridden=0
    result=bridge.labels(np.array([0,1,2]),torch.tensor([3,2,3]))
    assert result.tolist()==expected and (result>=2).all()
print('ONLINE_LABEL_COVERAGE_INTERFACE_OK; selected rows/status unchanged, RNG preserved')
