from types import SimpleNamespace
import numpy as np
import torch
from online_imp_structure import OnlineStructure

b=object.__new__(OnlineStructure);b.args=SimpleNamespace(shared_classes=2)
b.assignments=np.array([0,2,3]);b.reliable=np.array([True,False]);b.known_scope='entropy'
for eligibility,expected in [('raw',[1.,0.,0.]),('screened',[1.,0.,1.])]:
    b.veto_eligibility=eligibility
    for name in ('known_original_by_sample','known_effective_by_sample','entropy_effective_by_sample','alignment_effective_by_sample'):
        setattr(b,name,np.zeros(3,dtype=np.float32))
    e,a=b.objective_weights(np.arange(3),torch.ones(3))
    assert e.tolist()==expected and a.tolist()==[1.,1.,1.]
print('ENTROPY_VETO_ELIGIBILITY_INTERFACE_OK; alignment preserved')
