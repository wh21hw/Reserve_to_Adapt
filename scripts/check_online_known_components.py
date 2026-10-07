"""One targeted weight-routing check for entropy vs alignment."""
from types import SimpleNamespace
import numpy as np
import torch
from online_imp_structure import OnlineStructure

b=object.__new__(OnlineStructure);b.args=SimpleNamespace(shared_classes=2);b.assignments=np.array([0,2,1])
for scope in ('none','entropy','alignment','both'):
    b.known_scope=scope
    for name in ('known_original_by_sample','known_effective_by_sample','entropy_effective_by_sample','alignment_effective_by_sample'):
        setattr(b,name,np.zeros(3,dtype=np.float32))
    original=torch.tensor([1.,1.,0.]);e,a=b.objective_weights(np.arange(3),original)
    assert e.tolist()==([1.,0.,0.] if scope in ('entropy','both') else [1.,1.,0.])
    assert a.tolist()==([1.,0.,0.] if scope in ('alignment','both') else [1.,1.,0.])
    assert original.tolist()==[1.,1.,0.]
print('KNOWN_OBJECTIVE_COMPONENT_ROUTING_OK')
