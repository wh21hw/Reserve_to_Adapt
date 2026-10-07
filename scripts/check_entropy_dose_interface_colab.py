"""One targeted weight-interface check; no network or training."""
import sys
sys.path.insert(0,'/content')
import numpy as np
import torch
from types import SimpleNamespace
from online_imp_structure import OnlineStructure

obj=OnlineStructure.__new__(OnlineStructure)
obj.args=SimpleNamespace(shared_classes=2)
obj.assignments=np.asarray([0,2,3])
obj.reliable=np.asarray([True,False])
obj.known_scope='entropy'
obj.veto_eligibility='raw'
original=torch.tensor([1.,1.,0.5])
for scale in (1.,0.5,0.):
    obj.entropy_candidate_scale=scale
    for name in ('known_original','known_effective','entropy_effective','alignment_effective'):
        setattr(obj,name+'_by_sample',np.zeros(3))
    entropy,alignment=obj.objective_weights(np.arange(3),original)
    assert torch.equal(entropy,torch.tensor([1.,scale,scale*0.5]))
    assert torch.equal(alignment,original)
    print('ENTROPY_DOSE_INTERFACE_OK',scale,entropy.tolist(),alignment.tolist(),flush=True)
