"""Targeted known-weight interface; does not change unknown selections/labels."""
from types import SimpleNamespace
import numpy as np
import torch
from online_imp_structure import OnlineStructure

bridge=object.__new__(OnlineStructure)
bridge.args=SimpleNamespace(shared_classes=2)
bridge.assignments=np.array([0,2,1])
bridge.known_original_by_sample=np.zeros(3,dtype=np.float32)
bridge.known_effective_by_sample=np.zeros(3,dtype=np.float32)
weight=torch.tensor([1.,1.,0.])
bridge.known_veto=False
assert torch.equal(bridge.known_weights(np.arange(3),weight),weight)
bridge.known_veto=True
assert bridge.known_weights(np.arange(3),weight).tolist()==[1.,0.,0.]
assert torch.equal(weight,torch.tensor([1.,1.,0.]))
assert bridge.assignments.tolist()==[0,2,1]
print('ONLINE_KNOWN_VETO_INTERFACE_OK; original tensor/assignments unchanged')
