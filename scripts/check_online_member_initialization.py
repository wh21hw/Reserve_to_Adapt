"""Targeted current-coordinate initialization and permutation-free stability check."""
import numpy as np
import torch
from online_imp_structure import current_member_centers,membership_stability
from source_anchored_imp import SourceAnchoredIMP

known=2
old=np.array([0,0,1,1,2,2,3,3])
permuted=np.array([0,0,1,1,3,3,2,2])
assert membership_stability(permuted,old,known)['matched_assignment_change_fraction']==0
switched=permuted.copy();switched[0]=3
assert membership_stability(switched,old,known)['unknown_status_change_fraction']==1/8
features=torch.tensor([[0.,0.],[0.,.01],[1.,0.],[1.,.01],[4.,0.],[4.,.01],[6.,0.],[6.,.01]])
centers=current_member_centers(features+2,old,known)
assert torch.allclose(centers,torch.tensor([[6.,2.005],[8.,2.005]]))
anchors=torch.tensor([[0.,0.],[1.,0.]])
clusterer=SourceAnchoredIMP(.1,.001,steps=2,max_prototypes=20)
fit=clusterer.fit(features,anchors,birth_strategy='farthest',initial_candidates=current_member_centers(features,old,known))
assert fit['candidate_count']==2
extra=torch.cat([features,torch.tensor([[9.,0.]])])
fit=clusterer.fit(extra,anchors,birth_strategy='farthest',initial_candidates=current_member_centers(features,old,known))
assert fit['candidate_count']>=3
fit=clusterer.fit(features,anchors,birth_strategy='farthest',initial_candidates=torch.tensor([[4.,.005],[6.,.005],[100.,0.]]))
assert fit['candidate_count']==2
print('CURRENT_COORDINATE_MEMBER_INIT_PERMUTATION_METRIC_AND_BIRTH_OK')
