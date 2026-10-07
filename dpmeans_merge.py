"""Conditional objective-decreasing candidate merges, not full Bayesian inference."""
import math
import torch


@torch.no_grad()
def merge_candidates(features,known_centers,candidates,assignments,penalty):
    if not math.isfinite(penalty) or penalty<=0:raise ValueError('Positive finite cluster penalty required')
    dtype=features.dtype
    x=features.double();known=known_centers.double();cand=candidates.double()
    ids=torch.as_tensor(assignments,device=x.device,dtype=torch.long).clone()
    c=len(known)
    if ids.shape!=(len(x),) or not torch.isfinite(x).all():raise ValueError('Invalid merge features/memberships')
    def objective(protos,labels):
        centers=torch.cat([known,protos])
        return float((x-centers[labels]).square().sum()+penalty*len(protos))
    initial=objective(cand,ids);history=[initial];events=[]
    while len(cand)>1:
        counts=torch.stack([(ids==c+j).sum() for j in range(len(cand))]).double()
        if (counts==0).any():raise ValueError('Input candidates must be occupied')
        means=torch.stack([x[ids==c+j].mean(0) for j in range(len(cand))])
        offsets=counts*(cand-means).square().sum(1)
        ward=(counts[:,None]*counts[None]/(counts[:,None]+counts[None]))*(means[:,None]-means[None]).square().sum(-1)
        gain=penalty+offsets[:,None]+offsets[None]-ward
        gain=gain.masked_fill(~torch.ones_like(gain,dtype=torch.bool).triu(1),-float('inf'))
        best,index=gain.reshape(-1).max(0)
        if float(best)<=1e-10:break
        a,b=divmod(int(index),len(cand))
        merged=(counts[a]*means[a]+counts[b]*means[b])/(counts[a]+counts[b])
        proposal=cand.clone();proposal[a]=merged
        proposal=torch.cat([proposal[:b],proposal[b+1:]])
        proposal_ids=(x[:,None]-torch.cat([known,proposal])[None]).square().sum(-1).argmin(1)
        # Empty candidates pay no ongoing complexity cost. Preserve known IDs.
        occupied=torch.unique(proposal_ids[proposal_ids>=c],sorted=True)
        dense=proposal_ids.clone()
        for j,old in enumerate(occupied):dense[proposal_ids==old]=c+j
        proposal=proposal[occupied-c]
        after=objective(proposal,dense)
        before=history[-1]
        if not after<before-1e-10:
            raise RuntimeError('Merge did not decrease recorded objective')
        events.append(dict(pair=[a,b],before_K=len(cand),after_K=len(proposal),
            predicted_fixed_partition_gain=float(best),actual_gain=before-after))
        cand,ids=proposal,dense;history.append(after)
    report=dict(enabled=True,penalty=penalty,initial_K=len(candidates),final_K=len(cand),
        merge_events=events,objective_history=history,objective='target hard SSE + lambda candidate K',
        known_centers_unchanged=True,source_prior_constant_during_merge=True,
        global_optimum_claimed=False,target_semantics_used=False)
    return cand.to(dtype),ids.cpu().numpy(),report
