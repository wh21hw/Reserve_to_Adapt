"""One focused check: train() resets must not thaw encoder buffers or freeze head."""
import torch
from encoder_bn_policy import freeze_encoder_bn_on_forward

torch.manual_seed(7)
encoder=torch.nn.Sequential(torch.nn.Linear(4,4),torch.nn.BatchNorm1d(4))
head=torch.nn.Sequential(torch.nn.BatchNorm1d(4),torch.nn.Linear(4,2))
net=torch.nn.Sequential(encoder,head)
hook,count=freeze_encoder_bn_on_forward(encoder)
mean=encoder[1].running_mean.clone();variance=encoder[1].running_var.clone()
counter=encoder[1].num_batches_tracked.clone()
for _ in range(2):
    net.train()
    net(torch.randn(8,4)).square().sum().backward()
assert count==1 and not encoder[1].training and head[0].training
assert torch.equal(mean,encoder[1].running_mean) and torch.equal(variance,encoder[1].running_var)
assert torch.equal(counter,encoder[1].num_batches_tracked)
assert int(head[0].num_batches_tracked)==2
assert encoder[1].weight.requires_grad and torch.isfinite(encoder[1].weight.grad).all()
assert encoder[1].weight.grad.abs().sum()>0
print('ENCODER_BN_POLICY_CHECK_OK',dict(encoder_stats_fixed=True,affine_grad=True,head_updates=True),flush=True)
