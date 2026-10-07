"""One targeted check: fixed running stats with trainable affine parameters."""
import torch

torch.manual_seed(3)
head=torch.nn.Sequential(torch.nn.BatchNorm1d(8),torch.nn.Linear(8,12,bias=False))
head.train()
bn=head[0]
bn.running_mean.copy_(torch.arange(8).float()/10)
bn.eval()
before=bn.running_mean.clone()
x=torch.randn(16,8)
head(x).square().mean().backward()
assert torch.equal(before,bn.running_mean)
assert bn.weight.requires_grad and bn.weight.grad is not None
assert bn.weight.grad.abs().sum()>0
with torch.no_grad():
    alone=head(x[:1])[0]
    mixed=head(x)[0]
assert torch.allclose(alone,mixed,atol=1e-6)
print('HEAD_BN_FIXED_STATS_AFFINE_GRADIENT_AND_BATCH_INVARIANCE_OK')
