"""Optional RTA encoder-BN-statistics ablation; does not freeze affine weights."""
import torch


def freeze_encoder_bn_on_forward(encoder):
    layers=[module for module in encoder.modules()
            if isinstance(module,(torch.nn.BatchNorm1d,torch.nn.BatchNorm2d,torch.nn.BatchNorm3d))]
    if not layers:raise ValueError('Encoder has no BatchNorm modules')
    def apply_policy(module,inputs):
        # TrainingMode may call train() each batch. Reapply immediately before
        # encoder forward, without changing head mode or requires_grad flags.
        for layer in layers:layer.eval()
    apply_policy(encoder,None)
    return encoder.register_forward_pre_hook(apply_policy),len(layers)
