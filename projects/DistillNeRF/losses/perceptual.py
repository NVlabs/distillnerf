import torch
from lpips import LPIPS
from mmdet3d.models.builder import LOSSES

from .base import Loss


@LOSSES.register_module()
class LPIPSLoss(Loss):
    def __init__(
        self, coef: float = 1.0, net_type: str = "vgg", normalize: bool = True
    ):
        super().__init__(coef)
        self.lpips = LPIPS(net=net_type)

        # Doing this to ensure that the LPIPS parameters aren't optimized
        # (it seems that OpenMMLab pulls everything into the optimizer by default).
        self.lpips.requires_grad_(False)

        self.normalize = normalize

    def __call__(
        self, estimate: torch.Tensor, gt: torch.Tensor, name: str = "lpips_loss"
    ):
        B, N, C, H, W = estimate.shape

        return {
            name: self.lpips.forward(
                estimate.view(B * N, C, H, W),
                gt.view(B * N, C, H, W),
                normalize=self.normalize,
            )
            * self.coef
        }
