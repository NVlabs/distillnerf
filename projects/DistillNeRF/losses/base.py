import torch
import torch.nn as nn
import torch.nn.functional as F
from mmdet3d.models.builder import LOSSES


class Loss(nn.Module):
    def __init__(self, coef: float = 1.0):
        super(Loss, self).__init__()
        self.coef = coef

    def __call__(self, *args, name: str, **kwargs):
        raise NotImplementedError()


@LOSSES.register_module()
class RGBL1Loss(Loss):
    def __call__(self, estimate: torch.Tensor, gt: torch.Tensor, name: str = "l1_loss"):
        l1loss = F.l1_loss(estimate, gt)
        return {name: l1loss * self.coef, name+'_obs': l1loss.detach()}


@LOSSES.register_module()
class MSELoss(Loss):
    def __call__(
        self, estimate: torch.Tensor, gt: torch.Tensor, name: str = "mse_loss"
    ):
        return {name: F.mse_loss(estimate, gt) * self.coef}


@LOSSES.register_module()
class BCEWithLogitsLoss(Loss):
    def __call__(
        self, estimate: torch.Tensor, gt: torch.Tensor, name: str = "bce_logits_loss"
    ):
        return {name: F.binary_cross_entropy_with_logits(estimate, gt) * self.coef}
