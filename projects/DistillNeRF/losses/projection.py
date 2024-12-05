from typing import List
from torch import Tensor
import torch
import torch.nn.functional as F
from mmdet3d.models.builder import LOSSES
from .base import Loss


@LOSSES.register_module()
class NerfWeightEntropyLoss(Loss):
    def __call__(
        self,
        nerf_weights: torch.Tensor,
        name: str = "target_nerf_weight_entropy_loss",
    ):
        weight_entropy = -((nerf_weights + 1e-10) * torch.log(nerf_weights + 1e-10)).sum(1)
        return {
            name: weight_entropy.mean() * self.coef,
            name+'_obs': weight_entropy.mean().detach()
        }


@LOSSES.register_module()
class OpacityLoss(Loss):

    def nan_checking(self, opacity):
        if (opacity > 1).any(): print("(opacity > 1).any()")
        if torch.isnan(opacity).any(): 
            print("torch.isnan(opacity).any()")
            print(opacity)
        if torch.isinf(opacity).any(): 
            print("torch.isinf(opacity).any()")
            print(opacity)

    def __call__(
        self,
        weights: torch.Tensor,
        name: str = "_opacity_loss",
    ):
        opacity = torch.sum(weights, 1)
        opacity = torch.clamp(opacity, 0, 1)
        # self.nan_checking(opacity)
        
        with torch.cuda.amp.autocast(enabled=False):
            opacity_loss = F.binary_cross_entropy(opacity, torch.ones_like(opacity).to(opacity), reduction="none").mean()

        return {
            name: opacity_loss * self.coef,
            name + '_obs': opacity_loss.detach()
        }


@LOSSES.register_module()
class DepthLoss(Loss):
    def __init__(self, coef, depth_min, depth_max):
        super().__init__(coef)
        self.depth_min = depth_min
        self.depth_max = depth_max

    def __call__(
        self, estimate: torch.Tensor, gt: torch.Tensor, name: str = "depth_loss"
    ):
        nan_mask = gt.isfinite()

        depth_target = gt[nan_mask] / self.depth_max
        depth_predicted = estimate[nan_mask] / self.depth_max

        depth_loss = F.mse_loss(depth_predicted, depth_target) + 0.1 * F.l1_loss(
            depth_predicted, depth_target
        )
        return {name: depth_loss * self.coef}


@LOSSES.register_module()
class DepthClampLoss(Loss):
    '''
        compared to DepthLoss, this loss clamp the depth range
    '''
    def __init__(self, coef, depth_min, depth_max):
        super().__init__(coef)
        self.depth_min = depth_min
        self.depth_max = depth_max
    
    def nan_checking(self, gt, nan_mask):
        # Check if there are any NaN values in gt
        # if torch.isnan(gt).any():
        #     print("Warning: NaN values found in gt tensor")
        # # Check if there are any Inf values in gt
        # if torch.isinf(gt).any():
        #     print("Warning: Inf values found in gt tensor")
        # Check if there are any NaN values in gt
        if torch.isnan(nan_mask).any():
            print("Warning: NaN values found in nan_mask tensor")
            nan_mask = torch.nan_to_num(nan_mask, False)
        # Check if there are any Inf values in nan_mask
        if torch.isinf(nan_mask).any():
            nan_mask[torch.where(nan_mask>10)] = False
            print("nan_mask.dtype: ", nan_mask.dtype)
            print("nan_mask.shape:, ", nan_mask.shape)
            print("gt.shape:, ", gt.shape)
            print("nan_mask: ", nan_mask)
            print("Warning: Inf values found in nan_mask tensor")
        # Check if there are any Inf values in gt
        if torch.isinf(gt[nan_mask]).any():
            print("Warning: Inf values found in gt[nan_mask] tensor")
        if torch.isnan(gt[nan_mask]).any():
            print("Warning: NaN values found in gt[nan_mask] tensor")

    def __call__(
        self, estimate: torch.Tensor, gt: torch.Tensor, name: str = "lidar_depth_loss"
    ):
        nan_mask = gt.isfinite()
        # self.nan_checking(gt, nan_mask)
        gt = gt[nan_mask].clamp(self.depth_min, self.depth_max)
        estimate = estimate[nan_mask].clamp(self.depth_min, self.depth_max)

        depth_target = gt / self.depth_max
        depth_predicted = estimate / self.depth_max

        depth_loss = F.mse_loss(depth_predicted, depth_target) + 0.1 * F.l1_loss(depth_predicted, depth_target)

        return {name: depth_loss * self.coef, name+'_obs': depth_loss.detach()}

@LOSSES.register_module()
class EmernerfDepthClampLoss(Loss):
    '''
        compared to DepthClampLoss, this loss additionally normalize the inner/outer depth differently
    '''
    def __init__(self, coef, depth_min, depth_max, inner_range):
        super().__init__(coef)
        self.depth_min = depth_min
        self.depth_max = depth_max
        self.inner_range = inner_range

    def __call__(
        self, estimate: torch.Tensor, gt: torch.Tensor, name: str = "emernerf_depth_loss"
    ):
        nan_mask = gt.isfinite()
        gt = gt[nan_mask].clamp(self.depth_min, self.depth_max)
        estimate = estimate[nan_mask].clamp(self.depth_min, self.depth_max)
        
        # normalize the inner/outer depth differently
        inner_mask = gt < self.inner_range
        outter_mask = gt >= self.inner_range
        inner_depth_gt = gt[inner_mask] / self.inner_range
        inner_depth_estimate = estimate[inner_mask] / self.inner_range
        outter_depth_gt = gt[outter_mask] / self.depth_max
        outter_depth_estimate = estimate[outter_mask] / self.depth_max

        depth_loss = F.mse_loss(inner_depth_estimate, inner_depth_gt) + 0.1 * F.l1_loss(inner_depth_estimate, inner_depth_gt) \
                    + F.mse_loss(outter_depth_estimate, outter_depth_gt) + 0.1 * F.l1_loss(outter_depth_estimate, outter_depth_gt) \

        return {name: depth_loss * self.coef, name+'_obs': depth_loss.detach()}


@LOSSES.register_module()
class MidasDepthClampLoss(Loss):
    '''
        same as DepthClampLoss, just different name
    '''
    def __init__(self, coef, depth_min, depth_max):
        super().__init__(coef)
        self.depth_min = depth_min
        self.depth_max = depth_max

    def __call__(
        self, estimate: torch.Tensor, gt: torch.Tensor, name: str = "midas_depth_loss"
    ):
        nan_mask = gt.isfinite()
        gt = gt[nan_mask].clamp(self.depth_min, self.depth_max)
        estimate = estimate[nan_mask].clamp(self.depth_min, self.depth_max)

        depth_target = gt / self.depth_max
        depth_predicted = estimate / self.depth_max

        depth_loss = F.mse_loss(depth_predicted, depth_target) + 0.1 * F.l1_loss(depth_predicted, depth_target)

        return {name: depth_loss * self.coef, name+'_obs': depth_loss.detach()}


@LOSSES.register_module()
class DepthAnythingDepthClampLoss(Loss):
    '''
        same as DepthClampLoss, just different name
    '''
    def __init__(self, coef, depth_min, depth_max):
        super().__init__(coef)
        self.depth_min = depth_min
        self.depth_max = depth_max

    def __call__(
        self, estimate: torch.Tensor, gt: torch.Tensor, name: str = "depthanything_depth_loss"
    ):
        nan_mask = gt.isfinite()
        gt = gt[nan_mask].clamp(self.depth_min, self.depth_max)
        estimate = estimate[nan_mask].clamp(self.depth_min, self.depth_max)

        depth_target = gt / self.depth_max
        depth_predicted = estimate / self.depth_max

        depth_loss = F.mse_loss(depth_predicted, depth_target) + 0.1 * F.l1_loss(depth_predicted, depth_target)

        return {name: depth_loss * self.coef, name+'_obs': depth_loss.detach()}


def dirac_delta_approx(x, mu=0, sigma=1e-5):
    """
    Approximates the Dirac delta function with a Gaussian distribution.

    Args:
        x (torch.Tensor): The input tensor.
        mu (float, optional): The mean of the Gaussian distribution. Defaults to 0.
        sigma (float, optional): The standard deviation of the Gaussian distribution. Defaults to 1e-5.

    Returns:
        torch.Tensor: The output tensor.
    """
    return (1 / (torch.sqrt(2 * torch.pi * sigma**2))) * torch.exp(
        -((x - mu) ** 2) / (2 * sigma**2)
    )


def compute_line_of_sight_loss(
    gt_depth: Tensor,
    weights: Tensor,
    t_vals: Tensor,
    epsilon: float = 2.0,
):
    """
    Computes the line-of-sight loss between the predicted and ground truth depth.

    Args:
        gt_depth (Tensor): Ground truth termination point.
        weights (Tensor): weights of each sampled interval.
        t_vals (Tensor): midpoint of each sampled interval.
        epsilon (float, optional): Margin for the line-of-sight loss. Defaults to 2.0.

    Returns:
        Tensor: Line-of-sight loss between the predicted and ground truth depth.
    """
    D = weights.shape[1]
    gt_depth = gt_depth.unsqueeze(2)                                    # [b, n, 1, h, w]
    weights = weights.unsqueeze(0)                                      # [b, n, D, h, w]
    t_vals = t_vals.unsqueeze(0)                                        # [b, n, D, h, w]

    empty_mask = t_vals < (gt_depth - epsilon)                                          # [b, n, D, h, w]
    near_mask = (t_vals > (gt_depth - epsilon)) & (t_vals < gt_depth + epsilon)         # [b, n, D, h, w]
    far_mask = t_vals > (gt_depth + epsilon)                                            # [b, n, D, h, w]
    depth_mask = gt_depth > 0                                                           # [b, n, 1, h, w]

    sight_loss = 0
    if (empty_mask * depth_mask).any():
        empty_loss = (weights.square()[empty_mask * depth_mask]).mean()
        sight_loss += empty_loss
    if (near_mask * depth_mask).any():
        near_loss = (weights - dirac_delta_approx(t_vals - gt_depth, sigma=torch.tensor(epsilon) / 3)).square()[near_mask * depth_mask].mean()
        sight_loss += near_loss
    if (far_mask * depth_mask).any():
        far_loss = (weights.square()[far_mask * depth_mask]).mean()
        sight_loss += far_loss

    sight_loss = empty_loss + near_loss + far_loss
    return sight_loss


@LOSSES.register_module()
class LineOfSightLoss(Loss):
    """
    Line of sight loss function.

    Args:
        loss_type (Literal["my",]): The type of loss to use.
        name (str): The name of the loss function.
        depth_error_percentile (float): The percentile of rays to optimize within each batch that have smallest depth error.
        coef (float): The coefficient to multiply the loss by.
        upper_bound (float): The upper bound of the loss.
        reduction (str): The reduction method to use.
        check_nan (bool): Whether to check for NaN values in the loss.

    Attributes:
        loss_type (Literal["my",]): The type of loss being used.
        name (str): The name of the loss function.
        upper_bound (float): The upper bound of the loss.
        depth_error_percentile (float): The percentile of rays to optimize within each batch that have smallest depth error.
    """

    def __init__(
        self,
        coef: float = 0.1,
    ):
        super(LineOfSightLoss, self).__init__(coef)

    def __call__(
        self,
        gt_depth: Tensor,
        weights: Tensor,
        t_vals: Tensor,
        epsilon: float,
        name: str = 'line_of_sight_loss',
    ):
        LoS_loss = compute_line_of_sight_loss(
            gt_depth, weights, t_vals.detach(), epsilon
        )
        if type(LoS_loss) == int:
            return {name: LoS_loss * self.coef, name+'_obs': LoS_loss}
        else:
            return {name: LoS_loss * self.coef, name+'_obs': LoS_loss.detach()}