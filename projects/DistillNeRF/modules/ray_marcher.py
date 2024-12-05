import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class RayMarcher(nn.Module):
    def __init__(self):
        super().__init__()
        self.noise_std = 0.5

    def forward(self, colors, densities, depths, rendering_options):
        deltas = depths[:, :, 1:] - depths[:, :, :-1]
        delta_inf = 1e10 * torch.ones_like(deltas[:, :, :1])

        deltas = torch.cat([deltas, delta_inf], -2)

        noise = torch.randn(densities.shape, device=densities.device) * rendering_options.get('nerf_noise', 0) if rendering_options.get('nerf_noise', 0) > 0 else 0

        alphas = 1 - torch.exp(-deltas * (F.softplus(densities + rendering_options.get('volume_init', -1) + noise)))

        if densities.dtype == torch.float32:
            eps = 1e-10
        else:
            eps = 1e-6

        alphas_shifted = torch.cat([torch.ones_like(alphas[:, :, :1]), 1-alphas + eps], -2)
        weights = alphas * torch.cumprod(alphas_shifted, -2)[:, :, :-1]

        rgb_final = torch.sum(weights * colors, -2)
        depth_final = torch.sum(weights * depths, -2)/weights.sum(2)
        depth_final = torch.nan_to_num(depth_final, float('inf'))
        depth_final = torch.clamp(depth_final, torch.min(depths), torch.max(depths))

        if rendering_options.get('white_back', False):
            weights_sum = weights.sum(2)
            rgb_final = rgb_final + 1-weights_sum

        self.noise_std -= 0.5/5000 # reduce noise over 5000 steps

        rgb_final = rgb_final * 2 - 1 # Scale to (-1, 1)
        depth_final = depth_final * 2 - 1

        return rgb_final, depth_final, weights

class MipRayMarcher(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, colors, density_logits, depths, rendering_options, c_sky=None):
        colors_mid = (colors[:, :, :-1] + colors[:, :, 1:]) / 2
        density_logits_mid_ = (density_logits[:, :, :-1] + density_logits[:, :, 1:]) / 2
        depths_mid = (depths[:, :, :-1] + depths[:, :, 1:]) / 2
        deltas = depths[:, :, 1:] - depths[:, :, :-1]

        # activation bias of -1 makes things initialize better
        if rendering_options['density_activation'] == 'sigmoid':
            densities_mid = F.sigmoid(density_logits_mid_ - 1)
        elif rendering_options['density_activation'] == 'softplus':
            densities_mid = F.softplus(density_logits_mid_ - 1)
        elif rendering_options['density_activation'] == 'exp':
            densities_mid = torch.exp(density_logits_mid_ - 1)

        ''' typical rendering equations '''
        density_delta = densities_mid * deltas
        alpha = 1 - torch.exp(-density_delta)
        eps = 1e-10
        # put 1 front, since the transmittence should be 1 at the beginning
        alpha_shifted = torch.cat([torch.ones_like(alpha[:, :, :1]), 1 - alpha + eps], -2)
        transmittence = torch.cumprod(alpha_shifted, -2)[:, :, :-1]
        weights = (alpha + eps) * transmittence
        weight_total = weights.sum(2)

        if c_sky is None:                   # True in this project, where we do not use sky embeddings
            weights_all = weights
            # rgb
            composite_rgb = torch.sum(weights * colors_mid, -2)
            # depth is the expected depth along the ray, so higher value, greater depth
            composite_depth = torch.sum(weights * depths_mid, -2) / (eps + weight_total)

        else:
            # weight left for sky
            sky_weight = torch.clamp(1 - weight_total, 0, 1.0)
            # rgb
            composite_rgb = torch.sum(weights * colors_mid, -2) + sky_weight * c_sky
            # depth
            weights_all = torch.cat([weights, sky_weight.unsqueeze(-1)], dim=-2)
            depth_sky = torch.ones_like(depths_mid[:,:,0]) * rendering_options['sky_depth']
            depths_all = torch.cat([depths_mid, depth_sky.unsqueeze(-2)], dim=-2)
            composite_depth = torch.sum(weights_all * depths_all, -2) / (eps + weight_total)

        ''' post processing '''
        # clip the composite depth to min/max range of depths
        composite_depth = torch.nan_to_num(composite_depth, rendering_options['max_depth'])
        composite_depth = torch.clamp(composite_depth, rendering_options['min_depth'], rendering_options['max_depth'])
        # Scale rgb to (-1, 1)
        composite_rgb = composite_rgb * 2 - 1

        return composite_rgb, composite_depth, weights, weights_all, alpha, depths_mid
