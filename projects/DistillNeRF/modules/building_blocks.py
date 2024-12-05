from pdb import set_trace
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from einops import rearrange, repeat
from abc import abstractmethod
from functools import partial
import numpy as np
#from kornia.filters import filter2D

# from projects.DistillNeRF.utils.op import FusedLeakyReLU, fused_leaky_relu, upfirdn2d

class Normal:
    def __init__(self, mu, log_sigma, temp=1., min_sigma_value=0.01):
        self.mu = mu
        self.log_sigma = log_sigma
        self.sigma = torch.exp(log_sigma) + min_sigma_value
        if temp != 1.:
            self.sigma *= temp

    def sample(self):
        return sample_normal_jit(self.mu, self.sigma)

    def sample_given_eps(self, eps):
        return eps * self.sigma + self.mu
    def max_pdf_val(self):
        return 1 / torch.sqrt(2 * np.pi * self.sigma * self.sigma)

    def log_p(self, samples):
        normalized_samples = (samples - self.mu) / self.sigma
        log_p = - 0.5 * normalized_samples * normalized_samples - 0.5 * np.log(2 * np.pi) - torch.log(self.sigma)
        return log_p

    def kl(self, normal_dist):
        term1 = (self.mu - normal_dist.mu) / normal_dist.sigma
        term2 = self.sigma / normal_dist.sigma

        return 0.5 * (term1 * term1 + term2 * term2) - 0.5 - torch.log(term2)



def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module

class SinusoidalPositionEncoding(torch.nn.Module):
    '''
    https://chadrick-kwag.net/pytorch-implementation-of-sinusoidal-position-encoding/?utm_source=rss&utm_medium=rss&utm_campaign=pytorch-implementation-of-sinusoidal-position-encoding
    '''
    def __init__(self, dim, max_period=5000):
        assert dim % 2 == 0
        self.dim = dim
        self.max_period = max_period
        super().__init__()
        w_arr = torch.arange(0, self.dim // 2)
        w_arr = 1 / (max_period) ** (w_arr * 2 / dim)
        self.register_buffer("w_arr", w_arr)
    def forward(self, x):
        """
        assume x has shape (B,T) where B=batch size, T=token size(or sequence length)
        and values of x are integers >=0.
        """
        _x = torch.unsqueeze(x, -1)  # (B,T,1)
        v = _x * self.w_arr  # (B,T,dim//2)
        sin = torch.sin(v)
        sin = torch.unsqueeze(sin, -1)  # (B,T,m,1)
        cos = torch.cos(v)
        cos = torch.unsqueeze(cos, -1)  # (B,T,m,1)
        m = torch.cat([sin, cos], -1)  # (B,T,m,2)
        b, t, _, _ = m.shape
        y = m.reshape(b, t, -1)  # (B,T,dim) where 2m=`dim`
        return y


def filter_random(t, B, N, num_random):
    """
    t: tensor with shape (B*N, ...)
    B: batch size
    N: number of views
    num_random: number of random views (last num_random entries of N)
    """
    if num_random <= 0:
        return t

    input_shape = t.shape
    other_dims = input_shape[1:]
    t = t.reshape([B, N] + [d for d in other_dims])
    t = t[:, :N-num_random]
    t = t.reshape([B*(N-num_random)] + [d for d in other_dims])
    return t

class NoOpModule(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, xb):
        assert False, "Dummy module called, whoops"

class SimpleConvNet(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dims, kernel_sizes, paddings, activation=nn.ReLU):
        super().__init__()

        if type(kernel_sizes) == int:
            kernel_sizes = [kernel_sizes for _ in range(len(hidden_dims)+2)]

        if type(paddings) == int:
            paddings = [paddings for _ in range(len(hidden_dims)+2)]

        layers = [nn.Conv2d(input_dim, hidden_dims[0] if len(hidden_dims) > 0 else output_dim, kernel_sizes[0], padding=paddings[0])]

        if len(hidden_dims) > 0:
            last_hidden_dim = hidden_dims[0]

            for i,dim in enumerate(hidden_dims[1:]):
                layers += [activation()]

                layers += [nn.Conv2d(last_hidden_dim, dim, kernel_sizes[i+1], padding=paddings[i+1])]

                last_hidden_dim = dim

            layers += [activation()]

            layers += [nn.Conv2d(last_hidden_dim, output_dim, kernel_sizes[-1], padding=paddings[-1])]


        self.layers = nn.Sequential(*layers)

    def forward(self, x): return self.layers(x)

    @property
    def last_layer(self):
        return self.layers[-1]

class SimpleMLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dims, activation=nn.ReLU):
        super().__init__()

        layers = [nn.Linear(input_dim, hidden_dims[0] if len(hidden_dims) > 0 else output_dim)]

        if len(hidden_dims) > 0:
            last_hidden_dim = hidden_dims[0]

            for dim in hidden_dims[1:]:
                layers += [activation()]

                layers += [nn.Linear(last_hidden_dim, dim)]

                last_hidden_dim = dim

            layers += [activation()]

            layers += [nn.Linear(last_hidden_dim, output_dim)]


        self.layers = nn.Sequential(*layers)

    def forward(self, x): return self.layers(x)

class QuickCumMean(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, geom_feats, ranks):
        x = x.cumsum(0)
        kept = torch.ones(x.shape[0], device=x.device, dtype=torch.bool)
        kept[:-1] = (ranks[1:] != ranks[:-1])
        x, geom_feats = x[kept], geom_feats[kept]
        inds = kept.nonzero(as_tuple=True)[0]
        num_el = torch.cat([inds[:1]+1 ,(inds[1:]-inds[:-1])])

        x = torch.cat((x[:1], x[1:] - x[:-1]))


        # save kept for backward
        ctx.save_for_backward(kept)

        # no gradient for geom_feats
        ctx.mark_non_differentiable(geom_feats)
        ctx.mark_non_differentiable(num_el)

        return x, geom_feats, num_el

    @staticmethod
    def backward(ctx, gradx, gradgeom, gradnumel):
        kept, = ctx.saved_tensors
        back = torch.cumsum(kept, 0)
        back[kept] -= 1

        val = gradx[back]

        return val, None, None

class QuickCumMeanPBR(torch.autograd.Function):
    @staticmethod
    def forward(ctx, rgb, x, point_dist, coords, world_coords, ranks):
        x = x.cumsum(0)
        rgb = rgb.cumsum(0)
        point_dist = point_dist.cumsum(0)
        kept = torch.ones(x.shape[0], device=x.device, dtype=torch.bool)
        kept[:-1] = (ranks[1:] != ranks[:-1])
        rgb, x, coords, point_dist = rgb[kept], x[kept], coords[kept], point_dist[kept]
        world_coords = world_coords[kept]
        inds = kept.nonzero(as_tuple=True)[0]
        num_el = torch.cat([inds[:1]+1 ,(inds[1:]-inds[:-1])])

        x = torch.cat((x[:1], x[1:] - x[:-1]))
        rgb = torch.cat((rgb[:1], rgb[1:] - rgb[:-1]))
        point_dist = torch.cat((point_dist[:1], point_dist[1:] - point_dist[:-1]))
        # save kept for backward
        ctx.save_for_backward(kept)

        # no gradient for geom_feats
        ctx.mark_non_differentiable(coords)
        ctx.mark_non_differentiable(world_coords)
        ctx.mark_non_differentiable(num_el)
        ctx.mark_non_differentiable(rgb)
        # ctx.mark_non_differentiable(point_dist)

        return rgb, x, point_dist, coords, world_coords, num_el

    @staticmethod
    def backward(ctx, gradrgb, gradx, gradpd, gradgeom, gradworldcoords, gradnumel):
        kept, = ctx.saved_tensors
        back = torch.cumsum(kept, 0)
        back[kept] -= 1
        x_val = gradx[back]
        pd_val = gradpd[back]
        return None, x_val, pd_val, None, None, None



class QuickCumsum(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, geom_feats, ranks):
        x = x.cumsum(0)
        kept = torch.ones(x.shape[0], device=x.device, dtype=torch.bool)
        kept[:-1] = (ranks[1:] != ranks[:-1])
        x, geom_feats = x[kept], geom_feats[kept]

        x = torch.cat((x[:1], x[1:] - x[:-1]))

        # save kept for backward
        ctx.save_for_backward(kept)

        # no gradient for geom_feats
        ctx.mark_non_differentiable(geom_feats)

        return x, geom_feats

    @staticmethod
    def backward(ctx, gradx, gradgeom):
        kept, = ctx.saved_tensors
        back = torch.cumsum(kept, 0)
        back[kept] -= 1

        val = gradx[back]

        return val, None, None


class EncResBlock(nn.Module):
    def __init__(self, in_channel, channel):
        super().__init__()

        self.conv = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(in_channel, channel, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel, in_channel, 1),
        )

    def forward(self, input):
        out = self.conv(input)
        out += input

        return out

class DecResBlock(nn.Module):
    def __init__(self, in_channel, channel):
        super().__init__()

        self.conv = nn.Sequential(
            nn.ReLU(),
            nn.Conv3d(in_channel, channel, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(channel, in_channel, 1),
        )

    def forward(self, input):
        out = self.conv(input)
        out += input

        return out

class Conv2dStaticSamePadding(nn.Conv2d):
    """2D Convolutions like TensorFlow's 'SAME' mode, with the given input image size.
       The padding mudule is calculated in construction function, then used in forward.
    """

    # With the same calculation as Conv2dDynamicSamePadding

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, image_size=None, **kwargs):
        super().__init__(in_channels, out_channels, kernel_size, stride, **kwargs)
        self.stride = self.stride if len(self.stride) == 2 else [self.stride[0]] * 2

        # Calculate padding based on image size and save it
        assert image_size is not None
        ih, iw = (image_size, image_size) if isinstance(image_size, int) else image_size
        kh, kw = self.weight.size()[-2:]
        sh, sw = self.stride
        oh, ow = math.ceil(ih / sh), math.ceil(iw / sw)
        pad_h = max((oh - 1) * self.stride[0] + (kh - 1) * self.dilation[0] + 1 - ih, 0)
        pad_w = max((ow - 1) * self.stride[1] + (kw - 1) * self.dilation[1] + 1 - iw, 0)
        if pad_h > 0 or pad_w > 0:
            self.static_padding = nn.ZeroPad2d((pad_w // 2, pad_w - pad_w // 2,
                                                pad_h // 2, pad_h - pad_h // 2))
        else:
            self.static_padding = nn.Identity()

    def forward(self, x):
        x = self.static_padding(x)
        x = F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        return x

class Up(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor=2):
        super().__init__()
        self.scale_factor = scale_factor
        if self.scale_factor == 1:
            self.conv = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            )
        else:
            self.up = nn.Upsample(scale_factor=scale_factor, mode='bilinear',
                                  align_corners=True)

            self.conv = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )

    def forward(self, x1, x2):
        if self.scale_factor == 1:
            return self.conv(x1)
        else:
            x1 = self.up(x1)
            if x1.shape[-2] != x2.shape[-2] or x1.shape[-1] != x2.shape[-1]:
                x1 = F.interpolate(x1, (x2.shape[-2], x2.shape[-1]), mode='bilinear')
            x1 = torch.cat([x2, x1], dim=1)
            return self.conv(x1)

class PixelNorm(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return input * torch.rsqrt(torch.mean(input ** 2, dim=1, keepdim=True) + 1e-6)

class EqualConv2d(nn.Module):
    def __init__(
        self, in_channel, out_channel, kernel_size, stride=1, padding=0, bias=True, dilation=1
    ):
        super().__init__()

        self.weight = nn.Parameter(
            torch.randn(out_channel, in_channel, kernel_size, kernel_size)
        )
        self.scale = 1 / math.sqrt(in_channel * kernel_size ** 2)

        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channel))

        else:
            self.bias = None

    def forward(self, input):
        out = F.conv2d(
            input,
            self.weight * self.scale,
            bias=self.bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation
        )

        return out

    def __repr__(self):
        return (
            f'{self.__class__.__name__}({self.weight.shape[1]}, {self.weight.shape[0]},'
            f' {self.weight.shape[2]}, stride={self.stride}, padding={self.padding}, dilation={self.dilation})'
        )

class EqualLinear(nn.Module):
    def __init__(
        self, in_dim, out_dim, bias=True, bias_init=0, lr_mul=1, activation=None, zero_weight=False, force_bias=None
    ):
        super().__init__()
        if zero_weight:
            self.weight = nn.Parameter(torch.zeros(out_dim, in_dim))
        else:
            self.weight = nn.Parameter(torch.randn(out_dim, in_dim).div_(lr_mul))

        if force_bias is not None:
            self.bias = nn.Parameter(torch.FloatTensor(force_bias))
        else:
            if bias:
                self.bias = nn.Parameter(torch.zeros(out_dim).fill_(bias_init))

            else:
                self.bias = None

        self.activation = activation

        self.scale = (1 / math.sqrt(in_dim)) * lr_mul
        self.lr_mul = lr_mul

    def forward(self, input):
        if self.activation:
            out = F.linear(input, self.weight * self.scale)
            out = fused_leaky_relu(out, self.bias * self.lr_mul)

            # rest_dim = [1] * (out.ndim - self.bias.ndim - 1)
            # out = F.leaky_relu(
            #     out + self.bias.view(1, self.bias.shape[0], *rest_dim) * self.lr_mul, negative_slope=0.2
            # ) * (2 ** 0.5)
        else:
            out = F.linear(
                input, self.weight * self.scale, bias=self.bias * self.lr_mul
            )

        return out

    def __repr__(self):
        return (
            f'{self.__class__.__name__}({self.weight.shape[1]}, {self.weight.shape[0]})'
        )


class ConstantInput(nn.Module):
    def __init__(self, channel, size=4, w=-1):
        super().__init__()
        if w != -1:
            self.input = nn.Parameter(torch.randn(1, channel, size, w))
        else:
            self.input = nn.Parameter(torch.randn(1, channel, size, size))

    def forward(self, input):
        batch = input.shape[0]
        out = self.input.repeat(batch, 1, 1, 1)

        return out

class EqualConvTranspose2d(nn.Module):
    def __init__(
        self, in_channel, out_channel, kernel_size, stride=2, padding=0, bias=True
    ):
        super().__init__()

        self.weight = nn.Parameter(
            torch.randn(in_channel, out_channel, kernel_size, kernel_size)
        )
        self.scale = 1 / math.sqrt(in_channel * kernel_size ** 2)

        self.stride = stride
        self.padding = padding

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channel))

        else:
            self.bias = None

    def forward(self, input):
        out = F.conv_transpose2d(
            input,
            self.weight * self.scale,
            bias=self.bias,
            stride=self.stride,
            padding=self.padding,
        )

        return out

class SpatialModulatedConv2d(nn.Module):
    def __init__(
        self,
        in_channel,
        out_channel,
        kernel_size,
        style_dim,
        spatial_dim,
        demodulate=True,
        upsample=False,
        downsample=False,
        blur_kernel=[1, 3, 3, 1],
    ):
        super().__init__()

        self.eps = 1e-6
        self.kernel_size = kernel_size
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.upsample = upsample
        self.downsample = downsample
        self.spatial_dim = spatial_dim

        if upsample:
            factor = 2
            p = (len(blur_kernel) - factor) - (kernel_size - 1)
            pad0 = (p + 1) // 2 + factor - 1
            pad1 = p // 2 + 1

            self.blur = Blur(blur_kernel, pad=(pad0, pad1), upsample_factor=factor)

        if downsample:
            factor = 2
            p = (len(blur_kernel) - factor) + (kernel_size - 1)
            pad0 = (p + 1) // 2
            pad1 = p // 2

            self.blur = Blur(blur_kernel, pad=(pad0, pad1))

        fan_in = in_channel * kernel_size ** 2
        self.scale = 1 / math.sqrt(fan_in)
        self.padding = kernel_size // 2

        self.weight = nn.Parameter(
            torch.randn(1, out_channel, in_channel, kernel_size, kernel_size)
        )

        self.spatial_modulation_shared = nn.Sequential(EqualLinear(style_dim, in_channel * spatial_dim * spatial_dim // (2 * 2), bias_init=1,  activation='fused_lrelu'),
                                                       View((-1, in_channel, spatial_dim // 2, spatial_dim // 2)))
        self.spatial_upsample = nn.Upsample(scale_factor=2, mode='nearest')

        self.spatial_modulation_w = EqualConv2d(in_channel, out_channel, 3, stride=1, padding=1)
        self.spatial_modulation_b = EqualConv2d(in_channel, out_channel, 3, stride=1, padding=1)

        self.norm = nn.InstanceNorm2d(out_channel, affine=False)
        self.demodulate = demodulate


    def __repr__(self):
        return (
            f'{self.__class__.__name__}({self.in_channel}, {self.out_channel}, {self.kernel_size}, '
            f'upsample={self.upsample}, downsample={self.downsample})'
        )

    def forward(self, input, style):
        batch, in_channel, height, width = input.shape


        weight = self.scale * self.weight
        weight = weight.repeat(batch, 1, 1, 1, 1)
        weight = weight.view(
            batch * self.out_channel, in_channel, self.kernel_size, self.kernel_size
        )

        if self.upsample:
            input = input.view(1, batch * in_channel, height, width)
            weight = weight.view(
                batch, self.out_channel, in_channel, self.kernel_size, self.kernel_size
            )
            weight = weight.transpose(1, 2).reshape(
                batch * in_channel, self.out_channel, self.kernel_size, self.kernel_size
            )
            out = F.conv_transpose2d(input, weight, padding=0, stride=2, groups=batch)
            _, _, height, width = out.shape
            out = out.view(batch, self.out_channel, height, width)
            out = self.blur(out)

        elif self.downsample:
            input = self.blur(input)
            _, _, height, width = input.shape
            input = input.view(1, batch * in_channel, height, width)
            out = F.conv2d(input, weight, padding=0, stride=2, groups=batch)
            _, _, height, width = out.shape
            out = out.view(batch, self.out_channel, height, width)

        else:
            input = input.view(1, batch * in_channel, height, width)
            out = F.conv2d(input, weight, padding=self.padding, groups=batch)
            _, _, height, width = out.shape
            out = out.view(batch, self.out_channel, height, width)

        out = self.norm(out)
        _, _, height, width = out.shape

        style_map = self.spatial_modulation_shared(style)
        style_map = self.spatial_upsample(style_map)
        style_w = self.spatial_modulation_w(style_map)
        style_b = self.spatial_modulation_b(style_map)

        style_w = F.interpolate(style_w, size=(height, width), mode='nearest')
        style_b = F.interpolate(style_b, size=(height, width), mode='nearest')

        out = out * (1 + 0.3*style_w) + 0.3*style_b

        return out

class ModulatedConv2d(nn.Module):
    def __init__(
        self,
        in_channel,
        out_channel,
        kernel_size,
        style_dim,
        demodulate=True,
        upsample=False,
        downsample=False,
        blur_kernel=[1, 3, 3, 1],
    ):
        super().__init__()

        self.eps = 1e-6
        self.kernel_size = kernel_size
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.upsample = upsample
        self.downsample = downsample

        self.blur = None
        if upsample:
            factor = 2

            p = (4 - factor) - (kernel_size - 1) #p=2
            # p = (len(blur_kernel) - factor) - (kernel_size - 1)
            pad0 = (p + 1) // 2 + factor - 1 # pad0 is 2
            pad1 = p // 2 + 1 # pad1 is 2
            if blur_kernel is not None:
                self.blur = Blur(blur_kernel, pad=(pad0, pad1), upsample_factor=factor)

        if downsample:
            factor = 2
            p = (4 - factor) + (kernel_size - 1)
            # p = (len(blur_kernel) - factor) + (kernel_size - 1)
            pad0 = (p + 1) // 2
            pad1 = p // 2
            if blur_kernel is not None:
                self.blur = Blur(blur_kernel, pad=(pad0, pad1))

        fan_in = in_channel * kernel_size ** 2
        self.scale = 1 / math.sqrt(fan_in)
        self.padding = kernel_size // 2

        self.weight = nn.Parameter(
            torch.randn(1, out_channel, in_channel, kernel_size, kernel_size)
        )

        self.modulation = EqualLinear(style_dim, in_channel, bias_init=1)

        self.demodulate = demodulate

    def __repr__(self):
        return (
            f'{self.__class__.__name__}({self.in_channel}, {self.out_channel}, {self.kernel_size}, '
            f'upsample={self.upsample}, downsample={self.downsample})'
        )

    def forward(self, input, style):
        batch, in_channel, height, width = input.shape

        style = self.modulation(style).view(batch, 1, in_channel, 1, 1)
        weight = self.scale * self.weight * style

        if self.demodulate:
            demod = torch.rsqrt(weight.pow(2).sum([2, 3, 4]) + 1e-6)
            weight = weight * demod.view(batch, self.out_channel, 1, 1, 1)

        weight = weight.view(
            batch * self.out_channel, in_channel, self.kernel_size, self.kernel_size
        )
        if self.upsample:
            input = input.reshape(1, batch * in_channel, height, width)
            weight = weight.view(
                batch, self.out_channel, in_channel, self.kernel_size, self.kernel_size
            )
            weight = weight.transpose(1, 2).reshape(
                batch * in_channel, self.out_channel, self.kernel_size, self.kernel_size
            )
            out = F.conv_transpose2d(input, weight, padding=0, stride=2, groups=batch)
            _, _, height, width = out.shape
            out = out.view(batch, self.out_channel, height, width)
            if self.blur is not None:
                out = self.blur(out)

        elif self.downsample:
            if self.blur is not None:
                input = self.blur(input)
            _, _, height, width = input.shape
            input = input.view(1, batch * in_channel, height, width)
            out = F.conv2d(input, weight, padding=0, stride=2, groups=batch)
            _, _, height, width = out.shape
            out = out.view(batch, self.out_channel, height, width)

        else:
            input = input.view(1, batch * in_channel, height, width)
            out = F.conv2d(input, weight, padding=self.padding, groups=batch)
            _, _, height, width = out.shape
            out = out.view(batch, self.out_channel, height, width)

        return out

class ToRGB(nn.Module):
    def __init__(self, in_channel, style_dim, upsample=True, blur_kernel=[1, 3, 3, 1], spatial=False, spatial_dim=4):
        super().__init__()
        self.upsample = None
        if upsample:
            self.upsample = Upsample(blur_kernel)

        if spatial:
            self.conv = SpatialModulatedConv2d(in_channel, 3, 1, style_dim, spatial_dim, demodulate=False)
        else:
            self.conv = ModulatedConv2d(in_channel, 3, 1, style_dim, demodulate=False)
        self.bias = nn.Parameter(torch.zeros(1, 3, 1, 1))

    def forward(self, input, style, skip=None):
        out = self.conv(input, style)
        out = out + self.bias

        if skip is not None:
            if self.upsample is not None:
                skip = self.upsample(skip)

            out = out + skip

        return out

def make_kernel(k):
    k = torch.tensor(k, dtype=torch.float32)

    if k.ndim == 1:
        k = k[None, :] * k[:, None]

    k /= k.sum()

    return k

def c_native(
    input, kernel, up_x, up_y, down_x, down_y, pad_x0, pad_x1, pad_y0, pad_y1
):
    _, minor, in_h, in_w = input.shape
    kernel_h, kernel_w = kernel.shape
    out = input.view(-1, in_h, 1, in_w, 1, minor)
    out = F.pad(out, [0, 0, 0, up_x - 1, 0, 0, 0, up_y - 1])
    out = out.view(-1, in_h * up_y, in_w * up_x, minor)

    out = F.pad(
        out, [0, 0, max(pad_x0, 0), max(pad_x1, 0), max(pad_y0, 0), max(pad_y1, 0)]
    )
    out = out[
        :,
        max(-pad_y0, 0) : out.shape[1] - max(-pad_y1, 0),
        max(-pad_x0, 0) : out.shape[2] - max(-pad_x1, 0),
        :,
    ]

    out = out.permute(0, 3, 1, 2)
    out = out.reshape(
        [-1, 1, in_h * up_y + pad_y0 + pad_y1, in_w * up_x + pad_x0 + pad_x1]
    )
    w = torch.flip(kernel, [0, 1]).view(1, 1, kernel_h, kernel_w)
    out = F.conv2d(out, w)
    out = out.reshape(
        -1,
        minor,
        in_h * up_y + pad_y0 + pad_y1 - kernel_h + 1,
        in_w * up_x + pad_x0 + pad_x1 - kernel_w + 1,
    )
    # out = out.permute(0, 2, 3, 1)
    return out[:, :, ::down_y, ::down_x]

class Upsample(nn.Module):
    def __init__(self, kernel, factor=2):
        super().__init__()

        self.factor = factor
        kernel = make_kernel(kernel) * (factor ** 2)
        self.register_buffer('kernel', kernel)

        p = kernel.shape[0] - factor

        pad0 = (p + 1) // 2 + factor - 1
        pad1 = p // 2

        self.pad = (pad0, pad1)

    def forward(self, input):
        if True: #input.is_cuda:
            out = upfirdn2d(input, self.kernel, up=self.factor, down=1, pad=self.pad)
        else:
            out = upfirdn2d_native(input, self.kernel, self.factor, self.factor, 1, 1, self.pad[0], self.pad[1],
                                       self.pad[0], self.pad[1])

        return out

class NoiseInjection(nn.Module):
    def __init__(self):
        super().__init__()

        self.weight = nn.Parameter(torch.zeros(1))

    def forward(self, image, noise=None):
        if noise is None:
            batch, _, height, width = image.shape
            noise = image.new_empty(batch, 1, height, width).normal_()
        return image + self.weight * noise

class Blur(nn.Module):
    def __init__(self, kernel, pad, upsample_factor=1):
        super().__init__()

        kernel = make_kernel(kernel)

        if upsample_factor > 1:
            kernel = kernel * (upsample_factor ** 2)

        self.register_buffer('kernel', kernel)

        self.pad = pad

    def forward(self, input):
        if True: #input.is_cuda:

            out = upfirdn2d(input, self.kernel, pad=self.pad)


        else:
            out = upfirdn2d_native(input, self.kernel, 1, 1, 1, 1, self.pad[0], self.pad[1],
                                       self.pad[0], self.pad[1])

        return out

def upfirdn2d_native(
    input, kernel, up_x, up_y, down_x, down_y, pad_x0, pad_x1, pad_y0, pad_y1
):
    # BTODO: Fix upfirdn2d native and figure out why I had to change input dim order
    _, minor, in_h, in_w = input.shape
    kernel_h, kernel_w = kernel.shape

    out = input.view(-1, in_h, 1, in_w, 1, minor)
    out = F.pad(out, [0, 0, 0, up_x - 1, 0, 0, 0, up_y - 1])
    out = out.view(-1, in_h * up_y, in_w * up_x, minor)

    out = F.pad(
        out, [0, 0, max(pad_x0, 0), max(pad_x1, 0), max(pad_y0, 0), max(pad_y1, 0)]
    )
    out = out[
        :,
        max(-pad_y0, 0) : out.shape[1] - max(-pad_y1, 0),
        max(-pad_x0, 0) : out.shape[2] - max(-pad_x1, 0),
        :,
    ]

    out = out.permute(0, 3, 1, 2)
    out = out.reshape(
        [-1, 1, in_h * up_y + pad_y0 + pad_y1, in_w * up_x + pad_x0 + pad_x1]
    )
    w = torch.flip(kernel, [0, 1]).view(1, 1, kernel_h, kernel_w)
    out = F.conv2d(out, w)
    out = out.reshape(
        -1,
        minor,
        in_h * up_y + pad_y0 + pad_y1 - kernel_h + 1,
        in_w * up_x + pad_x0 + pad_x1 - kernel_w + 1,
    )

    return out[:, :, ::down_y, ::down_x]

class ChannelSpatialModulatedConv2d(nn.Module):
    def __init__(
            self,
            in_channel,
            out_channel,
            kernel_size,
            style_dim,
            spatial_dim,
            demodulate=True,
            upsample=False,
            downsample=False,
            demod_spatial=False,
            blur_kernel=[1, 3, 3, 1],
    ):
        super().__init__()

        self.eps = 1e-6
        self.kernel_size = kernel_size
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.upsample = upsample
        self.downsample = downsample
        self.spatial_dim = spatial_dim
        self.demod_spatial = demod_spatial
        self.style_dim = style_dim
        if upsample:
            factor = 2
            p = (len(blur_kernel) - factor) - (kernel_size - 1)
            pad0 = (p + 1) // 2 + factor - 1
            pad1 = p // 2 + 1

            self.blur = Blur(blur_kernel, pad=(pad0, pad1), upsample_factor=factor)

        if downsample:
            factor = 2
            p = (len(blur_kernel) - factor) + (kernel_size - 1)
            pad0 = (p + 1) // 2
            pad1 = p // 2

            self.blur = Blur(blur_kernel, pad=(pad0, pad1))

        fan_in = in_channel * kernel_size ** 2
        self.scale = 1 / math.sqrt(fan_in)
        self.padding = kernel_size // 2

        self.weight = nn.Parameter(
            torch.randn(1, out_channel, in_channel, kernel_size, kernel_size)
        )

        self.modulation = EqualLinear(style_dim // 2, in_channel, bias_init=1)

        max_spatial_dim = 64
        self.spatial_upsample = None
        if spatial_dim > max_spatial_dim:
            self.spatial_upsample = nn.Upsample((spatial_dim, spatial_dim), mode='bilinear')
            self.spatial_dim = max_spatial_dim
        if self.demod_spatial == 3:
            self.spatial_modulation = EqualLinear(style_dim // 2, 2 * self.spatial_dim * self.spatial_dim, bias_init=1)
        else:
            self.spatial_modulation = EqualLinear(style_dim // 2, self.spatial_dim * self.spatial_dim, bias_init=1)
        if self.demod_spatial == 1 or self.demod_spatial == 3:
            self.spatial_ln = nn.LayerNorm([self.spatial_dim * self.spatial_dim])
        self.demodulate = demodulate

    def __repr__(self):
        return (
            f'{self.__class__.__name__}({self.in_channel}, {self.out_channel}, {self.kernel_size}, '
            f'upsample={self.upsample}, downsample={self.downsample})'
        )

    def forward(self, input, style_in):
        style_chan, style_spatial = style_in.split(self.style_dim // 2, dim=1)
        batch, in_channel, height, width = input.shape

        style = self.modulation(style_chan).view(batch, 1, in_channel, 1, 1)
        weight = self.scale * self.weight * style

        if self.demodulate:
            demod = torch.rsqrt(weight.pow(2).sum([2, 3, 4]) + 1e-6)
            weight = weight * demod.view(batch, self.out_channel, 1, 1, 1)

        weight = weight.view(
            batch * self.out_channel, in_channel, self.kernel_size, self.kernel_size
        )

        if self.upsample:
            input = input.view(1, batch * in_channel, height, width)
            weight = weight.view(
                batch, self.out_channel, in_channel, self.kernel_size, self.kernel_size
            )
            weight = weight.transpose(1, 2).reshape(
                batch * in_channel, self.out_channel, self.kernel_size, self.kernel_size
            )
            out = F.conv_transpose2d(input, weight, padding=0, stride=2, groups=batch)
            _, _, height, width = out.shape
            out = out.view(batch, self.out_channel, height, width)
            out = self.blur(out)

        elif self.downsample:
            input = self.blur(input)
            _, _, height, width = input.shape
            input = input.view(1, batch * in_channel, height, width)
            out = F.conv2d(input, weight, padding=0, stride=2, groups=batch)
            _, _, height, width = out.shape
            out = out.view(batch, self.out_channel, height, width)

        else:
            input = input.view(1, batch * in_channel, height, width)
            out = F.conv2d(input, weight, padding=self.padding, groups=batch)
            _, _, height, width = out.shape
            out = out.view(batch, self.out_channel, height, width)

        _, _, height, width = out.shape



        if self.demod_spatial == 3:
            # multiplicative + additive
            spatial_style = self.spatial_modulation(style_spatial).view(batch, 2, self.spatial_dim, self.spatial_dim)
            if self.spatial_upsample is not None:
                spatial_style = self.spatial_upsample(spatial_style)
            spatial_style_mul, spatial_style_add = spatial_style.split(1, dim=1)
            demod = torch.sqrt(height * width / spatial_style_mul.pow(2).sum([2, 3]) + 1e-6)
            spatial_style_mul = spatial_style_mul * demod.view(batch, -1, 1, 1)
            spatial_style_add = self.spatial_ln(spatial_style_add.view(batch, -1)).view(batch, 1, self.spatial_dim,
                                                                                self.spatial_dim)
            out = (out * spatial_style_mul + spatial_style_add) / math.sqrt(2) ## heuristic for normalizing std, approx. assuming gaussian

        else:
            spatial_style = self.spatial_modulation(style_spatial).view(batch, 1, self.spatial_dim, self.spatial_dim)
            if self.spatial_upsample is not None:
                spatial_style = self.spatial_upsample(spatial_style)

            if self.demod_spatial == 2:
                # multiplicative
                demod = torch.sqrt(height * width / spatial_style.pow(2).sum([2, 3]) + 1e-6)
                spatial_style = spatial_style * demod.view(batch, -1, 1, 1)
                out = out * spatial_style
            elif self.demod_spatial == 1:
                # additive
                spatial_style = self.spatial_ln(spatial_style.view(batch, -1)).view(batch, 1, self.spatial_dim, self.spatial_dim)
                out = (out + spatial_style) / math.sqrt(2) ## heuristic for normalizing std, approx. assuming gaussian

        # out.view(batch, -1, height * width).std(2)
        return out

class StyledConv(nn.Module):
    def __init__(
        self,
        in_channel,
        out_channel,
        kernel_size,
        style_dim,
        upsample=False,
        blur_kernel=[1, 3, 3, 1],
        demodulate=True,
        spatial=False,
        spatial_dim=4,
        demod_spatial=False,
        chan_spatial_modulation=0
    ):
        super().__init__()

        if chan_spatial_modulation:
            self.conv = ChannelSpatialModulatedConv2d(
                in_channel,
                out_channel,
                kernel_size,
                style_dim,
                spatial_dim,
                upsample=upsample,
                blur_kernel=blur_kernel,
                demodulate=demodulate,
                demod_spatial=demod_spatial
            )
        elif spatial:
            self.conv = SpatialModulatedConv2d(
                in_channel,
                out_channel,
                kernel_size,
                style_dim,
                spatial_dim,
                upsample=upsample,
                blur_kernel=blur_kernel,
                demodulate=demodulate,
            )
        else:
            self.conv = ModulatedConv2d(
                in_channel,
                out_channel,
                kernel_size,
                style_dim,
                upsample=upsample,
                blur_kernel=blur_kernel,
                demodulate=demodulate,
            )

        self.noise = NoiseInjection()
        # self.bias = nn.Parameter(torch.zeros(1, out_channel, 1, 1))
        # self.activate = ScaledLeakyReLU(0.2)
        self.activate = nn.LeakyReLU(0.2) #FusedLeakyReLU(out_channel)

    def forward(self, input, style, noise=None):
        out = self.conv(input, style)
        out = self.noise(out, noise=noise)
        out = self.activate(out)

        return out


class ConvLayer(nn.Sequential):
    def __init__(
        self,
        in_channel,
        out_channel,
        kernel_size,
        downsample=False,
        upsample=False,
        blur_kernel=[1, 3, 3, 1],
        bias=True,
        activate=True,
        dilation=1,
        encoder_coord_conv2d=False
    ):
        layers = []

        if upsample:
            factor = 2
            p = (len(blur_kernel) - factor) - (kernel_size - 1)
            pad0 = (p + 1) // 2 + factor - 1
            pad1 = p // 2 + 1

            blur = Blur(blur_kernel, pad=(pad0, pad1), upsample_factor=factor)

        elif downsample:
            factor = 2
            p = (len(blur_kernel) - factor) + (kernel_size - 1)
            pad0 = (p + 1) // 2
            pad1 = p // 2

            layers.append(Blur(blur_kernel, pad=(pad0, pad1)))

            stride = 2
            self.padding = 0
        else:
            stride = 1
            self.padding = kernel_size // 2
        if dilation > 1:
            self.padding=dilation
        layers.append(
                EqualConv2d(
                    in_channel,
                    out_channel,
                    kernel_size,
                    padding=self.padding,
                    stride=stride,
                    dilation=dilation,
                    bias=bias and not activate,
                )
            )
        if upsample:
            layers.append(nn.Upsample(scale_factor=2))
            layers.append(blur)

        if activate:
            if bias:
                layers.append(FusedLeakyReLU(out_channel))
                # layers.append( nn.LeakyReLU(0.2)) #TODO fix this
            else:
                layers.append(ScaledLeakyReLU(0.2))

        super().__init__(*layers)


class ResBlock(nn.Module):
    def __init__(self, in_channel, out_channel, blur_kernel=[1, 3, 3, 1], encoder_coord_conv2d=False, downsample=True, filter_size=3):
        super().__init__()

        self.conv1 = ConvLayer(in_channel, in_channel, filter_size, encoder_coord_conv2d=encoder_coord_conv2d)
        self.conv2 = ConvLayer(in_channel, out_channel, filter_size, downsample=downsample, encoder_coord_conv2d=encoder_coord_conv2d)

        self.skip = ConvLayer(
            in_channel, out_channel, 1, downsample=downsample, activate=False, bias=False
        )

    def forward(self, input):
        out = self.conv1(input)
        out = self.conv2(out)

        skip = self.skip(input)
        out = (out + skip) / math.sqrt(2)

        return out

class ScalingLayer(nn.Module):
    def __init__(self):
        super(ScalingLayer, self).__init__()
        self.register_buffer('shift', torch.Tensor([-.030,-.088,-.188])[None,:,None,None])
        self.register_buffer('scale', torch.Tensor([.458,.448,.450])[None,:,None,None])

    def forward(self, inp):
        return (inp - self.shift) / self.scale


class NetLinLayer(nn.Module):
    ''' A single linear layer which does a 1x1 conv '''
    def __init__(self, chn_in, chn_out=1, use_dropout=False):
        super(NetLinLayer, self).__init__()

        layers = [nn.Dropout(),] if(use_dropout) else []
        layers += [nn.Conv2d(chn_in, chn_out, 1, stride=1, padding=0, bias=False),]
        self.model = nn.Sequential(*layers)

def spatial_average(in_tens, keepdim=True):
    return in_tens.mean([2,3],keepdim=keepdim)

def upsample(in_tens, out_H=64): # assumes scale factor is same for H and W
    in_H = in_tens.shape[2]
    scale_factor = 1.*out_H/in_H

    return nn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=False)(in_tens)

#################
# styleGAN Building blocks
#################

class Flatten(nn.Module):
    def forward(self, x):
        return x.reshape(x.shape[0], -1)

class PermuteToFrom(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        out, *_, loss = self.fn(x)
        out = out.permute(0, 3, 1, 2)
        return out, loss

class DepthWiseConv2d(nn.Module):
    def __init__(self, dim_in, dim_out, kernel_size, padding = 0, stride = 1, bias = True):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(dim_in, dim_in, kernel_size = kernel_size, padding = padding, groups = dim_in, stride = stride, bias = bias),
            nn.Conv2d(dim_in, dim_out, kernel_size = 1, bias = bias)
        )
    def forward(self, x):
        return self.net(x)

class LinearAttention(nn.Module):
    def __init__(self, dim, dim_head = 64, heads = 8):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        inner_dim = dim_head * heads

        self.nonlin = nn.GELU()
        self.to_q = nn.Conv2d(dim, inner_dim, 1, bias = False)
        self.to_kv = DepthWiseConv2d(dim, inner_dim * 2, 3, padding = 1, bias = False)
        self.to_out = nn.Conv2d(inner_dim, dim, 1)

    def forward(self, fmap):
        h, x, y = self.heads, *fmap.shape[-2:]
        q, k, v = (self.to_q(fmap), *self.to_kv(fmap).chunk(2, dim = 1))
        q, k, v = map(lambda t: rearrange(t, 'b (h c) x y -> (b h) (x y) c', h = h), (q, k, v))

        q = q.softmax(dim = -1)
        k = k.softmax(dim = -2)

        q = q * self.scale

        context = einsum('b n d, b n e -> b d e', k, v)
        out = einsum('b n d, b d e -> b n e', q, context)
        out = rearrange(out, '(b h) (x y) d -> b (h d) x y', h = h, x = x, y = y)

        out = self.nonlin(out)
        return self.to_out(out)


# one layer of self-attention and feedforward, for images

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
    def forward(self, x):
        return self.fn(x) + x

ChanNorm = partial(nn.InstanceNorm2d, affine = True)

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = ChanNorm(dim)

    def forward(self, x):
        return self.fn(self.norm(x))

attn_and_ff = lambda chan: nn.Sequential(*[
    Residual(PreNorm(chan, LinearAttention(chan))),
    Residual(PreNorm(chan, nn.Sequential(nn.Conv2d(chan, chan * 2, 1), leaky_relu(), nn.Conv2d(chan * 2, chan, 1))))
])

def noise(n, latent_dim, device):
    return torch.randn(n, latent_dim).cuda(device)

def log2(x): return math.log(x, 2)

def leaky_relu(p=0.2):
    return nn.LeakyReLU(p, inplace=True)

def exists(val):
    return val is not None

class Conv2DMod(nn.Module):
    def __init__(self, in_chan, out_chan, kernel, demod=True, stride=1, dilation=1, eps = 1e-6, **kwargs):
        super().__init__()
        self.filters = out_chan
        self.demod = demod
        self.kernel = kernel
        self.stride = stride
        self.dilation = dilation
        self.weight = nn.Parameter(torch.randn((out_chan, in_chan, kernel, kernel)))
        self.eps = eps
        nn.init.kaiming_normal_(self.weight, a=0, mode='fan_in', nonlinearity='leaky_relu')

    def _get_same_padding(self, size, kernel, dilation, stride):
        return ((size - 1) * (stride - 1) + dilation * (kernel - 1)) // 2

    def forward(self, x, y):
        b, c, h, w = x.shape

        w1 = y[:, None, :, None, None]
        w2 = self.weight[None, :, :, :, :]
        weights = w2 * (w1 + 1)

        if self.demod:
            d = torch.rsqrt((weights ** 2).sum(dim=(2, 3, 4), keepdim=True) + self.eps)
            weights = weights * d

        x = x.reshape(1, -1, h, w)

        _, _, *ws = weights.shape
        weights = weights.reshape(b * self.filters, *ws)

        padding = self._get_same_padding(h, self.kernel, self.dilation, self.stride)
        x = F.conv2d(x, weights, padding=padding, groups=b)

        x = x.reshape(-1, self.filters, h, w)
        return x

class SGEqualLinear(nn.Module):
    def __init__(self, in_dim, out_dim, lr_mul = 1, bias = True):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_dim, in_dim))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_dim))

        self.lr_mul = lr_mul

    def forward(self, input):
        return F.linear(input, self.weight * self.lr_mul, bias=self.bias * self.lr_mul)

class RGBBlock(nn.Module):
    def __init__(self, latent_dim, input_channel, upsample, rgba = False):
        super().__init__()
        self.input_channel = input_channel
        self.to_style = nn.Linear(latent_dim, input_channel)

        out_filters = 3 if not rgba else 4
        self.conv = Conv2DMod(input_channel, out_filters, 1, demod=False)

        self.upsample = nn.Sequential(
            nn.Upsample(scale_factor = 2, mode='bilinear', align_corners=False),
            SGBlur()
        ) if upsample else None

    def forward(self, x, prev_rgb, istyle):
        b, c, h, w = x.shape
        style = self.to_style(istyle)
        x = self.conv(x, style)

        if exists(prev_rgb):
            x = x + prev_rgb

        if exists(self.upsample):
            x = self.upsample(x)

        return x

class BlurNew(nn.Module):
    def __init__(self):
        super().__init__()
        f = torch.Tensor([1, 2, 1])
        self.register_buffer('f', f)
    def forward(self, x):
        f = self.f
        f = f[None, None, :] * f [None, :, None]
        return filter2D(x, f, normalized=True)

class GeneratorBlock(nn.Module):
    def __init__(self, latent_dim, input_channels, filters, upsampling_method = 'bilinear',
                upsampling_blur = True,  upsample = True, upsample_rgb = True, rgba = False,
                do_hier_resid=False, filters_past=None, decoder_noise=True):
        super().__init__()
        if not upsampling_blur:
            self.upsample = nn.Upsample(scale_factor=2, mode=upsampling_method, align_corners=False) if upsample else None
        else:
            self.upsample = nn.Sequential(
                nn.Upsample(scale_factor = 2, mode=upsampling_method, align_corners=False),
                BlurNew()
            )  if upsample else None

        self.to_style1 = nn.Linear(latent_dim, input_channels)

        self.decoder_noise = decoder_noise

        if self.decoder_noise:
            self.to_noise1 = nn.Linear(1, filters)
        self.conv1 = Conv2DMod(input_channels, filters, 3)

        self.to_style2 = nn.Linear(latent_dim, filters)
        if self.decoder_noise:
            self.to_noise2 = nn.Linear(1, filters)
        self.conv2 = Conv2DMod(filters, filters, 3)

        self.activation = leaky_relu()
        self.to_rgb = RGBBlock(latent_dim, filters, upsample_rgb, rgba)

        if do_hier_resid:
            self.bottom_decoder = nn.Sequential(
                nn.Conv2d(input_channels*2, input_channels*2, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(input_channels*2, input_channels, 1),
            )
            self.bottom_exit =  nn.Conv2d(input_channels, filters, 3, padding=1)

    def forward(self, x, prev_rgb, istyle, inoise, bottom=None):
        #print("1",x.shape,bottom.shape if bottom is not None else "none", self.bottom_decoder)

        if exists(self.upsample):
            x = self.upsample(x)

            if bottom is not None:
                bottom = self.upsample(bottom)

        bottom_resid = None
        if bottom is not None:
            combined = torch.cat((bottom, x), 1)
            bottom_resid = self.bottom_decoder(combined)
            x = x + bottom_resid
            bottom_resid = self.bottom_exit(bottom_resid)

        if self.decoder_noise:
            inoise = inoise[:, :x.shape[2], :x.shape[3], :]
            noise1 = self.to_noise1(inoise).permute((0, 3, 1, 2)) # TODO check if changing this is OK
            noise2 = self.to_noise2(inoise).permute((0, 3, 1, 2))

        style1 = self.to_style1(istyle)
        x = self.conv1(x, style1)
        x = self.activation(x + noise1 if self.decoder_noise else x)



        style2 = self.to_style2(istyle)
        x = self.conv2(x, style2)
        x = self.activation(x + noise2 if self.decoder_noise else x)


        rgb = self.to_rgb(x, prev_rgb, istyle)
        return x, rgb, bottom_resid

class SGBlur(nn.Module):
    def __init__(self):
        super().__init__()
        f = torch.Tensor([1, 2, 1])
        self.register_buffer('f', f)
    def forward(self, x):
        f = self.f
        f = f[None, None, :] * f [None, :, None]
        return filter2D(x, f, normalized=True)

class DiscriminatorBlock(nn.Module):
    def __init__(self, input_channels, filters, downsample=True):
        super().__init__()
        self.conv_res = nn.Conv2d(input_channels, filters, 1, stride = (2 if downsample else 1))

        self.net = nn.Sequential(
            nn.Conv2d(input_channels, filters, 3, padding=1),
            leaky_relu(),
            nn.Conv2d(filters, filters, 3, padding=1),
            leaky_relu()
        )

        self.downsample = nn.Sequential(
            SGBlur(),
            nn.Conv2d(filters, filters, 3, padding = 1, stride = 2)
        ) if downsample else None

    def forward(self, x):
        res = self.conv_res(x)
        x = self.net(x)
        if exists(self.downsample):
            x = self.downsample(x)
        x = (x + res) * (1 / math.sqrt(2))
        return x

def styles_def_to_tensor(styles_def):
    return torch.cat([t[:, None, :].expand(-1, n, -1) for t, n in styles_def], dim=1)

def evaluate_in_chunks(max_batch_size, model, *args):
    split_args = list(zip(*list(map(lambda x: x.split(max_batch_size, dim=0), args))))
    chunked_outputs = [model(*i) for i in split_args]
    if len(chunked_outputs) == 1:
        return chunked_outputs[0]
    return torch.cat(chunked_outputs, dim=0)





class TimestepBlock(nn.Module):
    """
    Any module where forward() takes timestep embeddings as a second argument.
    """

    @abstractmethod
    def forward(self, x, emb):
        """
        Apply the module to `x` given `emb` timestep embeddings.
        """


class TimestepEmbedSequential(nn.Sequential, TimestepBlock):
    """
    A sequential module that passes timestep embeddings to the children that
    support it as an extra input.
    """

    def forward(self, x, emb):
        for layer in self:
            if isinstance(layer, TimestepBlock):
                x = layer(x, emb)
            else:
                x = layer(x)
        return x

class AdaptiveGroupNorm(nn.Module):
    """
    https://github.com/NVlabs/denoising-diffusion-gan/blob/6818ded6443e10ab69c7864745457ce391d4d883/score_sde/models/layerspp.py
    """
    def __init__(self, in_channels=128, num_groups=32, style_dim=32):
        super().__init__()

        self.norm = nn.GroupNorm(num_groups, in_channels, affine=False, eps=1e-6)
        self.style = nn.Linear(style_dim, in_channels * 2)

        self.style.bias.data[:in_channels] = 1
        self.style.bias.data[in_channels:] = 0

    def forward(self, input, style):
        style = self.style(style).unsqueeze(2).unsqueeze(3)
        gamma, beta = style.chunk(2, 1)
        out = self.norm(input)
        out = gamma * out + beta

        return out

class Bottleneck(TimestepBlock):
    expansion = 1

    def __init__(self, in_planes, planes, norm_fn, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = norm_fn(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = norm_fn(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = norm_fn(self.expansion*planes)

        self.shortcut = nn.Sequential()
        self.shortcut_norm = None
        if stride != 1 or in_planes != self.expansion*planes:
            if isinstance(norm_fn, AdaptiveGroupNorm):

                self.shortcut_conv = nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False)
                self.shortcut_norm = norm_fn(self.expansion*planes)

            else:
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                    norm_fn(self.expansion*planes)
                )

    def forward(self, x, emb=None):
        if emb is None:
            out = F.relu(self.bn1(self.conv1(x)))
            out = F.relu(self.bn2(self.conv2(out)))
            out = self.bn3(self.conv3(out))
            out += self.shortcut(x)
        else:
            out = F.relu(self.bn1(self.conv1(x), emb))
            out = F.relu(self.bn2(self.conv2(out), emb))
            out = self.bn3(self.conv3(out), emb)
            if self.shortcut_norm is not None:
                out += self.shortcut_norm(self.shortcut_conv(x), emb)

        out = F.relu(out)
        return out
