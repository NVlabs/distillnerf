import pdb
import numpy as np
import torch
import functools
from copy import deepcopy

import math
import numbers
from torch import nn
from torch.nn import functional as F


def matrix_inverse(x):
    if x.dtype == torch.float32:
        try:
            invm = torch.inverse(x.cpu()).to(x)
        except:
            num_dim = len(x.shape)
            invm = torch.eye(x.shape[-1])
            inv_shape = []
            for ind in range(num_dim-2):
                invm = invm.unsqueeze(0)
                inv_shape.append(x.shape[ind])
            invm = invm.repeat(inv_shape + [1,1]).to(x)
            print('MATRIX INV ERROR')

        return invm
    else:
        indtype = x.dtype
        x = x.to(torch.float32)
        x = torch.inverse(x.cpu()).to(x)
        return x.to(indtype)



def reshape_BN(BN, x):
    return x.reshape(BN, *x.shape[2:])


def mask_image_helper(x, mask):
    B, _, H, W = x.shape
    Bm, _, Hm, Wm = mask.shape
    assert B == Bm, 'batch dimension does not match'
    if H != Hm or W != Wm:
        mask = F.interpolate(mask, (H, W))
    return x * (1-mask)

class GaussianSmoothing(nn.Module):
    """
    Apply gaussian smoothing on a
    1d, 2d or 3d tensor. Filtering is performed seperately for each channel
    in the input using a depthwise convolution.
    Arguments:
        channels (int, sequence): Number of channels of the input tensors. Output will
            have this number of channels as well.
        kernel_size (int, sequence): Size of the gaussian kernel.
        sigma (float, sequence): Standard deviation of the gaussian kernel.
        dim (int, optional): The number of dimensions of the data.
            Default value is 2 (spatial).
    """
    def __init__(self, channels, kernel_size, sigma, dim=2):
        super(GaussianSmoothing, self).__init__()
        if isinstance(kernel_size, numbers.Number):
            kernel_size = [kernel_size] * dim
        if isinstance(sigma, numbers.Number):
            sigma = [sigma] * dim

        # The gaussian kernel is the product of the
        # gaussian function of each dimension.
        kernel = 1
        meshgrids = torch.meshgrid(
            [
                torch.arange(size, dtype=torch.float32)
                for size in kernel_size
            ]
        )
        for size, std, mgrid in zip(kernel_size, sigma, meshgrids):
            mean = (size - 1) / 2
            kernel *= 1 / (std * math.sqrt(2 * math.pi)) * \
                      torch.exp(-((mgrid - mean) / std) ** 2 / 2)

        # Make sure sum of values in gaussian kernel equals 1.
        kernel = kernel / torch.sum(kernel)

        # Reshape to depthwise convolutional weight
        kernel = kernel.view(1, 1, *kernel.size())
        kernel = kernel.repeat(channels, *[1] * (kernel.dim() - 1))

        self.register_buffer('weight', kernel)
        self.groups = channels

        if dim == 1:
            self.conv = F.conv1d
        elif dim == 2:
            self.conv = F.conv2d
        elif dim == 3:
            self.conv = F.conv3d
        else:
            raise RuntimeError(
                'Only 1, 2 and 3 dimensions are supported. Received {}.'.format(dim)
            )

    def forward(self, input):
        """
        Apply gaussian filter to input.
        Arguments:
            input (torch.Tensor): Input to apply gaussian filter on.
        Returns:
            filtered (torch.Tensor): Filtered output.
        """
        return self.conv(input, weight=self.weight, groups=self.groups)

def soft_clamp(x, cval=3.):
    return x.div(cval).tanh_().mul(cval)

def format_ckpt(model, cfg):
    ckpt =  torch.load(cfg.ckpt, map_location="cuda:0")

    if cfg.old_ckpt:
        # Model surgery
        old_params = {k:v for k,v in ckpt["model"].items() if "latent_layer" not in k}
        new_params = {k:v for k,v in model.state_dict().items()}

        save_old_params = deepcopy(old_params)

        # old_params.pop("dx",None)
        # old_params.pop("bx",None)
        # old_params.pop("nx",None)
        old_params.pop("lss_encoder.bins0",None)
        old_params.pop("lss_encoder.bins1",None)
        old_params.pop("lss_encoder.bins2",None)
        old_params.pop("lss_encoder.dx",None)
        old_params.pop("lss_encoder.bx",None)
        old_params.pop("lss_encoder.nx",None)
        old_params.pop("lss_encoder.frustum",None)
        old_params.pop("volume_renderer.dx",None)
        old_params.pop("volume_renderer.bx",None)
        old_params.pop("volume_renderer.nx",None)

        old_params.pop("volume_renderer.bins0",None)
        old_params.pop("volume_renderer.bins1",None)
        old_params.pop("volume_renderer.bins2",None)

        new_state_dict = deepcopy(new_params)

        new_params.pop("frustum", None)
        new_params.pop("dec_frust", None)
        new_params.pop("bins0", None)
        new_params.pop("bins1", None)
        new_params.pop("bins2", None)
        new_params.pop("lss_encoder.bins0", None)
        new_params.pop("lss_encoder.bins1", None)
        new_params.pop("lss_encoder.bins2", None)

        new_params.pop("projector.volume_renderer.dx", None)
        new_params.pop("projector.volume_renderer.nx", None)
        new_params.pop("projector.volume_renderer.bx", None)

        # pdb.set_trace()

        mistmatch = False
        for (k1,v1), (k2,v2) in zip(old_params.items(), new_params.items()):
            if v1.shape != v2.shape:
                print('{:>100}'.format(str(k1)),'{:>100}'.format(str(k2)),'{:>40}'.format(str(v1.shape)), '{:>40}'.format(str(v2.shape)))
                mistmatch = True
            else:
                new_state_dict[k2] = v1

        model.projector.volume_renderer.bins0 = save_old_params["volume_renderer.bins0"]
        model.projector.volume_renderer.bins1 = save_old_params["volume_renderer.bins1"]
        model.projector.volume_renderer.bins2 = save_old_params["volume_renderer.bins2"]

        model.projector.volume_renderer.dx = save_old_params["volume_renderer.dx"]
        model.projector.volume_renderer.nx = save_old_params["volume_renderer.nx"]
        model.projector.volume_renderer.bx = save_old_params["volume_renderer.bx"]

        model.bins0 = save_old_params["lss_encoder.bins0"]
        model.bins1 = save_old_params["lss_encoder.bins1"]
        model.bins2 = save_old_params["lss_encoder.bins2"]

        model.nx = save_old_params["lss_encoder.nx"]
        model.bx = save_old_params["lss_encoder.bx"]
        model.dx = save_old_params["lss_encoder.dx"]

        del save_old_params

        if mistmatch: assert False
    else:
        new_state_dict = ckpt

    return new_state_dict



class Container():
    name = "Container"
    def __init__(self, containers=None, cat_dim=0):
        self.tensor_types = [torch.tensor, torch.Tensor, np.array]

        if containers is not None:
            for property in dir(containers[0]):
                if type(containers[0].get(property)) not in self.tensor_types: continue
                self.set(property, torch.cat([c.get(property) for c in containers], dim=cat_dim))

    def has(self, property):
        return hasattr(self, property) and self.get(property) is not None

    def set(self, property, value):
        setattr(self, property, value)

    def get(self, property):
        return getattr(self, property)

    def append(self, property, value):
        assert self.has(property) and type(self.get(property)) == list
        self.set(property, self.get(property) + [value])

    def keys(self):
        l = []
        for property in dir(self):
            # print(property, type(self.get(property)), type(self.get(property)) in self.tensor_types, self.tensor_types)
            if type(self.get(property)) in self.tensor_types:
                l.append(property)
        return l

    def __repr__(self):
        rep = f"{self.name}:"
        for property in dir(self):
            # print(property, type(self.get(property)), type(self.get(property)) in self.tensor_types, self.tensor_types)
            if type(self.get(property)) in self.tensor_types:
                rep += f"\n- {property}: {self.get(property).shape}"
        return rep

    def to(self, device):
        for property in dir(self):
            # print(property, type(self.get(property)), type(self.get(property)) in self.tensor_types, self.tensor_types)
            if type(self.get(property)) in self.tensor_types:
                if hasattr(self.get(property), "no_cuda"):
                    continue
                self.set(property, self.get(property).to(device))

def get_rz_index_from_bin(x, y, z, c_bin, dr, dz, dz_multiplier, xy_ego_size=2.0):
    # compute r index
    dist = torch.sqrt(torch.pow(x, 2) + torch.pow(y, 2))
    ind = torch.searchsorted(c_bin, dist.contiguous())
    r_bin = torch.cat([ torch.FloatTensor([c_bin[0] - (c_bin[1]-c_bin[0])]).to(x.device),
                        c_bin,
                        torch.FloatTensor([c_bin[-1]-c_bin[-2]]).to(x.device) ])
    interval = (dist - r_bin[ind]) / (r_bin[ind+1] - r_bin[ind])

    r_index = (ind-1) + interval

    # compute z index
    ego_offset = torch.round(xy_ego_size / dr) # for dist less than ego_offset, all same z grid size
    r_max = len(c_bin)
    # adaptive voxel size in z axis
    z_grid_size = dz * ((dz_multiplier - 1) * torch.pow(torch.clamp(r_index-ego_offset, min=0) / (r_max-ego_offset), 2) + 1)
    z_ind = z / z_grid_size
    # import pdb; pdb.set_trace();
    return r_index, z_ind


def get_theta_index_from_bin(x, y, dtheta, ntheta):
    theta = torch.atan(x/(y+1e-6)) # theta in -pi/2 ~ pi/2
    theta = theta * 180 / np.pi + 90 # in degrees from 0 ~ 180 degrees

    # for y positive, leave the same,
    # for y negative, handle accordingly
    theta[y<0] = 180 + theta[y<0]

    dtheta = 360 / ntheta

    theta_ind = theta / dtheta - 0.5
    theta_ind = torch.clip(theta_ind, -0.4999, ntheta-0.5001)
    return theta_ind

def get_r_bin(nr, dr, dr_multiplier, xy_ego_size=2.0):
    bins = []

    grid_size = nr
    min_dist = dr

    min_dist_num = xy_ego_size / min_dist
    mult = dr_multiplier
    max_dist = min_dist * mult

    r_bin = []
    for l in range(1, grid_size):
        if l < min_dist_num:
            r_bin.append(min_dist)
        else:
            r_bin.append(min_dist + np.power(max(0, l-min_dist_num) / (nr-min_dist_num), 2) * (max_dist - min_dist))

    r_bin = np.concatenate([np.array([0]), np.cumsum(r_bin)])
    return r_bin

def get_bins(nx, bx, dx, dx_multiplier, dz_multiplier, flip_z_axis=False, xy_ego_size=2.0):
    bins = []
    for axis in range(3):
        grid_size = nx[axis]
        min_dist = dx[axis]


        mult = dx_multiplier if axis < 2 else dz_multiplier
        max_dist = min_dist * mult

        if flip_z_axis and axis == 2:
            center_grid = int(-bx[axis] / min_dist) + 1
        else:
            center_grid = int(-bx[axis] / min_dist) + 1
        upperbound = max(center_grid, grid_size-center_grid)

        cur_bin = []
        ego_offset = (xy_ego_size / min_dist).astype(np.int32)

        for l in range(grid_size):
            center_dist = np.abs(l - center_grid)
            cur_bin.append(min_dist + np.power(max(0, center_dist-ego_offset) / (upperbound-ego_offset), 2) * (max_dist - min_dist))

        # if axis < 2 and xy_ego_size > 0:
        #     cur_bin = np.concatenate([-np.cumsum(cur_bin[:center_grid][::-1])[::-1]-xy_ego_size, np.cumsum(cur_bin[center_grid:])+xy_ego_size])
        # else:
        cur_bin = np.concatenate([-np.cumsum(cur_bin[:center_grid][::-1])[::-1], np.array([0]), np.cumsum(cur_bin[center_grid:])[:-1]])

        bins.append(cur_bin)

    return bins


def batch_convert(t, bs=None):
    if len(t.shape) >= 5: # has num_view dimension
        N,B = t.shape[:2]
        t = t.reshape(N*B, *t.shape[2:])
    elif len(t.shape) == 4: # has no num_view dimension
        N,_,_,_ = t.shape
        t = t.reshape(bs, N//bs, *t.shape[1:])
    else:
        assert False

    return t

def get_depth_bins(dbound, m, xy_ego_size=2.0, pow=2.0):
    n_D = (dbound[1] - dbound[0]) / dbound[2]

    start_D = dbound[0]
    init_dx_D = dbound[2]
    D_dxs = np.array([start_D+np.power(i/n_D, pow)*(m* init_dx_D) for i in range(int(n_D))])
    ds = np.cumsum(D_dxs)
    return ds, D_dxs

def get_depth_bins2(dbound, m, xy_ego_size=2.0, pow=2.0):
    n_D = (dbound[1] - dbound[0]) / dbound[2]

    start_D = dbound[0]
    init_dx_D = dbound[2]
    D_dxs = np.array([init_dx_D+np.power(i/n_D, pow)*(m*init_dx_D - init_dx_D) for i in range(int(n_D))])
    ds = np.cumsum(D_dxs)
    return ds + start_D, D_dxs

# def get_depth_bins(dbound, mult, xy_ego_size=2.0):
#     n_D = (dbound[1] - dbound[0]) / dbound[2]
#     grid_size = int(n_D)
#     min_dist = dbound[2]
#     min_dist_num = xy_ego_size / min_dist
#     max_dist = min_dist * mult
#
#     d_bin = []
#     for l in range(grid_size):
#         if l < min_dist_num:
#             d_bin.append(min_dist)
#         else:
#             d_bin.append(min_dist + np.power(max(0, l-min_dist_num) / (n_D-min_dist_num), 2) * (max_dist - min_dist))
#
#     d_bin_cumsum = np.cumsum(d_bin)
#     return d_bin_cumsum, d_bin


def get_index_from_bin(nx, dx, dz_multiplier, max_dist, gf, c_bin, adaptive_bin=False, x_ind=None, y_ind=None, xy_ego_size=2.0, z_lower_bound=0):
    if adaptive_bin:
        # distance from origin
        # dist = torch.sqrt(torch.pow(x_ind - nx[0]/2, 2) + torch.pow(y_ind - nx[1]/2, 2))
        dist = torch.maximum(torch.abs(x_ind-nx[0]/2), torch.abs(y_ind-nx[1]/2))
        min_dist = dx[2]
        ego_offset = torch.round(xy_ego_size / min_dist)

        # [torch.logical_and(x_ind>0, y_ind>0)]

        # adaptive voxel size in z axis
        z_grid_size = dx[2] * ((dz_multiplier - 1) * torch.pow(torch.clamp(dist-ego_offset, min=0) / (max_dist-ego_offset), 2) + 1)

        if z_lower_bound < 0:
            # assume z bound is [z_lower_bound, -z_lower_bound]
            total_z_size = z_grid_size * nx[2]
            shifted_coord = (gf + total_z_size / 2)
            return shifted_coord / z_grid_size
        else:
            return gf / z_grid_size

    ind = torch.searchsorted(c_bin, gf.contiguous())
    c_bin = torch.cat([ torch.FloatTensor([c_bin[0] - (c_bin[1]-c_bin[0])]).to(gf.device),
                        c_bin,
                        torch.FloatTensor([c_bin[-1]-c_bin[-2]]).to(gf.device) ])

    interval = (gf - c_bin[ind]) / (c_bin[ind+1] - c_bin[ind])

    return (ind-1) + interval



def gen_dx_bx(xbound, ybound, zbound):
    dx = torch.Tensor([row[2] for row in [xbound, ybound, zbound]])
    bx = torch.Tensor([row[0] + row[2]/2.0 for row in [xbound, ybound, zbound]])
    nx = torch.LongTensor([(row[1] - row[0]) / row[2] for row in [xbound, ybound, zbound]])

    return dx, bx, nx

def rgetattr(obj, attr, *args):
    def _getattr(obj, attr):
        return getattr(obj, attr, *args)
    return functools.reduce(_getattr, [obj] + attr.split('.'))

def load_old_ckpt(training_loop, ckpt):
    old_params = {k:v for k,v in ckpt["model"].items()}
    new_params = {k:v for k,v in training_loop.model.state_dict().items()}

    save_old_params = deepcopy(old_params)

    # old_params.pop("dx",None)
    # old_params.pop("bx",None)
    # old_params.pop("nx",None)
    old_params.pop("lss_encoder.dx",None)
    old_params.pop("lss_encoder.bx",None)
    old_params.pop("lss_encoder.nx",None)
    old_params.pop("lss_encoder.frustum",None)
    old_params.pop("volume_renderer.dx",None)
    old_params.pop("volume_renderer.bx",None)
    old_params.pop("volume_renderer.nx",None)

    old_params.pop("volume_renderer.bins0",None)
    old_params.pop("volume_renderer.bins1",None)
    old_params.pop("volume_renderer.bins2",None)

    new_state_dict = deepcopy(new_params)

    new_params.pop("frustum", None)
    new_params.pop("dec_frust", None)
    new_params.pop("lss_encoder.bins0", None)
    new_params.pop("lss_encoder.bins1", None)
    new_params.pop("lss_encoder.bins2", None)

    new_params.pop("projector.volume_renderer.dx", None)
    new_params.pop("projector.volume_renderer.nx", None)
    new_params.pop("projector.volume_renderer.bx", None)

    # pdb.set_trace()

    mistmatch = False
    for (k1,v1), (k2,v2) in zip(old_params.items(), new_params.items()):
        if v1.shape != v2.shape:
            print('{:>100}'.format(str(k1)),'{:>100}'.format(str(k2)),'{:>40}'.format(str(v1.shape)), '{:>40}'.format(str(v2.shape)))
            mistmatch = True
        else:
            new_state_dict[k2] = v1

    if mistmatch: assert False

    training_loop.model.load_state_dict(new_state_dict)

    if "lss_encoder.bins0" in save_old_params:

        training_loop.model.projector.volume_renderer.bins0 = save_old_params["volume_renderer.bins0"]
        training_loop.model.projector.volume_renderer.bins1 = save_old_params["volume_renderer.bins1"]
        training_loop.model.projector.volume_renderer.bins2 = save_old_params["volume_renderer.bins2"]

        training_loop.model.projector.volume_renderer.dx = save_old_params["volume_renderer.dx"]
        training_loop.model.projector.volume_renderer.nx = save_old_params["volume_renderer.nx"]
        training_loop.model.projector.volume_renderer.bx = save_old_params["volume_renderer.bx"]

        training_loop.model.bins0 = save_old_params["lss_encoder.bins0"]
        training_loop.model.bins1 = save_old_params["lss_encoder.bins1"]
        training_loop.model.bins2 = save_old_params["lss_encoder.bins2"]

        training_loop.model.nx = save_old_params["lss_encoder.nx"]
        training_loop.model.bx = save_old_params["lss_encoder.bx"]
        training_loop.model.dx = save_old_params["lss_encoder.dx"]

        del save_old_params
    return training_loop

def hook_fn(m, i, o):
    print(m)
    print("------------Input Grad------------")

    for grad in i:
        try:
            print(grad.shape)
            print(grad.mean())
        except AttributeError:
            print ("None found for Gradient")

    print("------------Output Grad------------")
    for grad in o:
        try:
            print(grad.shape)
            print(grad.mean())
        except AttributeError:
            print ("None found for Gradient")
    print("\n")


if __name__=='__main__':
    ds, D_dxs = get_depth_bins2([0.15, 7.15, 0.1], 20, pow=3.0)
    print(ds)
    print(D_dxs)
