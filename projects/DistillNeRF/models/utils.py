import numpy as np
import torch


def batched_backwards_polynomial(pixel_norms, intrinsic):
    ret = 0

    for k in range(intrinsic.shape[1]):
        ret += intrinsic[:, k : k + 1] * torch.pow(pixel_norms, k)
    return ret


def batched_pixel_2_camera_ray(pixel_coords, intrinsic, camera_model):
    """Convert the pixel coordinates to a 3D ray in the camera coordinate system.

    Args:
        pixel_coords (FloatTensor): pixel coordinates of the selected points [B,n,2]
        intrinsic (FloatTensor): camera intrinsic parameters (size depends on the camera model)
        camera_model (string): camera model used for projection. Must be one of ['pinhole', 'f_theta']

    Out:
        camera_rays (FloatTensor): rays in the camera coordinate system [B,n,3]
    """
    B, n, _ = pixel_coords.shape
    camera_rays = torch.ones((B, n, 3)).to(pixel_coords.device)

    if camera_model == "pinhole":
        pass
    elif camera_model == "f_theta":
        pixel_offsets = torch.ones((B, n, 2)).to(pixel_coords.device)
        pixel_offsets[:, :, 0] = pixel_coords[:, :, 0] - intrinsic[:, 0:1]
        pixel_offsets[:, :, 1] = pixel_coords[:, :, 1] - intrinsic[:, 1:2]

        pixel_norms = torch.norm(pixel_offsets, dim=2)
        alphas = batched_backwards_polynomial(pixel_norms, intrinsic[:, 4:9]).unsqueeze(
            -1
        )
        pixel_norms = pixel_norms.unsqueeze(-1)
        camera_rays[:, :, 0:1] = (
            torch.sin(alphas) * pixel_offsets[:, :, 0:1]
        ) / pixel_norms
        camera_rays[:, :, 1:2] = (
            torch.sin(alphas) * pixel_offsets[:, :, 1:2]
        ) / pixel_norms
        camera_rays[:, :, 2:3] = torch.cos(alphas)

        # special case: ray is perpendicular to image plane normal
        valid = (pixel_norms > np.finfo(np.float32).eps).squeeze(-1)
        camera_rays[~valid, :] = torch.FloatTensor([0, 0, 1]).to(
            pixel_coords.device
        )  # This is what DW sets these rays to

    return camera_rays


def gen_dx_bx(xbound, ybound, zbound):
    dx = torch.Tensor([row[2] for row in [xbound, ybound, zbound]])
    bx = torch.Tensor([row[0] + row[2] / 2.0 for row in [xbound, ybound, zbound]])
    nx = torch.LongTensor(
        [(row[1] - row[0]) / row[2] for row in [xbound, ybound, zbound]]
    )

    return dx, bx, nx


def get_index_from_bin(
    nx,
    dx,
    dz_multiplier,
    max_dist,
    gf,
    c_bin,
    adaptive_bin=False,
    x_ind=None,
    y_ind=None,
    xy_ego_size=2.0,
    z_lower_bound=0,
):
    if adaptive_bin:
        # distance from origin
        # dist = torch.sqrt(torch.pow(x_ind - nx[0]/2, 2) + torch.pow(y_ind - nx[1]/2, 2))
        dist = torch.maximum(torch.abs(x_ind - nx[0] / 2), torch.abs(y_ind - nx[1] / 2))
        min_dist = dx[2]
        ego_offset = torch.round(xy_ego_size / min_dist)

        # [torch.logical_and(x_ind>0, y_ind>0)]

        # adaptive voxel size in z axis
        z_grid_size = dx[2] * (
            (dz_multiplier - 1)
            * torch.pow(
                torch.clamp(dist - ego_offset, min=0) / (max_dist - ego_offset), 2
            )
            + 1
        )

        if z_lower_bound < 0:
            # assume z bound is [z_lower_bound, -z_lower_bound]
            total_z_size = z_grid_size * nx[2]
            shifted_coord = gf + total_z_size / 2
            return shifted_coord / z_grid_size
        else:
            return gf / z_grid_size

    ind = torch.searchsorted(c_bin, gf.contiguous())
    c_bin = torch.cat(
        [
            torch.FloatTensor([c_bin[0] - (c_bin[1] - c_bin[0])]).to(gf.device),
            c_bin,
            torch.FloatTensor([c_bin[-1] - c_bin[-2]]).to(gf.device),
        ]
    )

    interval = (gf - c_bin[ind]) / (c_bin[ind + 1] - c_bin[ind])

    return (ind - 1) + interval


def get_depth_bins(dbound, m):
    n_D = (dbound[1] - dbound[0]) / dbound[2]

    start_D = dbound[0]
    init_dx_D = dbound[2]
    D_dxs = np.array(
        [start_D + np.power(i / n_D, 2) * (m * init_dx_D) for i in range(int(n_D))]
    )
    ds = np.cumsum(D_dxs)
    return ds, D_dxs


def get_bins(
    nx, bx, dx, dx_multiplier, dz_multiplier, flip_z_axis=False, xy_ego_size=2.0
):
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
        upperbound = max(center_grid, grid_size - center_grid)

        cur_bin = []
        ego_offset = (xy_ego_size / min_dist).astype(np.int32)

        for l in range(grid_size):
            center_dist = np.abs(l - center_grid)
            cur_bin.append(
                min_dist
                + np.power(
                    max(0, center_dist - ego_offset) / (upperbound - ego_offset), 2
                )
                * (max_dist - min_dist)
            )

        # if axis < 2 and xy_ego_size > 0:
        #     cur_bin = np.concatenate([-np.cumsum(cur_bin[:center_grid][::-1])[::-1]-xy_ego_size, np.cumsum(cur_bin[center_grid:])+xy_ego_size])
        # else:
        cur_bin = np.concatenate(
            [
                -np.cumsum(cur_bin[:center_grid][::-1])[::-1],
                np.array([0]),
                np.cumsum(cur_bin[center_grid:])[:-1],
            ]
        )

        bins.append(cur_bin)

    return bins


def ensure_0to1(
    inp: torch.Tensor,
) -> torch.Tensor:
    inp_min = torch.clamp(inp.min(), max=0.0)
    inp_max = torch.clamp(inp.max(), min=1.0)
    return (inp - inp_min) / (inp_max - inp_min)


def unnormalize(
    imgs: torch.Tensor,
    mean: np.ndarray,
    std: np.ndarray,
    to_rgb: bool,
) -> torch.Tensor:
    # No use for to_rgb, but we accept it anyways since
    # img_norm_cfg in mmdet3d typically has it too.
    unnorm_img = imgs * torch.from_numpy(std)[:, None, None].to(
        imgs.device
    ) + torch.from_numpy(mean)[:, None, None].to(imgs.device)

    return unnorm_img / 255.0
