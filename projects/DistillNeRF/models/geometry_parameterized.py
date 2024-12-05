import pdb
import torch.nn as nn
import torch
import torch.nn.functional as F
from projects.DistillNeRF.utils.camera_utils import batched_pixel_2_camera_ray, batched_project_camera_rays_2_img
from projects.DistillNeRF.utils.utils import get_bins, get_depth_bins, get_r_bin, reshape_BN, matrix_inverse
from einops import rearrange
from nerfacc import (
    accumulate_along_rays,
    render_transmittance_from_density,
    render_weight_from_density,
)

def get_TRANSFORM(inner_range, contract_ratio, sampling_method = "uniform_lindisp"):
    TRANSFORM = {
        "uniform": (
            lambda x: x, 
            lambda s: s
        ),
        "uniform_lindisp": (
            lambda x: torch.where(torch.abs(x) < inner_range, x / inner_range * contract_ratio, (1 - inner_range / torch.abs(x) * (1 - contract_ratio)) * x / torch.abs(x)),   # x to s
            lambda s: torch.where(torch.abs(s) < contract_ratio, s * inner_range / contract_ratio, inner_range * (1 - contract_ratio) / (1 - torch.abs(s)) * s / torch.abs(s)),            # s to x
        )
    }
    return TRANSFORM[sampling_method]

def xy_z_transform(config):
    xy_inner_range = config.xy_inner_range
    z_inner_range = config.z_inner_range
    contract_ratio = config.contract_ratio
    sample_method = config.sample_method
    xy_x_to_s, xy_s_to_x = get_TRANSFORM(xy_inner_range, contract_ratio, sample_method)
    z_x_to_s, z_s_to_x = get_TRANSFORM(z_inner_range, contract_ratio, sample_method)

    return xy_x_to_s, xy_s_to_x, z_x_to_s, z_s_to_x

def create_categorical_depth(depth_geom, sparse_sample_ratio=1):
    sample_method = depth_geom.sample_method                    # "uniform_lindisp" or "uniform"
    inner_range = depth_geom.inner_range                        # 50.0, only for "uniform_lindisp"
    contract_ratio = depth_geom.contract_ratio
    if sample_method == "uniform_lindisp":   
        x_to_s, s_to_x = get_TRANSFORM(inner_range, contract_ratio, "uniform_lindisp")

        # sample in inner x space
        near_plane_in_x = depth_geom.near_plane_in_x
        inner_step_in_x = depth_geom.inner_step_in_x * sparse_sample_ratio
        sample_inner_in_x = torch.arange(near_plane_in_x, inner_range, step=inner_step_in_x)

        # set sampling param in out space
        inner_step_in_s = x_to_s(torch.tensor(inner_step_in_x))
        outer_step_in_s = inner_step_in_s * depth_geom.outer_sample_in_s_aug
        far_plane_in_x = depth_geom.far_plane_in_x
        far_plane_in_s = x_to_s(torch.tensor(far_plane_in_x))
        
        # sample in outer s space, transfrom outer s to x
        if far_plane_in_s > contract_ratio:
            sample_outer_in_s = torch.arange(contract_ratio, far_plane_in_s, step=outer_step_in_s)
            sample_outer_in_x = s_to_x(sample_outer_in_s)
        else:       # when we set contract_ratio as 1, where we only have inner voxel
            sample_outer_in_s = torch.arange(contract_ratio, contract_ratio + outer_step_in_s / 2, step=outer_step_in_s)
            sample_outer_in_x = s_to_x(sample_outer_in_s)

        # concat inner sample and outer sample
        sample_in_x = torch.cat([sample_inner_in_x, sample_outer_in_x])
        sample_depth_all = sample_in_x

        # create rendering samples, which is the mid point of each sample interval
        sample_depth_start = sample_depth_all[:-1]
        sample_depth_end = sample_depth_all[1:]
        sample_depth_mid = (sample_depth_start + sample_depth_end) / 2
        sample_depth_delta = sample_depth_end - sample_depth_start
        D = sample_depth_mid.shape[0]

    elif sample_method == "uniform":          
        x_to_s, s_to_x = get_TRANSFORM(None, "uniform")
        # sample in x space
        depth_max = depth_geom.dbound[1]
        depth_min = depth_geom.dbound[0]
        depth_step = depth_geom.dbound[2]
        sample_depth_mid = torch.arange(depth_min, depth_max, depth_step)
        sample_depth_start = sample_depth_mid - depth_step / 2
        sample_depth_end = sample_depth_mid + depth_step / 2
        sample_depth_delta = depth_step

        D = sample_depth_mid.shape[0]
    else:
        raise NotImplementedError(f"sample_method {sample_method} not implemented")
    
    return x_to_s, s_to_x, sample_depth_all, sample_depth_start, sample_depth_end, sample_depth_mid, sample_depth_delta, D


def volume_render(t_starts, t_ends, densities):
    trans, alphas = render_transmittance_from_density(
        t_starts, t_ends, densities
    )

    weights = trans * alphas

    # =============== Geometry ================ #
    opacities = accumulate_along_rays(weights, values=None).clamp(1e-6, 1.0)
    # expected depth
    depths = accumulate_along_rays(weights, values=(t_starts + t_ends)[..., None] / 2.0)
    depths = depths / opacities

    return depths, weights


class Geometry_P(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config    # model.config
        self.get_geom_info(self.config.geom_param)
        self.init_params()

    def get_geom_info(self, config):
        """
        Returns dictionary with:
        - dx: World spacing between consecutive voxels dimension of [3] for x,y,z
        - bx: World coord for first voxel [3] for x,y,z
        - nx: Number of voxels for each dimension [3] for x,y,z
        - frustum: 3D world coord in camera frame for each location in encoded image features
        - ds: downsample factor of features from original image
        """

        grid_conf = {
            'xbound': config.xbound,
            'ybound': config.ybound,
            'zbound': config.zbound,
            'dbound': config.dbound,
        }

        dx, bx, nx = self.gen_dx_bx(
            grid_conf['xbound'],
            grid_conf['ybound'],
            grid_conf['zbound'],
        )

        self.register_buffer('dx', dx)
        self.register_buffer('bx', bx)
        self.register_buffer('nx', nx)
        frustum = self.create_input_frustum(grid_conf)
        render_frust = self.create_render_frustum()

        self.register_buffer('frustum', frustum)
        self.register_buffer('render_frust', render_frust)

        self.grid_conf = grid_conf

        self.bins = None

    def gen_dx_bx(self, xbound, ybound, zbound):
        dx = torch.Tensor([row[2] for row in [xbound, ybound, zbound]])
        bx = torch.Tensor([row[0] + row[2]/2.0 for row in [xbound, ybound, zbound]])
        nx = torch.LongTensor([(row[1]+1e-6 - row[0]) / row[2] for row in [xbound, ybound, zbound]])

        return dx, bx, nx

    def init_params(self):
        self.D = self.frustum.shape[0]
        pass





    # def points_to_img_coord(self, points, trans, rots, intrins, post_trans, post_rots, img_height, img_width, camera_model="f_theta", return_continuous=False, return_dir=False, return_init_valid=False):
    #     '''
    #     go from world points to image coordinates
    #     '''
    #     B, N = trans.shape[0], trans.shape[1]
    #     BN = B*N
    #     trans = reshape_BN(BN, trans)
    #     rots = reshape_BN(BN, rots)
    #     intrins = reshape_BN(BN, intrins)
    #     post_trans = reshape_BN(BN, post_trans)
    #     post_rots = reshape_BN(BN, post_rots)

    #     # convert world voxel coords into camera coordinates
    #     points = points - trans.reshape(BN, 1, 3)
    #     if return_dir:
    #         point_dir = F.normalize(points, dim=2)
    #     point_dist = torch.clamp(torch.linalg.norm(points, dim=2), 0, 100)

    #     points = (matrix_inverse(rots).unsqueeze(1)).matmul(points.unsqueeze(-1)).squeeze(-1)

    #     # Perform projections to image planes
    #     proj_points = []
    #     valids = []
    #     for ind in range(len(intrins)):
    #         proj_point, valid = batched_project_camera_rays_2_img(points[ind], intrins[ind:ind+1], camera_model)
    #         proj_points.append(proj_point)
    #         valids.append(valid)
    #     proj_points = torch.stack(proj_points, dim=0)
    #     valids = torch.stack(valids, dim=0).squeeze(-1)
    #     # now proj_points are in image coordinates (original image plane)
    #     # perform adjustments for the size of image encoder sees
    #     resize_factor = post_rots[:, 0, 0]
    #     proj_points = post_rots.unsqueeze(1).matmul(proj_points.unsqueeze(-1)).squeeze(-1) + post_trans.unsqueeze(1)

    #     # NOTE: index 0 is width, index 1 is height, index 2 is z
    #     # checking number of valid points
    #     x_img_coord = (proj_points[:,:,0]).long()
    #     y_img_coord = (proj_points[:,:,1]).long()


    #     x_ok = torch.logical_and(0 <= x_img_coord, x_img_coord < img_width)
    #     y_ok = torch.logical_and(0 <= y_img_coord, y_img_coord < img_height)
    #     z_ok = proj_points[:,:,2] > 0.0

    #     valid = torch.logical_and(valids, torch.logical_and(torch.logical_and(x_ok, y_ok), z_ok))
    #     out = []
    #     if return_continuous:
    #         out = [proj_points[:,:,0], proj_points[:,:,1], valid, point_dist]
    #     else:
    #         out = [x_img_coord, y_img_coord, valid]

    #     if return_dir:
    #         out.append(point_dir)
    #     if return_init_valid:
    #         out.append(valids)
    #     return out


    def create_render_frustum(self):
        """
        Make camera frustum mapping location of image features to (width, height, depth) in camera frame
        """
        fh, fw = self.config.render_img_height, self.config.render_img_width
        ds = torch.arange(self.config.render_img_depth[0], self.config.render_img_depth[1], self.config.render_img_depth[2], dtype=torch.float).view(-1, 1, 1).expand(-1, fh, fw)
        D, _, _ = ds.shape
        xs = torch.linspace(0, self.config.input_img_width - 1, fw, dtype=torch.float).view(1, 1, fw).expand(D, fh, fw)
        ys = torch.linspace(0, self.config.input_img_height - 1, fh, dtype=torch.float).view(1, fh, 1).expand(D, fh, fw)

        # D x H x W x 3
        frustum = torch.stack((xs, ys, ds), -1)
        return nn.Parameter(frustum, requires_grad=False) # depth map of 3D points


    # def create_arbitrary_frustum(self, downsample, in_H, in_W, grid_conf, fH=None, fW=None):
    #     """
    #     Make camera frustum mapping location of image features to (width, height, depth) in camera frame
    #     """
    #     fH, fW = in_H // downsample,  in_W // downsample

    #     depths = torch.arange(grid_conf['dbound'][0], grid_conf['dbound'][1]-1e-6, grid_conf['dbound'][2], dtype=torch.float)
    #     ds = depths.view(-1, 1, 1).expand(-1, fH, fW)

    #     D, _, _ = ds.shape

    #     xs = torch.linspace(0,  in_W - 1, fW, dtype=torch.float).view(1, 1, fW).expand(D, fH, fW)
    #     ys = torch.linspace(0,  in_H - 1, fH, dtype=torch.float).view(1, fH, 1).expand(D, fH, fW)

    #     # W x H x D x 3
    #     frustum = torch.stack((xs, ys, ds), -1)

    #     return frustum, ds

    def create_input_frustum(self, grid_conf):
        """
            Create camera frustum (width, height, depth) in camera frame
        """
        H_input_img, W_input_img = self.config.input_img_height, self.config.input_img_width
        downsample = self.config.downsample

        fH, fW = H_input_img // downsample,  W_input_img // downsample

        depths = torch.arange(grid_conf['dbound'][0], grid_conf['dbound'][1]-1e-6, grid_conf['dbound'][2], dtype=torch.float)
        ds = depths.view(-1, 1, 1).expand(-1, fH, fW)

        D, _, _ = ds.shape

        xs = torch.linspace(0,  W_input_img - 1, fW, dtype=torch.float).view(1, 1, fW).expand(D, fH, fW)
        ys = torch.linspace(0,  H_input_img - 1, fH, dtype=torch.float).view(1, fH, 1).expand(D, fH, fW)

        # W x H x D x 3
        frustum = torch.stack((xs, ys, ds), -1)

        return frustum

    def get_3D_coord_in_world_frame(self, scene_data, depths=None):
        """ 
            Determine the (x,y,z) locations of lifted 3D features, in the ego frame (LiDAR coordinates)
            Returns B x N x D x H/downsample x W/downsample x 3
        """

        ''' get the frustum in the image plane'''
        B, N, _ = scene_data.trans.shape
        points = self.frustum[:2]
        # points: [2, H, W, 3], last dimension: [h_index, w_index, depth], depth: [0.25, 0.65]

        ''' scale on image plane, since the input image resolution can be smaller than the original image resolution '''
        points = points - scene_data.post_trans.view(B, N, 1, 1, 1, 3)
        points = matrix_inverse(scene_data.post_rots).view(B, N, 1, 1, 1, 3, 3).matmul(points.unsqueeze(-1))
        # points: [B, N, D, H, W, 3, 1]

        ''' image plane to camera frame, using intrinsics '''
        # [index * depth, depth]
        points = torch.cat((points[:, :, :, :, :, :2] * points[:, :, :, :, :, 2:3],
                            points[:, :, :, :, :, 2:3]
                            ), 5)
        # import pdb; pdb.set_trace()
        intrin_inv = matrix_inverse(scene_data.intrins).view(B, N, 1, 1, 1, 3, 3)
        points = intrin_inv.matmul(points)

        ''' create ray directions ''' 
        init_rays = F.normalize(points[:,:,1] - points[:,:,0], dim=-2)
        init_rays = init_rays.squeeze(-1)
        rays = init_rays.unsqueeze(2)

        ''' ray direction * depths ''' 
        D = depths.shape[1]
        depths = rearrange(depths, '(b n) d h w -> b n d h w 1', b=B, n=N, d=D)
        points = (rays * depths).unsqueeze(-1)

        ''' transform to world frame, using extrinsics '''
        combine = scene_data.rots
        points = combine.view(B, N, 1, 1, 1, 3, 3).matmul(points).squeeze(-1)
        points += scene_data.trans.view(B, N, 1, 1, 1, 3)

        return points


    def generate_render_rays_points(self, scene_data, intermediates, ray_name="rays", out_name="target_points", frustum=None):
        """
            Returns points/rays of target camera in world coordinates
            Returns:
                target_points ->
        """

        ''' camera poses '''
        trans, rots, intrins, post_trans, post_rots = scene_data.target_trans, scene_data.target_rots, scene_data.target_intrins, scene_data.target_post_trans, scene_data.target_post_rots

        ''' get the frustum in the image plane'''
        B, N, _ = trans.shape
        frustum = self.render_frust

        ''' scale on image plane, since the render image resolution can be smaller than the original image resolution '''
        points = frustum -  post_trans.view(B, N, 1, 1, 1, 3)
        points = matrix_inverse(post_rots).view(B, N, 1, 1, 1, 3, 3).matmul(points.unsqueeze(-1))
        # points: [b, n, d, h, w, 3, 1] 

        ''' image plane to camera frame, using intrinsics '''
        depths = points[:, :, :, :, :, -1]
        points = points[:, :, :2]            # equal to points = self.frumstum[:2] in get_geometry
        points = torch.cat((points[:, :, :, :, :, :2] * points[:, :, :, :, :, 2:3],
                            points[:, :, :, :, :, 2:3]
                            ), 5)
        intrin_inv = matrix_inverse(intrins).view(B, N, 1, 1, 1, 3, 3)
        points = intrin_inv.matmul(points)

        ''' create ray directions ''' 
        init_rays = F.normalize(points[:,:,1] - points[:,:,0], dim=-2)
        init_rays = init_rays.squeeze(-1)
        rays = init_rays.unsqueeze(2)

        ''' ray direction * depths ''' 
        points = (rays * depths).unsqueeze(-1)
        
        ''' transform to world frame '''
        # combine = rots
        points = rots.view(B, N, 1, 1, 1, 3, 3).matmul(points).squeeze(-1)
        points += trans.view(B, N, 1, 1, 1, 3)

        ''' transform ray directions to world frame, normalize them '''
        rays = rots.view(B, N, 1, 1, 1, 3, 3).matmul(rays.unsqueeze(-1)).squeeze(-1)
        rays = F.normalize(rays, dim=-1)

        intermediates.set(ray_name, rays)
        intermediates.set(out_name, points)

        return intermediates, init_rays, rots, trans

