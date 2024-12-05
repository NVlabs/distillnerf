import pdb
import torch.nn as nn
import torch.nn.functional as F
import time
import torch
from hydra.utils import instantiate

from projects.DistillNeRF.models.geometry_parameterized import Geometry_P

class NerfProjector(nn.Module):
    def set(self, property, config, default_value):
        if property in config:
            setattr(self, property, config.get(property))
        else:
            setattr(self, property, default_value)

    def __init__(self, config, geometry: Geometry_P):
        super().__init__()

        self.config = config
        self.geometry = [geometry]

        self.set('fine_depth_resolution', self.config, 0)
        self.set('fine_depth_interval', self.config, 0)
        self.render_ray_batch_num = self.config.render_ray_batch_num if 'render_ray_batch_num' in self.config else 1
        self.nerf_rendering_option = {
            'fine_depth_resolution':  self.fine_depth_resolution,
            'fine_depth_interval': self.fine_depth_interval,
            'max_depth': self.config.geom_param.max_depth,
            'min_depth': self.config.geom_param.min_depth,
            'sky_depth': self.config.geom_param.sky_depth,
            'depth_interval': self.config.geom_param.dbound[2],
            'density_activation': self.config.density_activation
        }

        self.volume_renderer = instantiate(config.volume_renderer)

    def set_local_rank(self, local_rank):
        self.local_rank = local_rank
        self.volume_renderer.set_local_rank(local_rank)

    def get_geometry(self):
        return self.geometry[0]

    def render_one_chunk(self, cam_locs, ray_dirs, intermediates):
        new_rgb, new_depth_img, new_nerf_weights_all, new_nerf_alpha, new_sample_depths = [], [], [], [], []

        # split one chunk to multiple ray batchs, to further fit in memory
        render_ray_batch_size = cam_locs.shape[1] // self.render_ray_batch_num
        for batch_idx in range(self.render_ray_batch_num):   # 16
            cur_rgb, cur_depth_img, cur_nerf_weights_all, cur_nerf_alpha, all_depths_mid \
            = self.volume_renderer(
                    cam_locs[:, batch_idx*render_ray_batch_size:(batch_idx+1)*render_ray_batch_size],
                    ray_dirs[:, batch_idx*render_ray_batch_size:(batch_idx+1)*render_ray_batch_size],
                    self.nerf_rendering_option,
                    intermediates,
                )
            import pdb; pdb.set_trace()
            new_rgb.append(cur_rgb)
            new_depth_img.append(cur_depth_img)
            new_nerf_weights_all.append(cur_nerf_weights_all)
            new_nerf_alpha.append(cur_nerf_alpha)
            new_sample_depths.append(all_depths_mid)
            # def malloc(name): pass#print(name, f"{torch.cuda.memory_allocated(0)/ 1000000000:,}","GB");

        rgb = torch.cat(new_rgb, dim=1)
        depth_img = torch.cat(new_depth_img, dim=1)
        nerf_weights_all = torch.cat(new_nerf_weights_all, dim=1)
        nerf_alpha = torch.cat(new_nerf_alpha, dim=1)
        sample_depths = torch.cat(new_sample_depths, dim=1)

        return rgb, depth_img, nerf_weights_all, nerf_alpha, sample_depths

    def forward(self, scene_data, intermediates):

        intermediates, target_init_rays, target_rots, target_trans = \
            self.get_geometry().generate_render_rays_points(scene_data, intermediates)


        ''' precompute the dimensions '''        
        points = intermediates.get("target_points")         # b, n, d, h, w, 3
        coords = points.flatten(0,1)                        # b*n, d, h, w, 3
        B, N, _, imH, imW, _ = points.shape
        BN = coords.shape[0]


        ''' reshape rendering rays/cam_loc '''
        ray_dirs = F.normalize(intermediates.rays[:,:,0,:,:], dim=-1).reshape(BN, imH*imW, 3)               # b*n, imH*imW, 3
        cam_locs = scene_data.target_trans.clone().reshape(BN, 1, 3).repeat(1, ray_dirs.shape[1], 1)        # b*n, imH*imW, 3
        ray_dirs = ray_dirs.reshape(-1, 3).unsqueeze(0)
        cam_locs = cam_locs.reshape(-1, 3).unsqueeze(0)


        ''' split rays into chuncks to fit memory '''
        render_ray_batch_size = cam_locs.shape[1] // self.render_ray_batch_num
        new_rgb, new_depth_img, new_nerf_weights_all, new_nerf_alpha, new_sample_depths = [], [], [], [], []
        for batch_idx in range(self.render_ray_batch_num):   # 16
            cur_rgb, cur_depth_img, cur_nerf_weights_all, cur_nerf_alpha, all_depths_mid \
            = self.volume_renderer(
                    cam_locs[:, batch_idx*render_ray_batch_size:(batch_idx+1)*render_ray_batch_size],
                    ray_dirs[:, batch_idx*render_ray_batch_size:(batch_idx+1)*render_ray_batch_size],
                    self.nerf_rendering_option,
                    intermediates,
                )
            new_rgb.append(cur_rgb)
            new_depth_img.append(cur_depth_img)
            new_nerf_weights_all.append(cur_nerf_weights_all)
            new_nerf_alpha.append(cur_nerf_alpha)
            new_sample_depths.append(all_depths_mid)
            
        rgb = torch.cat(new_rgb, dim=1).reshape(BN, imH*imW, -1)
        depth_img = torch.cat(new_depth_img, dim=1).reshape(BN, imH*imW, -1)
        nerf_weights_all = torch.cat(new_nerf_weights_all, dim=1).reshape(BN, imH*imW, -1, 1)
        nerf_alpha = torch.cat(new_nerf_alpha, dim=1).reshape(BN, imH*imW, -1, 1)
        sample_depths_all = torch.cat(new_sample_depths, dim=1).reshape(BN, imH*imW, -1, 1)


        ''' reshape and save '''
        # rgb features
        projection = rgb.permute(0, 2, 1)
        projection = projection.reshape(projection.shape[0], projection.shape[1], self.config.render_img_height, self.config.render_img_width)
        target_2d_features = projection.reshape(B, -1, projection.shape[1], self.config.render_img_height,self.config.render_img_width)
        target_2d_features = target_2d_features.permute(0,1,3,4,2)
        intermediates.set("target_2d_features", target_2d_features)

        # depth
        depth_image = depth_img.permute(0, 2, 1).reshape(BN, 1, imH, imW)
        intermediates.set("target_pred_depth_image", depth_image)

        # weights
        nerf_weights_all = nerf_weights_all.permute(0, 2, 3, 1).reshape(BN, -1, imH, imW)
        intermediates.set("target_weights_all", nerf_weights_all)

        # sampled depth
        sample_depths_all = sample_depths_all.permute(0, 2, 3, 1).reshape(BN, -1, imH, imW)
        intermediates.set("target_sample_depths_all", sample_depths_all)

        # delete some features not used later, to save memory
        del intermediates.target_points
        del intermediates.rays

        return intermediates


    def forward_old(self, scene_data, intermediates):

        intermediates, target_init_rays, target_rots, target_trans = \
            self.get_geometry().generate_render_rays_points(scene_data, intermediates)


        ''' precompute the dimensions '''        
        points = intermediates.get("target_points")         # b, n, d, h, w, 3
        coords = points.flatten(0,1)                        # b*n, d, h, w, 3
        B, N, _, imH, imW, _ = points.shape
        BN = coords.shape[0]


        ''' reshape rendering rays/cam_loc '''
        ray_dirs = F.normalize(intermediates.rays[:,:,0,:,:], dim=-1).reshape(BN, imH*imW, 3)               # b*n, imH*imW, 3
        cam_locs = scene_data.target_trans.clone().reshape(BN, 1, 3).repeat(1, ray_dirs.shape[1], 1)        # b*n, imH*imW, 3
    

        ''' split rays into chuncks to fit memory '''
        new_rgb, new_depth_img, new_nerf_weights_all, new_nerf_alpha, new_sample_depths_all = [], [], [], [], []
        chunk_num = BN # set the chunk number to be equal to number of cameras
        import pdb; pdb.set_trace()
        chunk_size = BN // chunk_num
        for chunk_idx in range(chunk_num):
            cur_rgb, cur_depth_img, cur_nerf_weights_all, cur_nerf_alpha, cur_sample_depths \
                    = self.render_one_chunk(
                        cam_locs[chunk_idx * chunk_size:(chunk_idx + 1) * chunk_size],
                        ray_dirs[chunk_idx * chunk_size:(chunk_idx + 1) * chunk_size],
                        intermediates
                    )
            new_rgb.append(cur_rgb)
            new_depth_img.append(cur_depth_img)
            new_nerf_weights_all.append(cur_nerf_weights_all)
            new_nerf_alpha.append(cur_nerf_alpha)
            new_sample_depths_all.append(cur_sample_depths)

        import pdb; pdb.set_trace()
        rgb = torch.cat(new_rgb, dim=0)
        depth_img = torch.cat(new_depth_img, dim=0)
        nerf_weights_all = torch.cat(new_nerf_weights_all, dim=0)
        nerf_alpha = torch.cat(new_nerf_alpha, dim=0)
        sample_depths_all = torch.cat(new_sample_depths_all, dim=0)


        ''' reshape and save '''
        # rgb features
        projection = rgb.permute(0, 2, 1)
        projection = projection.reshape(projection.shape[0], projection.shape[1], self.config.render_img_height, self.config.render_img_width)
        target_2d_features = projection.reshape(B, -1, projection.shape[1], self.config.render_img_height,self.config.render_img_width)
        target_2d_features = target_2d_features.permute(0,1,3,4,2)
        intermediates.set("target_2d_features", target_2d_features)

        # depth
        depth_image = depth_img.permute(0, 2, 1).reshape(BN, 1, imH, imW)
        intermediates.set("target_pred_depth_image", depth_image)

        # weights
        nerf_weights_all = nerf_weights_all.permute(0, 2, 3, 1).reshape(BN, -1, imH, imW)
        intermediates.set("target_weights_all", nerf_weights_all)

        # sampled depth
        sample_depths_all = sample_depths_all.permute(0, 2, 3, 1).reshape(BN, -1, imH, imW)
        intermediates.set("target_sample_depths_all", sample_depths_all)

        # delete some features not used later, to save memory
        del intermediates.target_points
        del intermediates.rays

        return intermediates