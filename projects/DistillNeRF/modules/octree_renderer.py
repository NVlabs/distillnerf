"""
The renderer is a module that takes in a ray bundle and returns an image
"""

import math
import pdb
import numpy as np
import torch
import torch.nn as nn
from copy import deepcopy
import time
import torch.nn.functional as F
from projects.DistillNeRF.modules.ray_marcher import MipRayMarcher
from projects.DistillNeRF.models.geometry_parameterized import Geometry_P, volume_render, get_TRANSFORM, create_categorical_depth, xy_z_transform
import kaolin.ops.spc as spc_ops
from einops import rearrange
do_print = False
def soft_clamp(x, cval=3.):
    return x.div(cval).tanh_().mul(cval)

def draw_fig(image1, image2):

    # Create a figure and axis object
    fig, axes = plt.subplots(1, 2)

    # import pdb; pdb.set_trace()
    # Plot the first image
    axes[0].imshow(image1)
    axes[0].axis('off')  # Turn off axis
    axes[0].set_title('Image 1')

    # Plot the second image
    # image2_normalized = (image2 - image2.min()) / (image2.max() - image2.min())
    # axes[1].imshow(image2_normalized)
    axes[1].imshow(image2)
    axes[1].axis('off')  # Turn off axis
    axes[1].set_title('Image 2')

    # Adjust layout to prevent overlap
    plt.tight_layout()

    # Show the plot
    plt.savefig('dino_test.png')
    # plt.show()

def query_image(clip_wrapper, image_pth, queries):
    import matplotlib.pyplot as plt
    # # Assuming similarity is a torch.Tensor, convert it to numpy
    similarity = clip_wrapper.get_similarity(image_pth, queries, 0.1)
    similarity_np = similarity.cpu().numpy()
    # Determine layout for subplots
    num_queries = len(queries)
    cols = 5  # You can adjust the number of columns
    rows = num_queries // cols + (num_queries % cols > 0)

    # Create composite image of heatmaps
    # import pdb; pdb.set_trace()
    plt.figure(figsize=(cols * 4, rows * 4))  # Adjust size as needed
    for idx, query in enumerate(queries):
        plt.subplot(rows, cols, idx + 1)
        plt.imshow(similarity_np[0, :, :, idx], cmap="turbo")
        plt.colorbar()
        plt.title(f"Similarity for query: {query}")

    # Save or show the composite image
    plt.tight_layout()
    plt.savefig('dino_test_query.png')

def reduce_feature_dim(feature, reduction_mat, clip_min, clip_max):
    reduced_feature = feature @ reduction_mat
    reduced_feature = (reduced_feature - clip_min) / (clip_max - clip_min)
    return reduced_feature
'''
https://github.com/googleinterns/IBRNet/blob/master/ibrnet/mlp_network.py
'''


#######################################################################################

def soft_clamp(x, cval=3.):
    return x.div(cval).tanh_().mul(cval)


def sample_from_3dgrid(grid, coordinates, options, geom_circular=False):
    """
    Expects coordinates in shape (batch_size, num_points_per_batch, 3)
    Expects grid in shape (1, channels, H, W, D)
    (Also works if grid has batch size)
    Returns sampled features of shape (batch_size, num_points_per_batch, feature_channels)
    """
    if geom_circular:
        # wrap around the zero for completing the circle
        grid = torch.cat([grid, grid[:, :, :, :1, :]], dim=3)

    batch_size, n_coords, n_dims = coordinates.shape

    sampled_features = torch.nn.functional.grid_sample(grid.expand(batch_size, -1, -1, -1, -1),
                                                       coordinates.reshape(batch_size, 1, 1, -1, n_dims),
                                                       mode='bilinear', padding_mode=options['padding_mode'], align_corners=options['align_corners'])
    N, C, H, W, D = sampled_features.shape
    sampled_features = sampled_features.permute(0, 4, 3, 2, 1).reshape(N, H*W*D, C)
    return sampled_features


class OctreeImportanceRenderer(torch.nn.Module):
    def set(self, property, config, default_value):
        if property in config:
            setattr(self, property, config.get(property))
        else:
            setattr(self, property, default_value)

    def __init__(self, config):
        super().__init__()

        self.config = config

        ''' parameters '''
        self.density_clamp_val = self.config.density_clamp_val
        self.render_clamp_val = self.config.render_clamp_val
        self.max_depth = self.config.max_depth
        self.set('seperate_feat_density_clamp', self.config, False)        
        self.set('sparse_sample_ratio', self.config, 1)
        self.feat_size = self.config.feature_size       # for feature rendering

        ''' True when we render foundation model features '''
        if self.config.embed_rendered_feat:
            self.net = nn.Sequential(
                nn.Linear(self.config.foundation_feat_input_dim, self.config.foundation_feat_hidden_dim),
                nn.ReLU(),
                nn.Linear(self.config.foundation_feat_hidden_dim, self.config.foundation_feat_hidden_dim),
                nn.ReLU(),
                nn.Linear(self.config.foundation_feat_hidden_dim, self.config.foundation_feat_output_dim),
            )

        ''' ray marcher '''
        self.ray_marcher = MipRayMarcher()


    def get_features_from_points(self, points, options, intermediates):
        return self.sample_from_octree(points, intermediates)


    def sample_from_octree(self, coordinates, intermediates):
        """
            get features from the octree, according to the coordinates
        """
        # import pdb; pdb.set_trace()
        batch_size, n_coords, n_dims = coordinates.shape
        # assert batch_size == 1, 'batch size not 1'

        ''' transform the coordinates to parameterized space '''
        xy_x_to_s, xy_s_to_x, z_x_to_s, z_s_to_x = xy_z_transform(self.config.geom_param)
        coordinates = torch.cat([xy_x_to_s(coordinates[:, :, :2]), z_x_to_s(coordinates[:, :, 2:])], dim=-1).squeeze(0)

        feats = []
        for ind, level in enumerate(self.config.octree_levels[::-1]): # [9, 7]
            # start from lowest level
            octree, features = intermediates.get(f"octree{level}"), intermediates.get(f"octree_feats{level}")
            pyramid, prefix = intermediates.get(f"octree_pyramid{level}"), intermediates.get(f"octree_prefix{level}")

            # snap to nearest voxel in the octree
            point_index = spc_ops.unbatched_query(octree, prefix, coordinates, level)

            try:
                # only take density values from the specified level
                # subtract the start index of the level
                point_index = point_index - pyramid[1, level]    
                # mask out index that are before the start index of the specified level
                valid = point_index >= 0                        
                point_index[~valid] = 0
                # collect features
                feat_collected = torch.gather(features, 0, point_index.unsqueeze(1).repeat(1,features.shape[1])) 
                # assign 0 to empty points that are before the start index of the level
                feat_collected[point_index == 0] = 0

                is_invalid_z = coordinates[...,-1] > 1          # invalid points because of z coordinate
                assert point_index[is_invalid_z].sum() == 0     # check these points

            except:
                features = torch.nan_to_num(features)
                feat_collected = torch.zeros(coordinates.shape[0], features.shape[1]).to(coordinates) + features.mean()
                valid = torch.zeros(coordinates.shape[0]).to(coordinates) < -1

            if ind == 0:    # record the finest voxel level
                torch.cuda.synchronize()
                start = time.time()
                feats.append(feat_collected[:, :-1] * valid.float().to(feat_collected).unsqueeze(1))
                fine_density = feat_collected[:, -1:] * valid.float().to(feat_collected).unsqueeze(1)
                if do_print: print("            Projectory - IO: ", round(time.time()-start,6))
                torch.cuda.synchronize()
                fine_occupied_index = valid

            else:           # record the second-finest voxel level
                torch.cuda.synchronize()
                start = time.time()
                feats.append(feat_collected[:, :-1] * valid.float().to(feat_collected).unsqueeze(1))
                coarse_density = feat_collected[:, -1:] * valid.float().to(feat_collected).unsqueeze(1)
                torch.cuda.synchronize()
                if do_print: print("            Projectory - IO: ", round(time.time()-start,6))
                coarse_occupied_index = valid

                # complement the empty space with the coarse density
                density = coarse_density * (~fine_occupied_index).float().unsqueeze(1) + fine_density * fine_occupied_index.float().unsqueeze(1)
                # the remaining empty space is specified by -self.density_clamp_val
                all_occupied_index = torch.logical_or(coarse_occupied_index, fine_occupied_index)
                density[~all_occupied_index] = torch.ones_like(density[~all_occupied_index]) * -self.density_clamp_val

                # concat the features from the two levels, and the final complemented density
                sampled_features = torch.cat(feats + [density], dim=1).unsqueeze(0)

                return sampled_features


    def set_local_rank(self, local_rank):
        self.local_rank = local_rank


    def forward(self, ray_origins, ray_directions, rendering_options, intermediates):
        
        ''' coarse sampling '''
        # uniform sample coarse depth, with stochasticity
        depths_coarse = self.sample_coarse_depth(ray_origins)
        batch_size, num_rays, samples_per_ray, _ = depths_coarse.shape

        # get 3D coordinates for the sampled coarse depth
        sample_coordinates = (ray_origins.unsqueeze(-2) + depths_coarse * (ray_directions).unsqueeze(-2)).reshape(batch_size, -1, 3)
        # [B, num_rays*samples_per_ray, 3]

        # get coarse features
        out = self.collect_feats_from_octree(sample_coordinates, intermediates)
        colors_coarse = out['rgb']                          # [B, num_rays*samples_per_ray, 65]
        density_logits_coarse = out['density_logits']       # [num_rays*samples_per_ray, 1]

        # reshape
        colors_coarse = colors_coarse.reshape(batch_size, num_rays, samples_per_ray, colors_coarse.shape[-1])
        density_logits_coarse = density_logits_coarse.reshape(batch_size, num_rays, samples_per_ray, 1)


        ''' fine sampling '''
        # find the depth with maximum weight
        _, _, weights, _, _, _ = self.ray_marcher(colors_coarse, density_logits_coarse, depths_coarse, rendering_options)
        _, max_weight_ind = torch.max(weights, dim=2)
        max_weight_depth = torch.gather(depths_coarse, 2, max_weight_ind.unsqueeze(-1))

        # create fine depth around the max-weight depth
        fine_depth_resolution = rendering_options['fine_depth_resolution']
        fine_depth_interval = (torch.arange(-fine_depth_resolution//2, fine_depth_resolution//2) * rendering_options['fine_depth_interval']).to(depths_coarse)
        depths_fine = max_weight_depth + fine_depth_interval.reshape(1, 1, -1, 1)
        depths_fine = torch.clamp(depths_fine, 0, self.max_depth)
        
        # get 3D coordinates for the sampled fine depth
        coarse_sample_coordinates = sample_coordinates
        sample_coordinates = (ray_origins.unsqueeze(-2) + depths_fine * ray_directions.unsqueeze(-2)).reshape(batch_size, -1, 3)
        
        # get fine features
        out = self.collect_feats_from_octree(sample_coordinates, intermediates)
        colors_fine = out['rgb']
        density_logits_fine = out['density_logits']

        # reshape
        colors_fine = colors_fine.reshape(batch_size, num_rays, fine_depth_resolution, colors_fine.shape[-1])
        density_logits_fine = density_logits_fine.reshape(batch_size, num_rays, fine_depth_resolution, 1)


        '''  rendering by combining coarse and fine depth  '''
        # add coarse and fine samples
        all_depths, all_colors, all_density_logits, all_coords = self.combine_coarse_and_fine_samples(depths_coarse, colors_coarse, density_logits_coarse,
                                                                depths_fine, colors_fine, density_logits_fine, coarse_sample_coordinates, sample_coordinates)
        # ray marching
        rgb_final, depth_final, weights, weights_all, alpha, all_depths_mid = self.ray_marcher(all_colors, all_density_logits, all_depths, rendering_options)
        # weights_all = weights + sky_weights. But we eventually do not consider sky embeddings, thus weights_all = weights

        return rgb_final, depth_final, weights_all, alpha, all_depths_mid
    

    def create_voxel_coordinate():
        ''' 
        for debugging:
            coordinates = create_voxel_coordinate().to(sample_coordinates.device)
            torch.save(self.sample_from_octree(pc, coordinates, options, intermediates, geom_circular=self.config.geom_circular), "my_tensor.pt")
        '''
        # Grid dimensions and voxel sizes
        bound = [[-50, 50], [-50, 50], [-5, 15]]
        # grid_dims = torch.tensor([200, 200, 40])
        voxel_size = [0.5, 0.5, 0.5]
        voxel_num = [(bound[0][1] - bound[0][0]) / voxel_size[0] + 1, 
                    (bound[1][1] - bound[1][0]) / voxel_size[1] + 1, 
                    (bound[2][1] - bound[2][0]) / voxel_size[2] + 1, 
        ]

        # (bound[0][1] - bound[0][0]) / voxel_size[0]
        # voxel_num = torch.tensor([0.5, 0.5, 0.5])


        # Create a grid of coordinates
        x = torch.linspace(bound[0][0], bound[0][1], int(voxel_num[0]))
        y = torch.linspace(bound[1][0], bound[1][1], int(voxel_num[1]))
        z = torch.linspace(bound[2][0], bound[2][1], int(voxel_num[2]))

        # x = torch.linspace(0, (grid_dims[0] - 1) * voxel_size[0], grid_dims[0]) - 50
        # y = torch.linspace(0, (grid_dims[1] - 1) * voxel_size[1], grid_dims[1]) - 50
        # z = torch.linspace(0, (grid_dims[2] - 1) * voxel_size[2], grid_dims[2]) - 5

        # Create a 3D grid of coordinates
        x, y, z = torch.meshgrid(x, y, z, indexing='ij')

        # Stack the coordinates into a single tensor
        voxel_coordinates = torch.stack([x, y, z], dim=-1)
        # import pdb; pdb.set_trace()
        
        return voxel_coordinates.reshape(1, -1, 3)


    def collect_feats_from_octree(self, sample_coordinates, intermediates):
        sampled_features = self.sample_from_octree(sample_coordinates, intermediates)
        # sampled_features: [B, num_rays*samples_per_ray, feat_dim]

        # give sampled_features a cleaner name
        x = sampled_features
        N, M, C = x.shape
        x = x.reshape(N*M, C)

        ''' clamp the density, and optionally further embed the feat'''
        if self.config.embed_rendered_feat:
            density_logits = x[..., -1:]
            x = x[..., :-1]
            density_logits = soft_clamp(density_logits, cval=10)

            x = self.net(x)

            x = x.reshape(N, M, -1)
            if self.config.seperate_feat_density_clamp:         # usually False
                rgb = soft_clamp(x[..., 1:], cval = self.render_clamp_val)
                density_logits = soft_clamp(x[..., 0:1], cval = self.render_clamp_val)
            else:
                rgb = soft_clamp(x, cval = self.render_clamp_val)
        else:
            density_logits = x[..., -1:]
            x = x[..., :-1]
            density_logits = soft_clamp(density_logits, cval = self.render_clamp_val)

            x = x.reshape(N, M, -1)
            x = torch.cat([x, torch.zeros_like(x[:,:,0:1])], dim=2)
            rgb = x
            
        return {'rgb': rgb, 'density_logits': density_logits}


    def combine_coarse_and_fine_samples(self, depths1, colors1, density_logits1, depths2, colors2, density_logits2, coords1, coords2):
        B, num_rays1, samples_per_ray1, _ = depths1.shape
        B, num_rays2, samples_per_ray2, _ = depths2.shape

        all_depths = torch.cat([depths1, depths2], dim = -2)
        all_colors = torch.cat([colors1, colors2], dim = -2)
        all_density_logits = torch.cat([density_logits1, density_logits2], dim = -2)
        all_coords = torch.cat([coords1.reshape(B, num_rays1, samples_per_ray1, 3), coords2.reshape(B, num_rays2, samples_per_ray2, 3)], dim = -2)

        _, indices = torch.sort(all_depths, dim=-2)
        all_depths = torch.gather(all_depths, -2, indices)
        all_colors = torch.gather(all_colors, -2, indices.expand(-1, -1, -1, all_colors.shape[-1]))
        all_density_logits = torch.gather(all_density_logits, -2, indices.expand(-1, -1, -1, 1))
        all_coords = torch.gather(all_coords, -2, indices.expand(-1, -1, -1, 3))

        return all_depths, all_colors, all_density_logits, all_coords.reshape(B, num_rays1*(samples_per_ray1+samples_per_ray2), 3)

    def sample_coarse_depth(self, ray_origins):
        N, M, _ = ray_origins.shape

        x_to_s, s_to_x, sample_depth_all, sample_depth_start, sample_depth_end, sample_depth_mid, sample_depth_delta, D \
            = create_categorical_depth(self.config.geom_param, self.sparse_sample_ratio)

        # reshpae
        torch.cuda.synchronize()
        start = time.time()
        sample_depth_start = sample_depth_start.reshape(1, 1, D, 1).repeat(N, M, 1, 1).to(device=ray_origins.device)
        sample_depth_end = sample_depth_end.reshape(1, 1, D, 1).repeat(N, M, 1, 1).to(device=ray_origins.device)
        sample_depth_mid = sample_depth_mid.reshape(1, 1, D, 1).repeat(N, M, 1, 1).to(device=ray_origins.device)
        sample_depth_delta = sample_depth_delta.reshape(1, 1, D, 1).repeat(N, M, 1, 1).to(device=ray_origins.device)
        torch.cuda.synchronize()
        if do_print: print("            Projectory - IO: ", round(time.time()-start,6))

        # add stochasticity
        if self.config.stochastic_render_sample:    # True
            sample_depth_mid += torch.rand_like(sample_depth_mid) * sample_depth_delta

        return sample_depth_mid
        