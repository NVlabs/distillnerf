from copy import deepcopy
import pdb, time
from projects.DistillNeRF.utils.custom_kaolin import custom_unbatched_pointcloud_to_spc
from projects.DistillNeRF.models.geometry_parameterized import Geometry_P, get_TRANSFORM, xy_z_transform
import torch.nn as nn
import torch.nn.functional as F
import torch
from hydra.utils import instantiate
do_print = False

def soft_clamp(x, cval=3.):
    return x.div(cval).tanh_().mul(cval)

def mem(name): return 

class MultViewPoolingOctreeEncoder(nn.Module):

    def set(self, property, config, default_value):
        if property in config:
            setattr(self, property, config.get(property))
        else:
            setattr(self, property, default_value)

    def __init__(self, config, geometry: Geometry_P):
        super().__init__()
        self.config = config
        self.geometry = [geometry]
        self.set('max_pool_feats', self.config, False)
        self.sparse_voxel_convnet = instantiate(config.sparse_voxel_convnet)
        if self.max_pool_feats:
            self.pooling = 'max'
        else:
            self.pooling = 'mean'

    def get_geometry(self):
        return self.geometry[0]

    def set_local_rank(self, local_rank):
        self.local_rank = local_rank

    def get_spc(self, feat_coord_in_world_frame, level, feat):
        '''
            pyramids:
                torch.IntTensor of shape [batch_size, 2, max_level+2]. Contains layout information for each octree 
                pyramids[:, 0] represent the number of points in each level of the octrees, 
                pyramids[:, 1] represent the starting index of each level of the octree.
            exsum:
                torch.IntTensor of shape 
                is the exclusive sum of the bit counts of each octrees byte.
            point_hierarchies:
                torch.IntTensor of shape 
                correspond to the sparse coordinates at all levels. We refer to this Packed tensor as the structured point hierarchies.
        '''
        spc = custom_unbatched_pointcloud_to_spc(feat_coord_in_world_frame, level, features=feat, mode=self.pooling)
        point_hierarchy, pyramid, prefix = spc.point_hierarchies, spc.pyramids[0], spc.exsum
        start = time.time()
        octree, features = spc.octrees, spc.features.to(feat)
        if do_print: print("            Multi-view fusion - Create octree - I/O:", round(time.time()-start,6))
        return spc, point_hierarchy, pyramid, prefix, octree, features

    def forward(self, scene_data, intermediates):
        """
            Pool features from seperate camera images using geometry of cameras
            Returns:
                voxels -> 3D voxel grid of shape (BS, C, Z, Y, X)
                feat_coord_in_world_frame ->
        """

        ''' 1. reshape the features '''
        encoded_scene = intermediates.encoded_scene
        feat_size = self.config.feature_size                    # 65
        D = encoded_scene.shape[2]
        B, N, _, imH, imW = scene_data.imgs.shape
        # [n, c, d, h, w] -> [b, n, c, d, h, w] -> [b, n, d, h, w, c = 65]
        encoded_scene = encoded_scene.view(B, N, feat_size, D, imH//self.config.downsample, imW//self.config.downsample)
        encoded_scene = encoded_scene.permute(0, 1, 3, 4, 5, 2)         # torch.Size([1, 6, 15, 128, 228, 65])
        intermediates.set("encoded_scene", encoded_scene)     
        
        
        ''' 2. Get world-frame coordinates of the features, according to depths and camera poses '''
        feat_coord_in_world_frame = self.get_geometry().get_3D_coord_in_world_frame(scene_data, depths=intermediates.get("fine_depth_candidates").detach())
        C = encoded_scene.shape[-1]
        encoded_scene = encoded_scene.reshape(-1, C)
        feat_coord_in_world_frame = feat_coord_in_world_frame.reshape(-1, 3)


        ''' 3. convert the coordinate to the parameterized space, [-∞， +∞] -> [-1, 1] '''
        xy_x_to_s, xy_s_to_x, z_x_to_s, z_s_to_x = xy_z_transform(self.config.geom_param)
        feat_coord_in_world_frame = torch.cat([xy_x_to_s(feat_coord_in_world_frame[:, :2]), z_x_to_s(feat_coord_in_world_frame[:, 2:])], dim=1)


        ''' 4. create octrees (multi-view pooling happens here) '''
        start = time.time()
        density_logit = encoded_scene[:, -1:]
        x_feat = torch.chunk(encoded_scene[:, :-1], len(self.config.octree_levels), dim=1)
        for ind, level in enumerate(self.config.octree_levels):                 # 7, 9
            feat = torch.cat([x_feat[ind], density_logit], dim=1)
            spc, point_hierarchy, pyramid, prefix, octree, features = self.get_spc(feat_coord_in_world_frame, level, feat)
                                                                        #        [2626560, 3]       [2626560, 33]
            intermediates.set(f"octree{level}", octree)
            intermediates.set(f"octree_feats{level}", features)
            intermediates.set(f"octree_pyramid{level}", pyramid)    # [2, 9], [2, 11]
            intermediates.set(f"octree_prefix{level}", prefix)
            intermediates.set(f"octree_point_hierarchy{level}", point_hierarchy)
        if do_print: print("        Multi-view fusion - Create octree:", round(time.time()-start,6))


        torch.cuda.synchronize()
        ''' 5. sparse convolution of octrees '''
        start = time.time()
        voxel_net_out_7 = self.sparse_voxel_convnet([intermediates.get(f"octree_feats{level}").to(dtype=torch.float32) for level in self.config.octree_levels[::-1]], octree, point_hierarchy, self.config.octree_levels, pyramid, prefix)
        if do_print: print("        Multi-view fusion - Sparse Convolution :", round(time.time()-start,6))
        torch.cuda.synchronize()

        # intermediates.set(f"octree_feats{self.config.octree_levels[0]}", soft_clamp(voxel_net_out_7, cval=5.).to(feat))
        intermediates.set(f"octree_feats{self.config.octree_levels[0]}", torch.cat((soft_clamp(voxel_net_out_7[:,:-1], cval=5.), soft_clamp(voxel_net_out_7[:,-1:], cval=self.config.density_clamp_val)), -1).to(feat))


        return intermediates
