import torch
import torch.nn as nn
import torch.nn.functional as F
from projects.DistillNeRF.modules.building_blocks import TimestepEmbedSequential, Bottleneck
from hydra.utils import instantiate
from projects.DistillNeRF.models.geometry_parameterized import volume_render, create_categorical_depth
import time
do_print = False
def soft_clamp(x, cval=3.):
    return x.div(cval).tanh_().mul(cval)


class CoarseDepthEstimatorFPN(nn.Module):
    '''
        adapted from
        https://github.com/kuangliu/pytorch-fpn/blob/master/fpn.py
    '''
    def set(self, property, config, default_value):
        if property in config:
            setattr(self, property, config.get(property))
        else:
            setattr(self, property, default_value)

    def __init__(self, config):
        super(CoarseDepthEstimatorFPN, self).__init__()
        self.config = config

        ''' depth settings '''
        self.density_activation = config.density_activation
        self.density_softclamp = config.density_softclamp
        self.density_clamp_val = config.density_clamp_val

        ''' parameterized space '''
        if 'sparse_sample_ratio' in self.config.keys(): self.sparse_sample_ratio = self.config.sparse_sample_ratio
        else: self.sparse_sample_ratio = 1
        x_to_s, s_to_x, sample_depth_all, categorical_depth_start, \
            categorical_depth_end, categorical_depth_mid, cateborical_depth_delta, D = create_categorical_depth(self.config.geom_param, self.sparse_sample_ratio)
        self.sample_depth_all, self.categorical_depth_start, self.categorical_depth_end, self.categorical_depth_mid, self.cateborical_depth_delta, self.D = \
            sample_depth_all, categorical_depth_start.cuda(), categorical_depth_end.cuda(), categorical_depth_mid, cateborical_depth_delta, D
        torch.cuda.synchronize()

        ''' feature combiner '''
        # combining depth feature from FPN and depth_anything
        self.set('depth_feature_dim', self.config, 64)          # 64 dimension of fpn depth feature
        self.depth_feature_combiner =  instantiate(config.depth_feature_combiner, encoder_feat_chan=self.depth_feature_dim)

        ''' depth network after feature combiner'''
        num_chan = self.depth_feature_dim
        self.depthnet =  nn.Sequential(
                    nn.Conv2d(self.depth_feature_dim, num_chan, kernel_size=3, padding=1),
                    nn.LeakyReLU(0.2),
                    nn.Conv2d(num_chan, self.D, kernel_size=1, padding=0)
        )

        ''' FPN image depth feature extractor '''
        num_blocks = self.config.fpn_blocks
        self.in_planes = self.depth_feature_dim # 64
        self.downsample = config.downsample
        self.force_width = self.config.input_dim_width
        self.force_height = self.config.input_dim_height
        block = Bottleneck
        self.cam_dep_norm = False
        norm_fn = nn.InstanceNorm2d

        self.conv1 = nn.Conv2d(3, self.in_planes, kernel_size=3, stride=2 if self.downsample > 1 else 1, padding=1, bias=False)
        self.bn1 = norm_fn(self.in_planes)

        # Bottom-up layers
        self.layer1 = self._make_layer(block,  num_chan, num_chan*2, num_blocks[0], norm_fn, stride=1)
        self.layer2 = self._make_layer(block, num_chan*2, num_chan*2, num_blocks[1], norm_fn, stride=2)
        self.layer3 = self._make_layer(block, num_chan*2, num_chan*3, num_blocks[2], norm_fn, stride=2)
        self.layer4 = self._make_layer(block, num_chan*3, num_chan*4, num_blocks[3], norm_fn, stride=2)

        if self.downsample == 1: # True
            self.layer5 = self._make_layer(block, num_chan*4, num_chan*4, num_blocks[3], norm_fn, stride=2)
            self.latlayer0 = nn.Sequential(
                nn.Conv2d(num_chan*4, num_chan*3, kernel_size=1, stride=1, padding=0),
                nn.LeakyReLU(0.2)
            )

        # Top layer
        self.toplayer = nn.Conv2d(num_chan*4, num_chan*3, kernel_size=1, stride=1, padding=0)  # Reduce channels

        # Lateral layers
        self.latlayer1 = nn.Sequential(
            nn.Conv2d(num_chan*3, num_chan*3, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(0.2)
        )
        self.latlayer2 = nn.Sequential(
            nn.Conv2d(num_chan*2, num_chan*3, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(0.2)
        )
        self.latlayer3 = nn.Sequential(
            nn.Conv2d(num_chan*2, num_chan*3, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(0.2)
        )

        if self.downsample == 4:
            self.feat_final = nn.Sequential(
                nn.Conv2d(num_chan*3, self.depth_feature_dim, kernel_size=3, stride=2, padding=1),
                nn.LeakyReLU(0.2)
            )
        elif self.downsample <= 2:
            self.feat_final = nn.Sequential(
                nn.Conv2d(num_chan*3, self.depth_feature_dim, kernel_size=3, stride=1, padding=1),
                nn.LeakyReLU(0.2)
            )

        self.feat_norm = None


    def set_local_rank(self, local_rank):
        self.local_rank = local_rank

    def _make_layer(self, block, in_planes, planes, num_blocks, norm_fn, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(in_planes, planes, norm_fn, stride))
            in_planes = planes
        return TimestepEmbedSequential(*layers)

    def _upsample_add(self, x, y):
        '''Upsample and add two feature maps.
        Args:
          x: (Variable) top feature map to be upsampled.
          y: (Variable) lateral feature map.
        Returns:
          (Variable) added feature map.
        Note in PyTorch, when input size is odd, the upsampled feature map
        with `F.upsample(..., scale_factor=2, mode='nearest')`
        maybe not equal to the lateral feature map size.
        e.g.
        original input size: [N,_,15,15] ->
        conv2d feature map size: [N,_,8,8] ->
        upsampled feature map size: [N,_,16,16]
        So we choose bilinear upsample which supports arbitrary output sizes.
        '''
        _,_,H,W = y.size()
        return F.interpolate(x, size=(H,W), mode='bilinear') + y

    def run_bottom(self, x, classes):
        # Bottom-up
        c6 = None
        class_emb,emb = None, None
        if self.cam_dep_norm:   # False
            emb = self.ind_emb(classes)
            c1 = F.relu(self.bn1(self.conv1(x), emb))
            class_emb = emb
        else:
            c1 = F.relu(self.bn1(self.conv1(x)))
        c2 = self.layer1(c1, class_emb)
        c3 = self.layer2(c2, class_emb)
        c4 = self.layer3(c3, class_emb)
        c5 = self.layer4(c4, class_emb)
        if self.downsample == 1:
            c6 = self.layer5(c5, class_emb)

        return c1,c2,c3,c4,c5,c6,emb

    def run_fpn(self, x, classes=None):
        # Bottom-up
        c1,c2,c3,c4,c5,c6,emb = self.run_bottom(x, classes)

        mv_density, cost = None, None
        if self.downsample == 1:
            p6 = self.toplayer(c6)
            p5 = self._upsample_add(p6, self.latlayer0(c5))
        else:
            p5 = self.toplayer(c5)
        p4 = self._upsample_add(p5, self.latlayer1(c4))

        p3 = self._upsample_add(p4, self.latlayer2(c3))
        p2 = self._upsample_add(p3, self.latlayer3(c2))

        p2 = self.feat_final(p2)
        if self.feat_norm is not None:  # False
            p2 = self.feat_norm(p2, emb)

        return p2, emb, cost, mv_density

    def ray_marching(self, density_logit):
        ''' density logit -> density'''
        start = time.time()
        # activation bias of -1 makes things initialize better
        if self.density_activation == 'sigmoid':
            densities = F.sigmoid(density_logit - 1)
        elif self.density_activation == 'softplus':
            densities = F.softplus(density_logit - 1)
        elif self.density_activation == 'exp':
            densities = torch.exp(density_logit - 1)
        if do_print: print("            Coarse Depth - ray marching - density logit -> density:", round(time.time()-start,6))


        start = time.time()
        ''' change the shape to [num_ray, depth] for rendering '''
        # [n, d, h, w] -> [n, h, w, d] -> [n*h*w, d]
        densities = densities.permute(0, 2, 3, 1).reshape(-1, self.D)
        if do_print: print("            Coarse Depth - ray marching - reshape1:", round(time.time()-start,6))

        ''' create categorical depth, same shape as density '''
        start = time.time()
        num_rays = densities.shape[0]
        # import pdb; pdb.set_trace()
        categorical_depth_start = self.categorical_depth_start.reshape(1, -1).repeat(num_rays, 1).to(densities.device)
        categorical_depth_end = self.categorical_depth_end.reshape(1, -1).repeat(num_rays, 1).to(densities.device)
        if do_print: print("            Coarse Depth - ray marching - reshape2:", round(time.time()-start,6))

        ''' run volume_render '''
        start = time.time()
        depths, weights = volume_render(categorical_depth_start, categorical_depth_end, densities)
        if do_print: print("            Coarse Depth - ray marching - volume_render:", round(time.time()-start,6))
        
        start = time.time()
        ''' recover the original shape '''
        depths = depths.reshape(-1, self.force_height, self.force_width)
        weights = weights.reshape(-1, self.force_height, self.force_width, self.D).permute(0, 3, 1, 2)

        categorical_depth_mid = (categorical_depth_start + categorical_depth_end) / 2.0
        categorical_depth_mid = categorical_depth_mid.reshape(-1, self.force_height, self.force_width, self.D).permute(0, 3, 1, 2)
        if do_print: print("            Coarse Depth - ray marching - recover shape:", round(time.time()-start,6))

        return depths, weights, categorical_depth_mid


    def forward(self, raw_imgs, intermediates, classes=None):
        ''' prepare image size '''
        if self.force_width > 0:
            reshaped_imgs = F.interpolate(raw_imgs, (self.force_height, self.force_width), mode='bilinear', antialias=True)
            intermediates.set('monodepth_img_size', (self.force_height, self.force_width))

        ''' run FPN '''
        start = time.time()
        img_feat, emb, cost, mv_density = self.run_fpn(reshaped_imgs, classes=classes)
        if do_print: print("        Coarse Depth - fpn:", round(time.time()-start,6))

        ''' combine depth features from depth anything '''
        start = time.time()
        combined_depth_feat = self.depth_feature_combiner(img_feat, intermediates)
        if do_print: print("        Coarse Depth - depth combine:", round(time.time()-start,6))
        intermediates.set('monodepthfeat', combined_depth_feat)
        del intermediates.depthanything_depth_feature

        ''' predict density by categorical depth '''
        start = time.time()
        density_logit = self.depthnet(combined_depth_feat)  
        if do_print: print("        Coarse Depth - depth layer:", round(time.time()-start,6))
        if self.density_softclamp:
            density_logit = soft_clamp(density_logit, cval=self.density_clamp_val)

        ''' rendering to get depth and weights'''
        start = time.time()
        depths, weights, categorical_depth_mid = self.ray_marching(density_logit)
        if do_print: print("        Coarse Depth - ray marching:", round(time.time()-start,6))
        intermediates.set("coarse_depth_weights", weights)
        intermediates.set("coarse_depth_samples", categorical_depth_mid)
        intermediates.set("coarse_depth", depths)

        return intermediates
    
