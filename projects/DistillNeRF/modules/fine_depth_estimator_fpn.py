'''
https://github.com/kuangliu/pytorch-fpn/blob/master/fpn.py

FPN in PyTorch.
See the paper "Feature Pyramid Networks for Object Detection" for more details.
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from projects.DistillNeRF.modules.building_blocks import TimestepEmbedSequential, Bottleneck
from projects.DistillNeRF.models.geometry_parameterized import volume_render
from einops import rearrange
import time
do_print = False

def soft_clamp(x, cval=3.):
    return x.div(cval).tanh_().mul(cval)

def soft_clamp_sig(x, cval=3.):
    return x.div(cval).sigomid_().mul(cval)

class FineDepthEstimatorFPN(nn.Module):
    def set(self, property, config, default_value):
        if property in config:
            setattr(self, property, config.get(property))
        else:
            setattr(self, property, default_value)

    def __init__(self, config):
        super(FineDepthEstimatorFPN, self).__init__()
        self.config = config

        ''' depth sampling config'''
        self.num_depth_candidate_samples = config.num_depth_candidate_samples
        self.sample_eps_ratio = config.sample_eps_ratio
        self.density_activation = config.density_activation
        self.depth_candidates = torch.arange(-self.num_depth_candidate_samples//2, self.num_depth_candidate_samples//2).cuda().reshape(1, -1, 1, 1)
        torch.cuda.synchronize()
        


        ''' feature combiner'''
        self.set('fpn_middle_channels', self.config, 16)
        self.set('fpn_final_channels', self.config, 16)
        num_chan = self.fpn_middle_channels
        self.fpn_feature_extractor_chan = self.fpn_final_channels                   # 16, very shallow network as we already have pretrained feature network
        self.monodepthfeature_chan = config.depth_feature_dim   
        self.combiner = nn.Sequential(
            nn.Conv2d(self.fpn_feature_extractor_chan+self.monodepthfeature_chan, num_chan*2, kernel_size=3, padding=1),
            nn.InstanceNorm2d(num_chan*2),
            nn.LeakyReLU(0.2),
            nn.Conv2d(num_chan*2, num_chan*2, kernel_size=3, padding=1),
            nn.InstanceNorm2d(num_chan*2),
            nn.LeakyReLU(0.2),
        )
        
        self.set('candidate_depth_feature_combine', self.config, 'multiply')        # 'multiple' or 'add'


        ''' depth and feature encoder after feature combiner '''
        voxel_depth_dim = 1
        self.cv_feat_size = self.config.feature_size
        
        self.featnet = nn.Sequential(
            nn.Conv2d(num_chan*2, num_chan, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(num_chan, self.cv_feat_size-voxel_depth_dim, kernel_size=1, padding=0),
        )
        self.depthnet = nn.Sequential(
            nn.Conv2d(num_chan*2, num_chan, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(num_chan, self.cv_feat_size, kernel_size=1, padding=0),
        )
        self.depth_feat_normalize = nn.LayerNorm([self.cv_feat_size])


        ''' depth expander '''
        self.depth_expander = nn.Sequential(
            nn.Linear(1, self.cv_feat_size),
            nn.LayerNorm([self.cv_feat_size])
        )
        self.feature_predictor = nn.Sequential(
            nn.Conv3d(self.cv_feat_size, num_chan, kernel_size=(3,1,1), padding=(1,0,0)),
            nn.InstanceNorm3d(num_chan),
            nn.LeakyReLU(0.2),
            nn.Conv3d(num_chan, num_chan, kernel_size=(3,1,1), padding=(1,0,0)),
            nn.InstanceNorm3d(num_chan),
            nn.LeakyReLU(0.2),
            nn.Conv3d(num_chan, 1, kernel_size=(3,1,1), padding=(1,0,0)),
            # nn.Conv3d(num_chan, 1 if not self.bypass_density_pred else self.config.feature_size, kernel_size=(3,1,1), padding=(1,0,0)),
        )


        ''' image feature extractor '''
        num_blocks = self.config.fpn_blocks
        self.set('in_planes', self.config, 16)
        self.downsample = self.config.downsample
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

        if self.downsample == 1:
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

        if self.downsample == 4:  # False
            self.feat_final = nn.Sequential(
                nn.Conv2d(num_chan*3, self.fpn_feature_extractor_chan, kernel_size=3, stride=2, padding=1),
                nn.LeakyReLU(0.2)
            )
        elif self.downsample <= 2:  # True
            self.feat_final = nn.Sequential(
                nn.Conv2d(num_chan*3, self.fpn_feature_extractor_chan, kernel_size=3, stride=1, padding=1),
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
        if self.cam_dep_norm:
            emb = self.ind_emb(classes)
            c1 = F.relu(self.bn1(self.conv1(x), emb))
            class_emb = emb
        else:
            # if self.bottleneck_cam_dep_layer_only:
                # emb = self.ind_emb(classes)
            c1 = F.relu(self.bn1(self.conv1(x)))
        c2 = self.layer1(c1, class_emb)
        c3 = self.layer2(c2, class_emb)
        c4 = self.layer3(c3, class_emb)
        c5 = self.layer4(c4, class_emb)
        if self.downsample == 1:
            c6 = self.layer5(c5, class_emb)

        return c1,c2,c3,c4,c5,c6,emb


    def run_fpn(self, x, classes=None):
        start = time.time()
        # Bottom-up        
        c1,c2,c3,c4,c5,c6,emb = self.run_bottom(x, classes)
        if do_print: print("            Encoder - Fine depth - fpn 1 ", round(time.time()-start,6))


        start = time.time()
        mv_density, cost = None, None
        if self.downsample == 1:
            p6 = self.toplayer(c6)
            p5 = self._upsample_add(p6, self.latlayer0(c5))
        else:
            p5 = self.toplayer(c5)
        p4 = self._upsample_add(p5, self.latlayer1(c4))
        if do_print: print("            Encoder - Fine depth - fpn 2 ", round(time.time()-start,6))


        start = time.time()
        p3 = self._upsample_add(p4, self.latlayer2(c3))
        p2 = self._upsample_add(p3, self.latlayer3(c2))
        # Smooth
        p2 = self.feat_final(p2)

        if self.feat_norm is not None:
            p2 = self.feat_norm(p2, emb)
        if do_print: print("            Encoder - Fine depth - fpn 3 ", round(time.time()-start,6))

        return p2, emb, cost, mv_density
    

    def ray_marching(self, density_logit, depth_candidates):
        ''' density logit -> density'''
        density_logit_mid = (density_logit[:, 0, 1:] + density_logit[:, 0, :-1]) / 2
        if self.density_activation == 'sigmoid':
            densities_mid = F.sigmoid(density_logit_mid - 1)
        elif self.density_activation == 'softplus':
            densities_mid = F.softplus(density_logit_mid - 1)
        elif self.density_activation == 'exp':
            densities_mid = torch.exp(density_logit_mid - 1)

        ''' change the shape to [num_ray, depth] for rendering '''
        D = densities_mid.shape[1]
        H = densities_mid.shape[2]
        W = densities_mid.shape[3]
        # [n, d, h, w] -> [n, h, w, d] -> [n*h*w, d]
        densities_mid_reshaped = densities_mid.permute(0, 2, 3, 1).reshape(-1, D)

        ''' create sample depth, same shape as density_mid '''
        t_starts = depth_candidates[:, :-1].permute(0, 2, 3, 1).reshape(-1, D)
        t_ends = depth_candidates[:, 1:].permute(0, 2, 3, 1).reshape(-1, D)

        ''' run volume_render '''
        depths, weights = volume_render(t_starts, t_ends, densities_mid_reshaped)

        ''' recover the original shape '''
        depths = depths.reshape(-1, H, W)
        weights = weights.reshape(-1, H, W, D).permute(0, 3, 1, 2)

        sample_depth_mid = (t_starts + t_ends) / 2.0
        sample_depth_mid = sample_depth_mid.reshape(-1, H, W, D).permute(0, 3, 1, 2)

        return depths, weights, sample_depth_mid


    def forward(self, imgs, intermediates, classes=None):
        ''' load coarse_depth and monodepthfeat from finetuned monocular model '''
        with torch.no_grad():
            coarse_depth = intermediates.get("coarse_depth") 
            monodepthfeat = intermediates.get("monodepthfeat")

        start = time.time()
        ''' add small depth offsets to coarse_depth to get fine_depth_candidates'''
        # import pdb; pdb.set_trace()
        # depth_candidates = torch.arange(-self.num_depth_candidate_samples//2, self.num_depth_candidate_samples//2).to(coarse_depth).reshape(1, -1, 1, 1)
        depth_candidates = self.depth_candidates
        if do_print: print("        Encoder - Fine depth - add offsets ", round(time.time()-start,6))
        D = depth_candidates.shape[1]
        depth_candidates_raw = coarse_depth.unsqueeze(1) + depth_candidates * torch.clamp(coarse_depth.unsqueeze(1) * self.sample_eps_ratio, 0.05, 0.3)
                            # ([1, 128, 228])           [1, 15, 1, 1]                                                       0.025
        fine_depth_candidates = F.relu(depth_candidates_raw)
        intermediates.set("fine_depth_candidates", fine_depth_candidates)

        start = time.time()
        ''' encode fine_depth_candidates to get it features '''
        fine_depth_candidates_feat = self.depth_expander(fine_depth_candidates.unsqueeze(-1))
        fine_depth_candidates_feat = rearrange(fine_depth_candidates_feat, 'b d h w c -> b c d h w')  # torch.Size([6, 65, 15, 128, 228])
        if do_print: print("        Encoder - Fine depth - encode offsets ", round(time.time()-start,6))

        torch.cuda.synchronize()
        start = time.time()
        # torch.cuda.empty_cache()
        ''' run FPN '''
        # import pdb; pdb.set_trace()
        fpn_feat, _,_,_ = self.run_fpn(imgs, classes=classes)
        if do_print: print("        Encoder - Fine depth - fpn ", round(time.time()-start,6))
        torch.cuda.synchronize()

        start = time.time()
        ''' combine depth features from FPNG and pretrained depth model ''' 
        combined_feat = self.combiner(torch.cat([fpn_feat, monodepthfeat], dim=1))
        if do_print: print("        Encoder - Fine depth - combine feature ", round(time.time()-start,6))

        start = time.time()
        ''' encode combined features to get imgfeat and depthfeat'''
        imgfeat = soft_clamp(self.featnet(combined_feat), cval=5.)
        depthfeat = self.depthnet(combined_feat)
        depthfeat = self.depth_feat_normalize(depthfeat.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)    # torch.Size([6, 65, 128, 228])
        depthfeat = depthfeat.unsqueeze(2).repeat(1,1,D,1,1)    # torch.Size([6, 65, 15, 128, 228])
        if do_print: print("        Encoder - Fine depth - imgfeat and depthfeat ", round(time.time()-start,6))

        start = time.time()
        ''' predict density for fine_depth_candidates, according to fine_depth_candidates_feat and depthfeat '''
        if self.candidate_depth_feature_combine == 'multiply':
            # [n, 1, d, h, w] = conv( [n, c, d, h, w] * [n, c, d, h, w] )
            out = self.feature_predictor(depthfeat * fine_depth_candidates_feat)
        elif self.candidate_depth_feature_combine == 'add':
            out = self.feature_predictor(depthfeat + fine_depth_candidates_feat)
        density_prob = torch.sigmoid(out)
        density_logit = (density_prob * 2 - 1) * self.config.density_clamp_val
        fine_depth, fine_depth_candidates_mid_weights, fine_depth_candidates_mid = self.ray_marching(density_logit, fine_depth_candidates.squeeze(1))
        intermediates.set("fine_depth", fine_depth)                                                            # [n, h, w]
        intermediates.set("fine_depth_candidates_mid_weights", fine_depth_candidates_mid_weights)              # [n, d, h, w]
        intermediates.set("fine_depth_candidates_mid", fine_depth_candidates_mid)                              # [n, d, h, w]
        if do_print: print("        Encoder - Fine depth - ray marching ", round(time.time()-start,6))

        start = time.time()
        ''' lift 2D image features to 3D according to density_prob, append density_logit to the end of the features '''
        final_feat = density_prob * imgfeat.unsqueeze(2)
        final_feat = torch.cat([final_feat, density_logit], dim=1)
        if do_print: print("        Encoder - Fine depth - lift 3D ", round(time.time()-start,6))

        return [final_feat, fine_depth_candidates_mid_weights], intermediates