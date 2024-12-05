import pdb
import torch.nn as nn
import torch.nn.functional as F
import torch
from projects.DistillNeRF.modules.depthanything.dpt import DepthAnything
from projects.DistillNeRF.modules.building_blocks import TimestepEmbedSequential, Bottleneck
import cv2

class DepthAnythingExtractor(nn.Module):
    def __init__(self, config):
        '''
            extract depth feature from depth anything
        '''
        super().__init__()
        self.config = config

        ''' load the model '''
        encoder = 'vitb' # can also be 'vitb' or 'vitl'
        # online
        # self.model = DepthAnything.from_pretrained('LiheYoung/depth_anything_{:}14'.format(encoder))
        # local
        # self.model = DepthAnything.from_pretrained(self.config['model_path'], local_files_only=True)
        model_configs = {
            'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
            'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
            'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]}
        }
        self.model = DepthAnything(model_configs[encoder])
        self.model.load_state_dict(torch.load(f'./aux_models/depth_anything_{encoder}14.pth'))

        ''' image processing '''
        self.target_w = 518
        self.target_h = 518
        self.multiple_of = 14
        self.norm_mean = [0.485, 0.456, 0.406]
        self.norm_std = [0.229, 0.224, 0.225]
        self.norm_mean = torch.tensor(self.norm_mean).unsqueeze(0).unsqueeze(2).unsqueeze(3)
        self.norm_std = torch.tensor(self.norm_std).unsqueeze(0).unsqueeze(2).unsqueeze(3)

        ''' original processing in depth_anything， whcih was rewritten as above to fit in the codebase '''
        # image = np.array(image) / 255.0
        # self.transform = Compose([
        #     Resize(
        #         width=518,
        #         height=518,
        #         resize_target=False,
        #         keep_aspect_ratio=True,
        #         ensure_multiple_of=14,
        #         resize_method='lower_bound',
        #         image_interpolation_method=cv2.INTER_CUBIC,
        #     ),
        #     NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        #     PrepareForNet(),
        # ])

        ''' depth settings, used to calibrate the relative depth from depth anything'''
        self.min_depth_calibrate = 0.1
        self.max_depth_calibrate = 50


    def set_local_rank(self, local_rank):
        self.local_rank = local_rank


    def process_depth(self, depth, min=None, max=None):
        # not used here
        if min is None:
            depth_min = depth.min()
        else:
            depth_min = min
        if max is None:
            depth_max = depth.max()
        else:
            depth_max = max

        depth = torch.clamp(depth, depth_min, depth_max)
        out = (depth - depth_min) / (depth_max - depth_min+0.00001)

        # depth is so far inverse depth
        return out


    def visulaize_depth(self, raw_images, imgs, depth_pred, calibrated_depth, gt_depth, feats_pred):
        import numpy as np
        from sklearn.decomposition import PCA

        # Perform PCA to reduce the deapth_anything feature dimension to 3 features
        w, h = feats_pred.shape[-2], feats_pred.shape[-1]
        w = feats_pred.shape[-2]
        h = feats_pred.shape[-1]
        reshaped_feats_pred = feats_pred.permute(0, 2, 3, 1).reshape(-1, 128).cpu().numpy()
        n_components = 3
        pca = PCA(n_components=n_components)
        pca.fit(reshaped_feats_pred)
        reduced_features = pca.transform(reshaped_feats_pred)
        # reduced_features.reshape(6, w, h, -1)
        reduced_features_ori_shape = torch.from_numpy(reduced_features.reshape(6, w, h, -1))
        reduced_features_ori_shape = reduced_features_ori_shape - torch.min(reduced_features_ori_shape)
        reduced_features_ori_shape = reduced_features_ori_shape / torch.max(reduced_features_ori_shape)

        # plot image and depth
        import matplotlib.pyplot as plt
        # Create a figure with two subplots
        plt.figure(figsize=(12, 6))  # Size of the figure
        for i in range(6):
            # Plot depth image
            plt.subplot(6, 6, i*6+1)  # 1 row, 2 columns, 1st subplot
            plt.imshow(raw_images[i].permute(1,2,0).cpu())  # You can choose different colormaps like 'jet'
            plt.title('RGB Image')
            plt.axis('off')  # Hide axis
            # plt.colorbar()
            
            # Plot RGB image
            plt.subplot(6, 6, i*6+2)  # 1 row, 2 columns, 2nd subplot
            plt.imshow(imgs[i].permute(1,2,0).cpu())  # You can choose different colormaps like 'jet'
            plt.title('Processed input Image')
            plt.axis('off')  # Hide axis
            # Display the plots
            # plt.show(block=False)

            plt.subplot(6, 6, i*6+3)  # 1 row, 2 columns, 2nd subplot
            plt.imshow(depth_pred[i].permute(0,1).cpu(), cmap='gray')  # You can choose different colormaps like 'jet'
            plt.title('Depth Image (disparity)')
            plt.axis('off')  # Hide axis

            plt.subplot(6, 6, i*6+4)  # 1 row, 2 columns, 2nd subplot
            plt.imshow(calibrated_depth[i].permute(0,1).cpu())  # You can choose different colormaps like 'jet'
            plt.title('Calibrated Depth Image')
            plt.axis('off')  # Hide axis

            plt.subplot(6, 6, i*6+5)  # 1 row, 2 columns, 2nd subplot
            plt.imshow(gt_depth[i].permute(1,2,0).cpu())  # You can choose different colormaps like 'jet'
            plt.title('GT Depth Image')
            plt.axis('off')  # Hide axis


            plt.subplot(6, 6, i*6+6)  # 1 row, 2 columns, 2nd subplot
            plt.imshow(reduced_features_ori_shape[i].permute(0,1,2).cpu())  # You can choose different colormaps like 'jet'
            plt.title('Feature Image')
            plt.axis('off')  # Hide axis
            # Display the plots
            # plt.show(block=False)
        plt.show()
        pdb.set_trace()


    def feat_postprocess(self, raw_images, feat):
        w, h = raw_images.shape[-2], raw_images.shape[-1]
        feat_reshaped = F.interpolate(feat, (w, h), mode='bicubic', antialias=True)
        return feat_reshaped
    

    def calibrate_depth(self, depth, gt_depth):
        ''' depth_anything is trained on inverse depth, so we need to calibrate the depth to match the gt_depth'''
        if gt_depth is None:
            print('GT DEPTH NOT available')
            return self.process_depth(depth, min=self.min_depth_calibrate, max=self.max_depth_calibrate)

        # TODO: set sky distance to 500
        valid = torch.logical_and(gt_depth > self.min_depth_calibrate, gt_depth < self.max_depth_calibrate)

        # prediction is in inv depth space so convert gt to inv space
        inv_gt = 1/gt_depth[valid]
        inv_gt = torch.stack([inv_gt, torch.ones_like(inv_gt)], dim=1)
        pred = depth[valid[:,0]].unsqueeze(1)

        # solve Ax = B where A is pred and B is GT to find the mapping a * depthanything_space + b = GT inverse depth
        sol = torch.linalg.lstsq(pred.to(inv_gt), inv_gt)[0].to(depth)

        # return mapped depth
        depth = 1 / (depth * sol[0][0] + sol[0][1])
        return depth
    

    def image_preprocessing(self, raw_images, gt_depth):
        # Calculate new size while maintaining aspect ratio, the target size is lower bound
        w, h = raw_images.shape[-2], raw_images.shape[-1]
        scale_w = self.target_w / w
        scale_h = self.target_h / h
        if scale_w > scale_h:
            new_w = self.target_w
            new_h = int(h * scale_w)
        else:
            new_h = self.target_h
            new_w = int(w * scale_h)

        # resize the shape to be multiple of 14   
        new_w = (int(new_w / self.multiple_of) * self.multiple_of)
        new_h = (int(new_h / self.multiple_of) * self.multiple_of)

        # resize the image using bicubic interpolation
        imgs = F.interpolate(raw_images, (new_w, new_h), mode='bicubic', antialias=True)

        # normalize the image
        imgs = (imgs - self.norm_mean.to(imgs.device)) / self.norm_std.to(imgs.device)

        # resize depth
        if gt_depth is not None:
            gt_depth_resized = F.interpolate(gt_depth, (new_w, new_h), mode='bicubic', antialias=True)
        else:
            gt_depth_resized = None
        
        return imgs, gt_depth_resized
    

    def run_depthanything(self, raw_images, gt_depth):
        ''' prepare data '''
        imgs, gt_depth_resized = self.image_preprocessing(raw_images, gt_depth)

        ''' run DepthAnything '''
        depth_pred, feats_pred = self.model(imgs, get_feature=True)

        ''' calibrate depth (this part is not used) '''
        # calibrated_depth = self.calibrate_depth(depth_pred, gt_depth_resized)
        calibrated_depth = None

        ''' post process feature size '''
        feat = self.feat_postprocess(raw_images, feats_pred)

        ''' visualize for debugging '''
        # self.visulaize_depth(raw_images, imgs, depth_pred, calibrated_depth, gt_depth, feat)

        return depth_pred, feat, calibrated_depth


    def forward(self, scene_data, intermediates):
        ''' prepare data '''
        img = scene_data.get("imgs")
        B,Nv,_,H,W = img.shape
        assert B==1, 'in depth extractor, expecting batch size of 1'
        img = img.reshape(B*Nv, 3, H, W)
        input_depth = scene_data.get("depths")[0]  # input_depth.shape == torch.Size([1, 6, 1, 518, 910]), discard the first dimension
        
        ''' run DepthAnything to get feature on input imgs  '''
        depth_pred, feat, calibrate_depth_pred = self.run_depthanything(img, input_depth) # input_depth: None
        # feat: mean -0.7950, max 107.4845, min -90.7356, std: 8.6076
        # feat = feat / 8.6076
        intermediates.set("depthanything_depth_feature", feat)

        return intermediates

        # ''' run DepthAnything to get depth prediction on target imgs (calibrate, depth_limit)'''
        # # run it to get target depth
        # target_img = scene_data.get("target_imgs")
        # target_cam_classes = scene_data.get("target_cam_classes")
        # target_depth = scene_data.get("target_depth_imgs")[0]
        # B, Nv,_,H,W = target_img.shape

        # if len(target_depth.shape) == 4:
            # target_depth = target_depth.reshape(B*Nv, 1, target_depth.shape[-2], target_depth.shape[-1])
            # target_depth = F.interpolate(target_depth, (H,W))
        # else:
        #     target_depth = None

        # target_img = target_img.reshape(B*Nv, 3, H, W)
        # target_cam_classes = target_cam_classes.reshape(B*Nv)
        # target_depth_pred, _ = self.run_depthanything(target_img, target_cam_classes, target_depth) # target_depth: lidar
        # import pdb; pdb.set_trace()
        # target_depth_pred, target_depth_pred_vis = self.limit_depth(target_depth_pred)
        # intermediates.set("target_depthanything_depth_pred", target_depth_pred)
        # intermediates.set("target_depthanything_depth_pred_vis", target_depth_pred_vis)
    


class DepthAnythingCombiner(nn.Module):
    def __init__(self, config, encoder_feat_chan=256):
        super().__init__()
        self.config = config
        self.encoder_feat_chan = encoder_feat_chan
        self.depthanything_feat_chan = config.depthanything_feat_chan
        self.model_hidden_channels = config.model_hidden_channels
        self.num_blocks = config.num_blocks

        self.combiner = self._make_layer(Bottleneck, self.encoder_feat_chan+self.depthanything_feat_chan, self.model_hidden_channels, self.num_blocks, nn.InstanceNorm2d, stride=1)
        self.parameter_layer = nn.Sequential(
            nn.Conv2d(self.model_hidden_channels, self.depthanything_feat_chan, 1),
            nn.InstanceNorm2d(self.depthanything_feat_chan),
        )
        self.depth_feat_multiplier = config.depth_feat_multiplier
        self.last_layer = nn.Sequential(
            nn.Conv2d(self.depthanything_feat_chan, self.encoder_feat_chan, 1),
            nn.LeakyReLU(0.2),
        )

    def _make_layer(self, block, in_planes, planes, num_blocks, norm_fn, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(in_planes, planes, norm_fn, stride))
            in_planes = planes
        return TimestepEmbedSequential(*layers)


    def forward(self, feat, intermediates):
        B,C,H,W = feat.shape
        depthanything_feat = intermediates.get("depthanything_depth_feature")
        depthanything_feat = F.interpolate(depthanything_feat, (H, W), mode='bilinear')
        comb_feat = self.combiner(torch.cat([feat, depthanything_feat], dim=1), None)
        b_param = self.parameter_layer(comb_feat)
        inp = self.depth_feat_multiplier * depthanything_feat + b_param
        return self.last_layer(inp)
