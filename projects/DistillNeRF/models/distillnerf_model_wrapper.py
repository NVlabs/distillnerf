import warnings
from typing import Tuple
import os, math
import torch
import torch.nn.functional as F
import numpy as np
from mmdet3d.models import builder
from mmdet3d.models.builder import DETECTORS
from mmdet.models.detectors import BaseDetector
from hydra.utils import instantiate
from ..losses import DepthClampLoss, LPIPSLoss, NerfWeightEntropyLoss, RGBL1Loss, MSELoss, EmernerfDepthClampLoss, DepthAnythingDepthClampLoss, LineOfSightLoss, OpacityLoss
from omegaconf import OmegaConf
from projects.DistillNeRF.utils.utils import Container
from projects.DistillNeRF.utils.vis import save_image_horizontally, DenseCLIPWrapper, visualize_foundation_feat, language_query
from einops import rearrange
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.projects.point_rend import add_pointrend_config
import pdb
import random
import kaolin
from skimage.metrics import structural_similarity as ssim
import cv2 as cv
import matplotlib.pyplot as plt


def filterlist(item):
    if isinstance(item, list):
        return item[0]
    return item


@DETECTORS.register_module()
class DistillNerfModelWrapper(BaseDetector):
    def __init__(
        self,
        # data  
        num_camera = 6,
        num_input_seq = 1,
        input_cam_temporal_index = 2,
        target_cam_temporal_idx = 2,
        force_same_seq = True,
        all_prev_frames = False,
        training_target_smart_sample = False,
        virtual_img_target_smart_sample = False,
        enable_upward_virtual_cam = True,

        # visualizations
        visualize_imgs = False,
        save_visualized_imgs = True,
        visualize_voxels = False,
        vis_save_directory = './vis/',

        # segmentation model
        seg_model_path = '',

        # foundation model feat
        render_foundation_model_feat = False,     # False, 'dino', 'clip'
        visualize_foundation_model_feat = False,
        language_query = False,

        # model
        model_yaml_path = None,
        pretrained_model_path = '',

        # appearance loss
        l1_loss_coef = 1.0,
        lpips_loss_coef = 1.0,
        rgb_clamp_0_1 = False,

        # weight entropy loss
        nerf_weight_entropy_loss_coef = 0.01,
        # mono_weight_entropy_loss_coef = 0.01,

        # opacity loss
        opacity_loss_coef = 0.1,

        # depth loss
        lidar_depth_loss_coef = 10.0,
        emernerf_depth_loss_coef = 10,
        max_depth_compensate = 5,
        enable_fine_depth_loss = False,
        # midas_depth_loss_coef = 0,

        # optional depth edge loss
        enable_edge_loss = False,
        enforce_edge_LoS_loss = False,
        edge_loss_aug_weight = 1,
        edge_loss_start_iter = 1000,
        edge_loss_distance = 56,

        # Optional line of sight terms
        enforce_LoS_loss = False,
        LoS_start_iter = 14000,
        LoS_fix_iter = 28000,
        LoS_epsilon_start = 6.0,
        LoS_epsilon_end = 2.5,
        LoS_decay_steps = 5000,
        LoS_decay_rate = 0.5,
        lidar_LoS_loss_coef = 0.1,
        emernerf_LoS_loss_coef = 0.1,
        
        # optional 3D density regularization loss
        use_emernerf_reg = False,
        ermernerf_reg_coef = 1.0,

        # optional object detectionfloss
        det_loss_coef = 1.0,
        det_head_3d = None,  # Detect 3D objects in the current frame
        dense_voxel_level = 7,
        voxel_feat_dim = 32,
        voxel_grid_shape = None,

        # novel view synthesis mode
        novel_view_mode = False,
        recoloring_img = False,

        # not used, only for mmdet3d compatibility
        train_cfg = None,
        test_cfg = None,

    ):
        super(DistillNerfModelWrapper, self).__init__()
        ''' data, model, loss, detector '''

        ''' data '''
        self.num_camera = num_camera
        self.num_input_seq = num_input_seq
        self.target_cam_temporal_idx = target_cam_temporal_idx
        self.input_cam_temporal_index = input_cam_temporal_index
        self.force_same_seq = force_same_seq                # for training, make the input and target images the same
        self.all_prev_frames = all_prev_frames              # for training, use all 3 previous frames as input and target images
        self.training_target_smart_sample = training_target_smart_sample
        self.virtual_img_target_smart_sample = virtual_img_target_smart_sample
        self.enable_upward_virtual_cam = enable_upward_virtual_cam
        self.novel_view_mode = novel_view_mode
        self.recoloring_img = recoloring_img

        ''' visualization '''
        self.visualize_imgs = visualize_imgs
        self.save_visualized_imgs = save_visualized_imgs
        self.visualize_voxels = visualize_voxels
        self.vis_save_directory = vis_save_directory

        
        ''' load segmentation model for sky mask generation '''
        self.seg_model = None
        if seg_model_path != '':
            self.seg_model = self.set_up_seg_models(seg_model_path)
            self.seg_model.eval()
            for param in self.seg_model.parameters():
                param.requires_grad = False

        ''' foundation model '''
        self.render_foundation_model_feat = render_foundation_model_feat
        self.visualize_foundation_model_feat = visualize_foundation_model_feat
        self.language_query = language_query


        ''' create/load the DistillNerf model '''
        model_cfg = OmegaConf.load(model_yaml_path)
        self.model = instantiate(model_cfg.model)
        if os.path.exists(pretrained_model_path):
            state_dict =  torch.load(pretrained_model_path, map_location=torch.device('cpu'))["state_dict"]
            new_state_dict = {}
            for k in state_dict:
                if k.startswith('model.'):
                    new_state_dict[k[6:]] = state_dict[k]
            missing,unexpected = self.model.load_state_dict(new_state_dict, strict=False)
            print("\n Loading pretrained main model, Missing: ", missing, "\n")
            print("\n Loading pretrained main model, Unexpected: ", unexpected, "\n")

        ''' 
            loss: appearance loss, weight entropy loss, opacity loss, depth loss (input/encoder/target depth, LiDAR/EmerNerf GT)
            optional loss (not used): depth edge enhancing loss, line of sight loss, 3D density EmerNerf regularization loss, detection loss
        '''
        # appearance loss
        self.l1_loss = RGBL1Loss(coef=l1_loss_coef)
        self.lpips_loss = LPIPSLoss(coef=lpips_loss_coef)
        self.rgb_clamp_0_1 = rgb_clamp_0_1

        # entropy loss
        self.target_weight_entropy_loss = NerfWeightEntropyLoss(coef=nerf_weight_entropy_loss_coef)
        self.coarse_weight_entropy_loss = NerfWeightEntropyLoss(coef=nerf_weight_entropy_loss_coef)

        # opacity loss
        self.coarse_opacity_loss = OpacityLoss(coef=opacity_loss_coef)
        self.fine_opacity_loss = OpacityLoss(coef=opacity_loss_coef)
        self.target_opacity_loss = OpacityLoss(coef=opacity_loss_coef)

        ''' depth loss '''
        # parameters
        self.sky_depth = model_cfg.geom_param.sky_depth
        self.min_depth = model_cfg.geom_param.min_depth
        self.max_depth = model_cfg.geom_param.max_depth
        self.inner_range = model_cfg.geom_param.inner_range
        self.lidar_range = model_cfg.geom_param.lidar_range
        self.depth_bounds = [self.min_depth, self.max_depth]
        self.max_depth_compensate = max_depth_compensate
        self.enable_fine_depth_loss = enable_fine_depth_loss

        # depth loss
        self.create_depth_loss_terms(lidar_depth_loss_coef, emernerf_depth_loss_coef)
            
        # depth edge enhancing loss
        self.enable_edge_loss = enable_edge_loss
        self.enforce_edge_LoS_loss = enforce_edge_LoS_loss
        self.edge_loss_aug_weight = edge_loss_aug_weight    
        self.edge_loss_start_iter = edge_loss_start_iter
        self.edge_loss_distance = edge_loss_distance

        # line of sight loss
        self.enforce_LoS_loss = enforce_LoS_loss
        self.LoS_start_iter = LoS_start_iter
        self.LoS_fix_iter = LoS_fix_iter
        self.LoS_epsilon_start = LoS_epsilon_start
        self.LoS_epsilon_end = LoS_epsilon_end
        self.LoS_decay_steps = LoS_decay_steps
        self.LoS_decay_rate = LoS_decay_rate
        self.coarse_LoS_loss_lidar = LineOfSightLoss(coef=lidar_LoS_loss_coef)
        self.coarse_LoS_loss_emernerf = LineOfSightLoss(coef=emernerf_LoS_loss_coef)
        self.fine_LoS_loss_lidar = LineOfSightLoss(coef=lidar_LoS_loss_coef)
        self.fine_LoS_loss_emernerf = LineOfSightLoss(coef=emernerf_LoS_loss_coef)
        self.target_LoS_loss_lidar = LineOfSightLoss(coef=lidar_LoS_loss_coef)
        self.target_LoS_loss_emernerf = LineOfSightLoss(coef=emernerf_LoS_loss_coef)

        # 3D density regularization loss
        self.use_emernerf_reg = use_emernerf_reg
        self.emernerf_reg_loss = MSELoss(coef=ermernerf_reg_coef)

        ''' detector '''
        self.dense_voxel_level = dense_voxel_level
        self.voxel_feat_dim = voxel_feat_dim
        self.voxel_grid_shape = voxel_grid_shape
        self.det_loss_coef = det_loss_coef
        # Multi-Task Decoder Heads
        if det_head_3d is not None:
            self.det_head_3d = builder.build_head(det_head_3d)


    def create_depth_loss_terms(self, lidar_depth_loss_coef, emernerf_depth_loss_coef):
        # input depth
        self.coarse_depth_loss_lidar = DepthClampLoss(
            coef=lidar_depth_loss_coef,
            depth_min=self.min_depth,
            depth_max=self.lidar_range
        )
        self.coarse_depth_loss_emernerf = EmernerfDepthClampLoss(
            coef=emernerf_depth_loss_coef,
            depth_min=self.min_depth,
            depth_max=self.max_depth,
            inner_range=self.inner_range,
        )

        # encoder depth
        self.fine_depth_loss_lidar = DepthClampLoss(
            coef=lidar_depth_loss_coef,
            depth_min=self.min_depth,
            depth_max=self.lidar_range
        )
        self.fine_depth_loss_emernerf = EmernerfDepthClampLoss(
            coef=emernerf_depth_loss_coef,
            depth_min=self.min_depth,
            depth_max=self.max_depth,
            inner_range=self.inner_range,
        )

        # target depth
        self.target_lidar_depth_loss = DepthClampLoss(
            coef=lidar_depth_loss_coef,
            depth_min=self.min_depth,
            depth_max=self.lidar_range
        )
        self.target_emernerf_depth_loss = EmernerfDepthClampLoss(
            coef=emernerf_depth_loss_coef,
            depth_min=self.min_depth,
            depth_max=self.max_depth,
            inner_range=self.inner_range,
        )

        # self.midas_depth_loss = MidasDepthClampLoss(
        #     coef=midas_depth_loss_coef,
        #     depth_min=self.min_depth,
        #     depth_max=self.max_depth,
        # )

    def set_local_rank(self, local_rank):
        if self.seg_model is not None:
            self.seg_model.local_rank = local_rank
        self.model.set_local_rank(local_rank)
        torch.cuda.empty_cache()

    @property
    def with_pts_bbox(self):
        """bool: Whether the model has a 3D box head."""
        return hasattr(self, "det_head_3d") and self.det_head_3d is not None

    def train_step(self, data, optimizer, iteration=None):
        """The iteration step during training.

        This method defines an iteration step during training, except for the
        back propagation and optimizer updating, which are done in an optimizer
        hook. Note that in some complicated cases or models, the whole process
        including back propagation and optimizer updating is also defined in
        this method, such as GAN.

        Args:
            data (dict): The output of dataloader.
            optimizer (:obj:`torch.optim.Optimizer` | dict): The optimizer of
                runner is passed to ``train_step()``. This argument is unused
                and reserved.

        Returns:
            dict: It should contain at least 3 keys: ``loss``, ``log_vars``, \
                ``num_samples``.

                - ``loss`` is a tensor for back propagation, which can be a
                  weighted sum of multiple losses.
                - ``log_vars`` contains all the variables to be sent to the
                  logger.
                - ``num_samples`` indicates the batch size (when the model is
                  DDP, it means the batch size on each GPU), which is used for
                  averaging the logs.
        """
        training_outputs = self(**data, force_same_seq=self.force_same_seq, all_prev_frames=self.all_prev_frames, \
            training_target_smart_sample=self.training_target_smart_sample, iteration=iteration, virtual_img_target_smart_sample=self.virtual_img_target_smart_sample)   

        losses = {k: v for k, v in training_outputs.items() if "loss" in k}
        other_outputs = {k: v for k, v in training_outputs.items() if "loss" not in k}

        loss, log_vars = self._parse_losses(losses)
        outputs = dict(
            loss=loss,
            log_vars=log_vars,
            num_samples=len(data["img_metas"]),
            **other_outputs,
        )

        return outputs
    
    def extract_feat(self, img) -> torch.Tensor:
        return self.img_backbone(*img)

    def select_function(self, x, temporal_idx, num_seq, num_cam, debug=False):
        # retreive the temporal index specified by "temporal_idx" for each camera, temporal_idx.shape [num_cam, num_temporal]
        # assume a batch size of 1
        x = x.reshape(num_seq, num_cam, *x.shape[2:])      # [5, num_cam, 3, h, w]
        res = []
        for temporal_ind in range(temporal_idx.shape[1]):
            res.append(x[temporal_idx[:, temporal_ind], torch.arange(num_cam)])
        res = torch.stack(res)
        return res.reshape(1, temporal_idx.shape[1]*num_cam, *x.shape[2:])

    def sampling_function(self, imgs, num_seq, num_cam, default_frame=None, all_prev_frames=False, exclude_vals=[], \
                          training_target_smart_sample=False, add_next_frame_for_eval=False, virtual_img_target_smart_sample=False):
        '''
            for each camera, select the temporal index - temporal_idx: [num_cam, num_temporal]
            assume a batch size of 1
            5 settings in total: 1 default setting and 4 special settings
        '''

        # 1. take the temporal index specified by default_frame
        temporal_idx = torch.tensor([[default_frame]]).repeat(num_cam, 1)
        assert [all_prev_frames, training_target_smart_sample, add_next_frame_for_eval, virtual_img_target_smart_sample].count(True) <= 1, \
            "There should be at most one special setting to be True"
        # 2. take all the previous frames as inputs or targets
        if all_prev_frames:
            temporal_idx = (torch.tensor([[0, 1, 2]])).repeat(num_cam, 1)
        # 3. select the current frame, and add a frame that is randomly sampled from previous/next frame
        elif training_target_smart_sample:
            # current frame
            current_temporal_idx = torch.tensor([[2] for i in range(num_cam)])
            # auxiliary sampled frame
            aux_temporal_idx = []
            for _ in range(num_cam): aux_temporal_idx.append(random.sample([1, 3], self.num_input_seq))
            aux_temporal_idx = torch.LongTensor(aux_temporal_idx)
            # combine together
            temporal_idx = torch.cat([aux_temporal_idx, current_temporal_idx], dim=1)
        # 4. select the current frame, and add the next frame for novel-view evaluation
        elif add_next_frame_for_eval:
             # current frame
            current_temporal_idx = torch.tensor([[2] for i in range(num_cam)])
            # next frame
            next_temporal_idx = torch.tensor([[3] for i in range(num_cam)])
            # combine together
            temporal_idx = torch.cat([next_temporal_idx, current_temporal_idx], dim=1)
        # 5. using virtual cameras
        elif virtual_img_target_smart_sample:
            virtual_cam_index = []
            if self.enable_upward_virtual_cam:
                for _ in range(num_cam): virtual_cam_index.append(random.sample([0,1,4], self.num_input_seq))
            else:
                for _ in range(num_cam): virtual_cam_index.append(random.sample([0,4], self.num_input_seq))
            virtual_cam_index = torch.LongTensor(virtual_cam_index)
            current_temporal_idx = torch.tensor([[2] for i in range(num_cam)])
            temporal_idx = torch.cat([virtual_cam_index, current_temporal_idx], dim=1)
        # 6. randomly sample, but exclude one frame (exclude_vals)
        # else:
        #     temporal_idx = []
        #     for _ in range(num_cam):
        #         temporal_idx.append(random.sample(list(set([i for i in range(0, num_seq)]) - set(exclude_vals)), self.num_input_seq))
        #     temporal_idx = torch.LongTensor(temporal_idx)

        return self.select_function(imgs, temporal_idx, num_seq, num_cam), temporal_idx
    

    def set_up_seg_models(self, seg_model_path):
        ''' load sementic segmentation network '''
        # Download pretrained model here:  https://dl.fbaipublicfiles.com/detectron2/PointRend/SemanticSegmentation/pointrend_semantic_R_101_FPN_1x_cityscapes/202576688/model_final_cf6ac1.pkl
        # clone the github repo to get the config file
        # Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well        

        # load default detectron2 config
        cfg = get_cfg()
        add_pointrend_config(cfg)
        # modify config
        best_model_config = './projects/DistillNeRF/configs/models/semseg/SemanticSegmentation/pointrend_semantic_R_101_FPN_1x_cityscapes.yaml'           # model config file
        cfg.merge_from_file(best_model_config)                                                                      # update config
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5                                                                 # set threshold for this model
        cfg.MODEL.WEIGHTS = os.path.join(seg_model_path, 'model_final_cf6ac1.pkl')                                  # local path to the model weight

        # load model
        predictor = DefaultPredictor(cfg)
        sem_seg_model = predictor.model

        return sem_seg_model

    @torch.no_grad()
    def run_semantic_segmentation(self, seg_img, visualize=False):
        self.seg_model.eval()
        cur_input = [{'image': i.contiguous(), 'height': i.shape[1], 'width': i.shape[2]} for i in seg_img[0]]

        output = self.seg_model(cur_input)
        sem_seg_masks = []
        for cur_output in output:
            classes = torch.argmax(cur_output['sem_seg'], dim=0) #.cpu().numpy()
            sem_seg_masks.append(classes)

        return sem_seg_masks

    def prepare_data_in_scene_format(self, imgs, num_cam, is_val=False, all_prev_frames=False, \
            training_target_smart_sample=False, add_next_frame_for_eval=False, virtual_img_target_smart_sample=False, **kwargs) -> Container:
        '''
            prepare the data into a certain format:
                1. collect data for input images
                2. collect data for target render images
                3. create camera pose trajectory for novel-view synthesis
                4. apply sky masks on the depth targets
        '''
        scene_data = Container()
        assert imgs.shape[0] == 1, 'assuming batch size of 1!'
        # temporaray check: post_trans is 0 so aspect ratio between orig img and resized img should match (1.77x)
        assert np.abs((imgs.shape[-1] / imgs.shape[-2]) - 1.77) < 0.02 or np.abs((imgs.shape[-1] / imgs.shape[-2]) - 1.5) < 0.02
        #                   nuscenes                                                    waymo                         
        if virtual_img_target_smart_sample: 
            if is_val:      # format issues
                img_source = torch.cat([torch.stack(kwargs['virtual_img'][0][:num_cam*2], dim=0).permute(1, 0, 4, 2, 3), imgs[:, num_cam*2:num_cam*4], torch.stack(kwargs['virtual_img'][0][num_cam*4:], dim=0).permute(1, 0, 4, 2, 3)], dim=1)
            else:
                img_source = torch.cat([torch.stack(kwargs['virtual_img'][:num_cam*2], dim=0).permute(1, 0, 4, 2, 3), imgs[:, num_cam*2:num_cam*4], torch.stack(kwargs['virtual_img'][num_cam*4:], dim=0).permute(1, 0, 4, 2, 3)], dim=1)
            # self.visualize_virtual_cam(img_source[:,:,[2,1,0]], kwargs["gt_depth_img"][0], kwargs["emernerf_depth_img"][0].unsqueeze(2), 1)
            # self.visualize_virtual_cam(imgs[:,:,[2,1,0]], kwargs["gt_depth_img"][0], kwargs["emernerf_depth_img"][0].unsqueeze(2), 2)
        else:
            img_source = imgs

        ''' 1. data for input images '''
        num_seq = img_source.shape[1] // num_cam
        # specify the temporal frame index for each camera
        input_imgs, input_inds = self.sampling_function(img_source, num_seq, num_cam, \
                                    default_frame=self.input_cam_temporal_index, all_prev_frames=all_prev_frames)
        # collect data
        input_imgs = torch.flip(input_imgs, dims=(2,)) # BGR to RGB
        scene_data.set("imgs", input_imgs / 255.0)
        scene_data.set("masks", (input_imgs*0)[:,:,0]) # all pixels in e.g. nuscenes is valid
        # extrinsics and intrinsics
        scene_data.set("rots", self.select_function(filterlist(kwargs["extrinsic"]), input_inds, num_seq, num_cam, debug=False)[..., :3, :3])
        scene_data.set("trans", self.select_function(filterlist(kwargs["extrinsic"]), input_inds, num_seq, num_cam)[..., :3, 3])
        scene_data.set("intrins", self.select_function(filterlist(kwargs["intrinsic"]), input_inds, num_seq, num_cam))
        # resolution scaling
        scene_data.set("post_rots", self.select_function(filterlist(kwargs["aug_transform"]), input_inds, num_seq, num_cam)[..., :3, :3])
        scene_data.set("post_trans", self.select_function(filterlist(kwargs["aug_transform"]), input_inds, num_seq, num_cam)[..., :3, 3])
        # lidar and EmerNerf depth
        scene_data.set("depths", self.select_function(filterlist(kwargs["gt_depth_img"]), input_inds, num_seq, num_cam))
        scene_data.set("emernerf_depth_img", self.select_function(filterlist(kwargs["emernerf_depth_img"]), input_inds, num_seq, num_cam))
        scene_data.set("emernerf_sky_mask", self.select_function(filterlist(kwargs["emernerf_sky_mask"]), input_inds, num_seq, num_cam))
        scene_data.set("cam_classes", torch.ones(input_imgs.shape[0], input_imgs.shape[1]).to(input_imgs.device).long())
        # lidar depths (different preprocessing)
        if "lidar_depths" in kwargs.keys():
            scene_data.set("lidar_depths", self.select_function(filterlist(kwargs["lidar_depths"]), input_inds, num_seq, num_cam))
            scene_data.set("lidar_depth_loc2ds", self.select_function(filterlist(kwargs["lidar_depth_loc2ds"]), input_inds, num_seq, num_cam))
            scene_data.set("lidar_depth_masks", self.select_function(filterlist(kwargs["lidar_depth_masks"]), input_inds, num_seq, num_cam))
        # foundation model feature image
        if self.render_foundation_model_feat: scene_data.set("fm_feat", self.select_function(filterlist(kwargs["feat_img"]), input_inds, num_seq, num_cam))


        ''' 2. data for target render images '''
        # specify the temporal frame index for each camera
        target_imgs, target_inds = self.sampling_function(img_source, num_seq, num_cam, default_frame=self.input_cam_temporal_index, \
                all_prev_frames=all_prev_frames, training_target_smart_sample=training_target_smart_sample, add_next_frame_for_eval=add_next_frame_for_eval, \
                virtual_img_target_smart_sample=virtual_img_target_smart_sample)
        # collect data
        target_imgs = torch.flip(target_imgs, dims=(2,))  # BGR to RGB
        scene_data.set("target_imgs", target_imgs / 255.0)
        scene_data.set("target_masks", (target_imgs*0)[:,:,0]) # all pixels in e.g. nuscenes is valid
        # extrinsics and intrinsics
        scene_data.set("target_rots", self.select_function(filterlist(kwargs["extrinsic"]), target_inds, num_seq, num_cam)[..., :3, :3])
        scene_data.set("target_trans", self.select_function(filterlist(kwargs["extrinsic"]), target_inds, num_seq, num_cam)[..., :3, 3])
        scene_data.set("target_intrins", self.select_function(filterlist(kwargs["intrinsic"]), target_inds, num_seq, num_cam))
        # resolution scaling
        scene_data.set("target_post_rots", self.select_function(filterlist(kwargs["aug_transform"]), target_inds, num_seq, num_cam)[..., :3, :3])
        scene_data.set("target_post_trans", self.select_function(filterlist(kwargs["aug_transform"]), target_inds, num_seq, num_cam)[..., :3, 3])
        # lidar and EmerNerf depth
        scene_data.set("target_depth_imgs", self.select_function(filterlist(kwargs["gt_depth_img"]), target_inds, num_seq, num_cam))
        scene_data.set("target_emernerf_depth_img", self.select_function(filterlist(kwargs["emernerf_depth_img"]), target_inds, num_seq, num_cam))
        scene_data.set("target_emernerf_sky_mask", self.select_function(filterlist(kwargs["emernerf_sky_mask"]),  target_inds, num_seq, num_cam))
        scene_data.set("target_cam_classes", torch.ones(target_imgs.shape[0], target_imgs.shape[1]).to(target_imgs.device).long())
        # lidar depths (different preprocessing)
        if "lidar_depths" in kwargs.keys():
            scene_data.set("target_lidar_depths", self.select_function(filterlist(kwargs["lidar_depths"]), target_inds, num_seq, num_cam))
            scene_data.set("target_lidar_depth_loc2ds", self.select_function(filterlist(kwargs["lidar_depth_loc2ds"]), target_inds, num_seq, num_cam))
            scene_data.set("target_lidar_depth_masks", self.select_function(filterlist(kwargs["lidar_depth_masks"]), target_inds, num_seq, num_cam))
        # foundation model feature image
        if self.render_foundation_model_feat: scene_data.set("target_fm_feat", self.select_function(filterlist(kwargs["feat_img"]), target_inds, num_seq, num_cam))
        

        ''' 
            3. create novel view synthesis camera poses, according to the specified novel view traj
            only extrinsics are indeed modified, others are just repearted to match the traj_num and run through the code
        '''
        if "novel_view_traj" in kwargs.keys():
            self.novel_view_mode = True
            novel_view_traj = kwargs["novel_view_traj"]
            traj_num = len(novel_view_traj)
            novel_view_traj = [torch.from_numpy(novel_view_traj[i]).to(target_imgs.device).float() for i in range(traj_num)]
            # simply repeat the image
            target_imgs = torch.repeat_interleave(target_imgs, traj_num, dim=1)
            scene_data.set("target_imgs", target_imgs / 255.0)
            scene_data.set("target_masks", (target_imgs*0)[:,:,0]) # all pixels in e.g. nuscenes is valid
            # change extrinsic rotation matrix
            target_rots = self.select_function(filterlist(kwargs["extrinsic"]), target_inds, num_seq, num_cam)[..., :3, :3]
            target_rots = torch.cat([novel_view_traj[i][:3, :3] @ target_rots for i in range(traj_num)], dim=1)
            scene_data.set("target_rots", target_rots)
            # change extrinsic translation matrix
            target_trans = self.select_function(filterlist(kwargs["extrinsic"]), target_inds, num_seq, num_cam)[..., :3, 3]
            target_trans = torch.cat([novel_view_traj[i][:3, 3] + target_trans for i in range(traj_num)], 1)
            scene_data.set("target_trans", target_trans)
            # simply repeat the intrinsics
            scene_data.set("target_intrins", torch.tile(self.select_function(filterlist(kwargs["intrinsic"]), target_inds, num_seq, num_cam), (1,traj_num,1,1)))
            # simply repeat the resolution scaling
            scene_data.set("target_post_rots", torch.tile(self.select_function(filterlist(kwargs["aug_transform"]), target_inds, num_seq, num_cam)[..., :3, :3], (1,traj_num,1,1)))
            scene_data.set("target_post_trans", torch.tile(self.select_function(filterlist(kwargs["aug_transform"]), target_inds, num_seq, num_cam)[..., :3, 3], (1,traj_num,1)))
            # simply repeat the lidar and EmerNerf depth
            scene_data.set("target_depth_imgs", torch.tile(self.select_function(filterlist(kwargs["gt_depth_img"]), target_inds, num_seq, num_cam), (1,traj_num,1,1,1)))
            scene_data.set("target_cam_classes", torch.ones(target_imgs.shape[0], target_imgs.shape[1]).to(target_imgs.device).long())
            scene_data.set("target_emernerf_depth_img", torch.tile(self.select_function(filterlist(kwargs["emernerf_depth_img"]), target_inds, num_seq, num_cam), (1,traj_num,1,1)))
            scene_data.set("target_emernerf_sky_mask", torch.tile(self.select_function(filterlist(kwargs["emernerf_sky_mask"]),  target_inds, num_seq, num_cam), (1,traj_num,1,1)))
            # simply repeat the lidar depths (different preprocessing)
            if "lidar_depths" in kwargs.keys():
                scene_data.set("target_lidar_depths", torch.tile(self.select_function(filterlist(kwargs["lidar_depths"]), target_inds, num_seq, num_cam), (1,traj_num,1)))
                scene_data.set("target_lidar_depth_loc2ds", torch.tile(self.select_function(filterlist(kwargs["lidar_depth_loc2ds"]), target_inds, num_seq, num_cam), (1,traj_num,1)))
                scene_data.set("target_lidar_depth_masks", torch.tile(self.select_function(filterlist(kwargs["lidar_depth_masks"]), target_inds, num_seq, num_cam), (1,traj_num,1)))


        ''' 4. apply sky masks to specify the depth of the sky region '''
        if self.seg_model is not None:
            ''' input lidar depth '''
            input_depth = scene_data.get("depths")  
            # run segmentation to get sky regions
            sem_seg_masks = self.run_semantic_segmentation(torch.flip(input_imgs, dims=(2,)))
            sem_seg_masks = torch.stack(sem_seg_masks).unsqueeze(1)
            sem_seg_masks = F.interpolate(sem_seg_masks.float(), (input_depth.shape[-2], input_depth.shape[-1]), mode='nearest')
            sky_region = sem_seg_masks == 10
            # specify the sky depth
            empty_input_depth = input_depth.isnan()
            fill_sky = torch.logical_and(empty_input_depth, sky_region.unsqueeze(0))
            input_depth[fill_sky] = self.sky_depth - self.max_depth_compensate
            scene_data.set("depths", torch.clamp(input_depth, self.min_depth, self.max_depth - self.max_depth_compensate))

            ''' input EmerNerf depth '''
            emernerf_depth_img = scene_data.get("emernerf_depth_img")
            emernerf_sky_mask = scene_data.get("emernerf_sky_mask")
            emernerf_sky_mask = F.interpolate(emernerf_sky_mask.float(), (emernerf_depth_img.shape[-2], emernerf_depth_img.shape[-1]), mode='nearest')
            sky_region = emernerf_sky_mask == 1
            emernerf_depth_img[sky_region] = self.sky_depth - self.max_depth_compensate   # this will change scene.target_depth_imgs
            scene_data.set("emernerf_depth_img", torch.clamp(emernerf_depth_img, self.min_depth, self.max_depth - self.max_depth_compensate))

            ''' target lidar depth '''
            target_depth = scene_data.get("target_depth_imgs")  # gt depth
            # run segmentation to get sky regions
            sem_seg_masks = self.run_semantic_segmentation(torch.flip(target_imgs, dims=(2,)))
            sem_seg_masks = torch.stack(sem_seg_masks).unsqueeze(1)
            sem_seg_masks = F.interpolate(sem_seg_masks.float(), (target_depth.shape[-2], target_depth.shape[-1]), mode='nearest')
            sky_region = sem_seg_masks == 10
            # specify the sky depth
            empty_target_depth = target_depth.isnan()
            fill_sky = torch.logical_and(empty_target_depth, sky_region.unsqueeze(0))
            target_depth[fill_sky] = self.sky_depth - self.max_depth_compensate
            scene_data.set("target_depth_imgs", torch.clamp(target_depth, self.min_depth, self.max_depth - self.max_depth_compensate))

            ''' target EmerNerf depth '''
            target_emernerf_depth_img = scene_data.get("target_emernerf_depth_img")
            if not virtual_img_target_smart_sample:     
                # use sky masks come from EmerNeRF
                target_emernerf_sky_mask = scene_data.get("target_emernerf_sky_mask")
                target_emernerf_sky_mask = F.interpolate(target_emernerf_sky_mask.float(), (target_emernerf_depth_img.shape[-2], target_emernerf_depth_img.shape[-1]), mode='nearest')
                sky_region = target_emernerf_sky_mask == 1
                target_emernerf_depth_img[sky_region] = self.sky_depth - self.max_depth_compensate   # this will change scene.target_depth_imgs
                scene_data.set("target_emernerf_depth_img", torch.clamp(target_emernerf_depth_img, self.min_depth, self.max_depth - self.max_depth_compensate))
            else:
                # for virtual cameras, the sky masks from EmerNeRF do not apply, need to generate masks online
                sem_seg_masks = F.interpolate(sem_seg_masks.float(), (target_emernerf_depth_img.shape[-2], target_emernerf_depth_img.shape[-1]), mode='nearest')
                sky_region = sem_seg_masks == 10
                target_emernerf_depth_img[sky_region.permute(1,0,2,3)] = self.sky_depth - self.max_depth_compensate   # this will change scene.target_depth_imgs
                # for original camears, still use the sky masks from EmerNeRF
                target_emernerf_depth_img = scene_data.get("target_emernerf_depth_img")  # gt depth
                target_emernerf_sky_mask = scene_data.get("target_emernerf_sky_mask")  # gt depth
                sky_region = target_emernerf_sky_mask == 1
                target_emernerf_depth_img[:, num_cam:][sky_region[:, num_cam:]] = self.sky_depth - self.max_depth_compensate
                # set the target EmerNeRF depth
                scene_data.set("target_emernerf_depth_img", torch.clamp(target_emernerf_depth_img, self.min_depth, self.max_depth - self.max_depth_compensate))

        return scene_data

    def detection_model(self, intermediates, is_val, kwargs, img_metas, outputs):
        ''' This code is outdated and not used'''
        point_hierarchies = intermediates.get(f"point_hierarchy")
        pyramids = intermediates.get(f"octree_pyramid{self.dense_voxel_level}") # 7
        features = intermediates.get(f"octree_feats{self.dense_voxel_level}")[:, -self.voxel_feat_dim:] # 32

        # make it dense and merge across z axis and get indices of occupied voxels
        pre_features = kaolin.ops.spc.to_dense(point_hierarchies, pyramids.unsqueeze(0), features.float(), level=self.dense_voxel_level).to(features)
        pre_features = rearrange(pre_features, 'b c x y (z d) -> b d c z y x', d=2).mean(1)
        features = pre_features[:, :, :self.voxel_grid_shape[2]]
        voxel_inds = torch.nonzero(features.abs().sum(1) > 0)
        if voxel_inds.sum() == 0:
            features = pre_features[:, :, self.voxel_grid_shape[2]:self.voxel_grid_shape[2]*2]
            voxel_inds = torch.nonzero(features.abs().sum(1) > 0)

        voxel_inds = torch.flip(voxel_inds, dims=[1]) # det_head_3d expects [x,y,z,b]
        features = features[voxel_inds[:,3], :, voxel_inds[:,2], voxel_inds[:,1], voxel_inds[:,0]]
        with warnings.catch_warnings():
            # Doing this because there are a lot of floor division warnings,
            # which is fine since none of our dividing elements are negative.
            warnings.simplefilter("ignore", UserWarning)
            if not is_val:
                det_3d_losses = self.det_head_3d(
                    voxel_features=features,
                    feature_coors=voxel_inds,
                    gt_bboxes_3d=kwargs["gt_bboxes_3d"],
                    gt_labels_3d=kwargs["gt_labels_3d"],
                )
                det_3d_losses['task0.loss_heatmap'] = self.det_loss_coef * det_3d_losses['task0.loss_heatmap']
                det_3d_losses['task0.loss_bbox'] = self.det_loss_coef * det_3d_losses['task0.loss_bbox']
                outputs.update(det_3d_losses)
            else:
                det_3d_out = self.det_head_3d.simple_test(
                    voxel_features=features,
                    feature_coors=voxel_inds,
                    img_metas=img_metas,
                )
                outputs.update(det_3d_out[0])

    def emernerf_regularization(self, kwargs, outputs):
        ''' This code is outdated and not used'''
        # TODO: change the query to parameterized space
        # TODO: add regularization on rgb/density directly
        static_voxel_feat, static_voxel_coord = kwargs['static_voxel'][0][0], kwargs['static_voxel'][0][1]
        prev_dynamic_voxel_feat, cur_dynamic_voxel_feat = kwargs['dynamic_voxel'][0][0][0], kwargs['dynamic_voxel'][0][0][1]
        prev_dynamic_voxel_coord, cur_dynamic_voxel_coord = kwargs['dynamic_voxel'][0][1][0], kwargs['dynamic_voxel'][0][1][1]
        static_voxel_density, static_voxel_coord = kwargs['static_densities'][0][0], kwargs['static_densities'][0][1]
        prev_dynamic_voxel_density, cur_dynamic_voxel_density = kwargs['dynamic_densities'][0][0][0], kwargs['dynamic_densities'][0][0][1]
        prev_dynamic_voxel_coord, cur_dynamic_voxel_coord = kwargs['dynamic_densities'][0][1][0], kwargs['dynamic_densities'][0][1][1]

        emernerf_voxel_feat = torch.concat((static_voxel_feat, cur_dynamic_voxel_feat))
        emernerf_voxel_coord = torch.unsqueeze(torch.concat((static_voxel_coord, cur_dynamic_voxel_coord)), 0)
        emernerf_voxel_density = torch.concat((static_voxel_density, cur_dynamic_voxel_density))

        emernerf_voxel_range = [[-50, 50], [-50, 50], [-5, 3]]
        emernerf_voxel_size = 0.5
        emernerf_voxel_coord = emernerf_voxel_coord * emernerf_voxel_size + torch.tensor([emernerf_voxel_range[0][0], emernerf_voxel_range[1][0], emernerf_voxel_range[2][0]]).to(emernerf_voxel_coord.device)

        queried_octree_feat = self.model.projector.volume_renderer.sample_from_octree(grid=None, options = None, coordinates = emernerf_voxel_coord, intermediates=intermediates).squeeze(0)
        # concat: [octree7_feat, octree9_feat, density]
        emernerf_voxel = torch.cat([emernerf_voxel_feat[:,:64], emernerf_voxel_density], dim=1)
        outputs.update(self.emernerf_reg_loss(queried_octree_feat, emernerf_voxel, name='emernerf_reg_loss'))

    def compute_depth_errors(self, gt_depth_orin, pred_depth_orin, depth_min=1e-3, depth_max=80):
        """Computation of error metrics between predicted and ground truth depths
        """
        gt_depth = gt_depth_orin.detach().cpu().numpy()
        pred_depth = pred_depth_orin.detach().cpu().numpy()
        mask = np.logical_and(gt_depth > depth_min, gt_depth < depth_max)

        pred_depth = pred_depth[mask]
        gt_depth = gt_depth[mask]

        pred_depth = np.clip(pred_depth, depth_min, depth_max)

        thresh = np.maximum((gt_depth / pred_depth), (pred_depth / gt_depth))

        a1 = (thresh < 1.25     ).mean()
        a2 = (thresh < 1.25 ** 2).mean()
        a3 = (thresh < 1.25 ** 3).mean()

        rmse = (gt_depth - pred_depth) ** 2
        rmse = np.sqrt(rmse.mean())

        rmse_log = (np.log(gt_depth) - np.log(pred_depth)) ** 2
        rmse_log = np.sqrt(rmse_log.mean())

        abs_rel = np.mean(np.abs(gt_depth - pred_depth) / gt_depth)

        sq_rel = np.mean(((gt_depth - pred_depth) ** 2) / gt_depth)

        # print(abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3)

        return abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3
    
    def visualize_img_depth(self, scene_data, intermediates, img_metas, add_next_frame_for_eval, novel_view):
        def visualize_depth(value, lo=4, hi=120, curve_fn=lambda x: -np.log(x + 1e-6)):
            import matplotlib.cm as cm
            color_map = cm.get_cmap("turbo")
            value, lo, hi = [curve_fn(x) for x in [value, lo, hi]]
            value = np.nan_to_num(
                    np.clip((value - np.minimum(lo, hi)) / np.abs(hi - lo), 0, 1)
                )
            colorized = color_map(value)[..., :3]
            return colorized

        ''' organize images '''
        to_numpy = lambda x: x.cpu().detach().numpy()
        input_img = to_numpy(scene_data.imgs[0].permute(0,2,3,1))
        if add_next_frame_for_eval: 
            recon_img = to_numpy(intermediates.recons[self.num_camera:].permute(0,2,3,1))
            target_pred_depth_image = to_numpy(intermediates.target_pred_depth_image[self.num_camera:, 0])
        else:
            recon_img = to_numpy(intermediates.recons.permute(0,2,3,1))
            target_pred_depth_image = to_numpy(intermediates.target_pred_depth_image[:, 0])
        coarse_depth = to_numpy(intermediates.coarse_depth)
        fine_depth = to_numpy(intermediates.fine_depth)
        h, w = intermediates.target_pred_depth_image.shape[-2:]
        emernerf_depth_img = to_numpy(F.interpolate(scene_data.emernerf_depth_img, (h, w), mode='bilinear')[0])
        lidar_depth_img = to_numpy(scene_data.depths.squeeze(2)[0])
        emernerf_sky_mask = to_numpy(scene_data.emernerf_sky_mask)[0]

        '''' for rebuttal '''
        # import matplotlib.pyplot as plt
        # # import pdb; pdb.set_trace()
        # gt_save_directory = "./rebuttal/gt_imgs/"
        # os.makedirs(gt_save_directory, exist_ok=True)
        # for i, img in enumerate(input_img):
        #     files_in_directory = os.listdir(gt_save_directory)
        #     image_files = [file for file in files_in_directory if file.endswith(('.png', '.jpg', '.jpeg'))]
        #     image_count = len(image_files)

        #     # Convert image to the correct format if necessary (e.g., from tensor to numpy array)
        #     img_to_save = img.numpy() if hasattr(img, 'numpy') else img
        #     # Save the image
        #     # print(img_to_save.max(), img_to_save.min())
        #     from scipy.ndimage import zoom
        #     img_to_save = zoom(img_to_save, (4, 4, 1))
        #     plt.imsave(gt_save_directory+f'image_{image_count}.png', np.clip(img_to_save, 0, 1))
        
        # render_save_directory = "./rebuttal/rendered_imgs/"
        # os.makedirs(render_save_directory, exist_ok=True)
        # for i, img in enumerate(recon_img):
        #     files_in_directory = os.listdir(render_save_directory)
        #     image_files = [file for file in files_in_directory if file.endswith(('.png', '.jpg', '.jpeg'))]
        #     image_count = len(image_files)

        #     # Convert image to the correct format if necessary (e.g., from tensor to numpy array)
        #     img_to_save = img.numpy() if hasattr(img, 'numpy') else img
        #     # Save the image
        #     # print(img_to_save.max(), img_to_save.min())
        #     from scipy.ndimage import zoom
        #     img_to_save = zoom(img_to_save, (4, 4, 1))
        #     plt.imsave(render_save_directory+f'image_{image_count}.png', np.clip(img_to_save, 0, 1))

        # print("psnr: ", (-10 * torch.log10(F.mse_loss(torch.from_numpy(input_img), torch.from_numpy(recon_img)))).item())
        # return None
        '''' for rebuttal '''

        ''' for waymo debugging '''
        # import matplotlib.pyplot as plt
        # fig, axes = plt.subplots(nrows=6, ncols=7)
        # axes[0, 0].set_title("Input Images")
        # axes[0, 1].set_title("Rendered \n Images")
        # axes[0, 2].set_title("Pred \n coarse depth")
        # axes[0, 3].set_title("Pred \n fine depth")
        # axes[0, 4].set_title("Pred \n target depth")
        # axes[0, 5].set_title("Depth \n Lidar GT")
        # axes[0, 6].set_title("Depth \n EmerNeRF Sky Mask")
        # cam_num = intermediates.coarse_depth.shape[0]
        # for cam in range(cam_num):
        #     axes[cam, 0].imshow(input_img[cam])
        #     axes[cam, 1].imshow(recon_img[cam].astype(np.float32))
        #     axes[cam, 2].imshow(visualize_depth(coarse_depth[cam]))
        #     # axes[cam, 2].imshow(fine_depth[cam], cmap='turbo')
        #     axes[cam, 3].imshow(visualize_depth(fine_depth[cam]))
        #     axes[cam, 4].imshow(visualize_depth(target_pred_depth_image[cam]))
        #     # axes[cam, 5].imshow(visualize_depth(emernerf_depth_img[cam]))
        #     # axes[cam, 5].imshow(emernerf_depth_img[cam], cmap='turbo')
        #     # axes[cam, 6].imshow(np.clip(target_pred_depth_image[cam], self.min_depth, self.max_depth) - np.clip(emernerf_depth_img[cam], self.min_depth, self.max_depth))
        #     # axes[cam, 7].imshow(visualize_depth(lidar_depth_img[cam]))
        #     axes[cam, 5].imshow(lidar_depth_img[cam], cmap='turbo')
        #     axes[cam, 6].imshow(emernerf_sky_mask[cam], cmap='turbo')
        #     # axes[cam, 8].imshow(np.clip(target_pred_depth_image[cam], self.min_depth, self.max_depth) - np.clip(lidar_depth_img[cam], self.min_depth, self.max_depth))
        #     # axes[cam, 9].imshow(emernerf_depth_img[cam] - lidar_depth_img[cam])
        #     # import pdb; pdb.set_trace()
        # plt.show(block=True)
        # fig.suptitle(f"flip={img_metas[0]['flip']}")
        # plt.close()
        ''' for waymo debugging '''

        if novel_view:
            N_cam = self.num_camera
            N_novel_frame = int(recon_img.shape[0] / N_cam)
            for i in range(N_novel_frame):
                save_image_horizontally(recon_img[i*N_cam:(i+1)*N_cam], self.vis_save_directory, 'novel_rgb')
                save_image_horizontally(visualize_depth(target_pred_depth_image[i*N_cam:(i+1)*N_cam]), self.vis_save_directory, name='novel_depth')

        else:
            ''' visualize rgb and depth '''
            import matplotlib.pyplot as plt
            fig, axes = plt.subplots(nrows=6, ncols=11)
            axes[0, 0].set_title("Input Images")
            axes[0, 1].set_title("Rendered \n Images")
            axes[0, 2].set_title("Pred \n coarse depth")
            axes[0, 3].set_title("Pred \n fine depth")
            axes[0, 4].set_title("Pred \n target depth")
            axes[0, 5].set_title("EmerNerf \n Depth GT")
            axes[0, 6].set_title("Target \n EmerNerf error ")
            axes[0, 7].set_title("Depth \n Lidar GT")
            axes[0, 8].set_title("Target \n Lidar Error")
            axes[0, 9].set_title("EmerNerf \n Lidar Error")
            axes[0, 10].set_title("EmerNerf \n Sky Mask")
            cam_num = intermediates.coarse_depth.shape[0]
            for cam in range(cam_num):
                axes[cam, 0].imshow(input_img[cam])
                axes[cam, 1].imshow(recon_img[cam].astype(np.float32))
                axes[cam, 2].imshow(visualize_depth(coarse_depth[cam]))
                # axes[cam, 2].imshow(fine_depth[cam], cmap='turbo')
                axes[cam, 3].imshow(visualize_depth(fine_depth[cam]))
                axes[cam, 4].imshow(visualize_depth(target_pred_depth_image[cam]))
                axes[cam, 5].imshow(visualize_depth(emernerf_depth_img[cam]))
                # axes[cam, 5].imshow(emernerf_depth_img[cam], cmap='turbo')
                axes[cam, 6].imshow(np.clip(target_pred_depth_image[cam], self.min_depth, self.max_depth) - np.clip(emernerf_depth_img[cam], self.min_depth, self.max_depth))
                # axes[cam, 7].imshow(visualize_depth(lidar_depth_img[cam]))
                axes[cam, 7].imshow(lidar_depth_img[cam], cmap='turbo')
                axes[cam, 8].imshow(np.clip(target_pred_depth_image[cam], self.min_depth, self.max_depth) - np.clip(lidar_depth_img[cam], self.min_depth, self.max_depth))
                axes[cam, 9].imshow(emernerf_depth_img[cam] - lidar_depth_img[cam])
                axes[cam, 10].imshow(emernerf_sky_mask[cam], cmap='turbo')
                # import pdb; pdb.set_trace()
            plt.show(block=True)
            plt.close()
            fig.suptitle(f"flip={img_metas[0]['flip']}")

            if self.save_visualized_imgs:
                save_image_horizontally(input_img, self.vis_save_directory, name='gt_rgb')
                save_image_horizontally(recon_img, self.vis_save_directory, name='recon_rgb')
                save_image_horizontally(visualize_depth(coarse_depth), self.vis_save_directory, name='coarse_depth_pred')
                save_image_horizontally(visualize_depth(fine_depth), self.vis_save_directory, name='fine_depth_pred')
                save_image_horizontally(visualize_depth(target_pred_depth_image), self.vis_save_directory, name='render_depth_pred')
                save_image_horizontally(visualize_depth(lidar_depth_img), self.vis_save_directory, name='gt_lidar_depth')
                save_image_horizontally(visualize_depth(emernerf_depth_img), self.vis_save_directory, name='gt_emernerf_depth')


            ''' visualize depth edge '''
            # import cv2 as cv
            # import matplotlib.pyplot as plt
            # fig, axes = plt.subplots(nrows=6, ncols=7)
            # axes[0, 0].set_title("Images")
            # axes[0, 1].set_title("Image edge")
            # axes[0, 2].set_title("EmerNerf \n Depth GT")
            # axes[0, 3].set_title("Pred mono \n depth")
            # axes[0, 4].set_title("Pred mono \n depth edge")
            # axes[0, 5].set_title("Pred encoder \n depth edge")
            # axes[0, 6].set_title("Pred target \n depth edge")
            # for cam in range(cam_num):
            #     image_edge = cv.Canny((scene_data.imgs[0, cam].permute(1, 2, 0).cpu().numpy()*255).astype(np.uint8), 100, 200)
            #     mono_depth_edge = cv.Canny((visualize_depth(coarse_depth[cam])*255).astype(np.uint8), 150, 200)
            #     encoder_depth_edge = cv.Canny((visualize_depth(fine_depth[cam])*255).astype(np.uint8), 150, 200)
            #     target_depth_edge = cv.Canny((visualize_depth(target_pred_depth_image[cam])*255).astype(np.uint8), 150, 200)

            #     axes[cam, 0].imshow(scene_data.imgs[0, cam].permute(1, 2, 0).cpu().numpy())
            #     axes[cam, 1].imshow(image_edge)
            #     axes[cam, 2].imshow(scene_data.emernerf_depth_img[0, cam].cpu().detach().numpy(), cmap='turbo')
            #     axes[cam, 3].imshow(coarse_depth[cam], cmap='turbo')
            #     axes[cam, 4].imshow(mono_depth_edge)
            #     axes[cam, 5].imshow(encoder_depth_edge)
            #     axes[cam, 6].imshow(target_depth_edge)
            
            # # plt.show(block=True)
            # plt.close()

    def visualize_virtual_cam(self, rgb_img, lidar_depth_img, emernerf_depth_img, frame_id):
        def visualize_depth(value, lo=4, hi=120, curve_fn=lambda x: -np.log(x + 1e-6)):
            import matplotlib.cm as cm
            color_map = cm.get_cmap("turbo")
            value, lo, hi = [curve_fn(x) for x in [value, lo, hi]]
            value = np.nan_to_num(
                    np.clip((value - np.minimum(lo, hi)) / np.abs(hi - lo), 0, 1)
                )
            colorized = color_map(value)[..., :3]
            return colorized
        
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(nrows=6, ncols=6)
        axes[0, 0].set_title("Input Images")
        axes[0, 1].set_title("EmerNerf Depth GT")
        axes[0, 2].set_title("Lidar Depth GT")
        for cam in range(6):
            axes[cam, 0].imshow(rgb_img[0, frame_id*6+cam].permute(1, 2, 0).cpu())
            axes[cam, 1].imshow(visualize_depth(emernerf_depth_img[0, frame_id*6+cam].permute(1, 2, 0).cpu().numpy())[:,:,0])
            axes[cam, 2].imshow(visualize_depth(lidar_depth_img[0, frame_id*6+cam].permute(1, 2, 0).cpu().numpy())[:,:,0])

        # fig.suptitle(f"flip={img_metas[0]['flip']}")
        plt.show()

    def sample_from_octree_wrapper(self, intermediates, scene_data):
        def create_voxel_coordinate():
            # Grid dimensions and voxel sizes
            bound = [[-40, 40], [-40, 40], [-5, 3]]
            # bound = [[-120, 120], [-120, 120], [-5, 20]]
            # bound = [[-100, 100], [-100, 100], [-5, 20]]
            # bound = [[-62.5, 62.5], [-62.5, 62.5], [-5, 20]]
            # bound = [[-50, 50], [-50, 50], [-5, 6.4]] # 15
            # bound = [[-10, 10], [-10, 10], [-5, 10]]

            voxel_size = [0.5, 0.5, 0.5]
            voxel_num = [(bound[0][1] - bound[0][0]) / voxel_size[0] + 1, 
                        (bound[1][1] - bound[1][0]) / voxel_size[1] + 1, 
                        (bound[2][1] - bound[2][0]) / voxel_size[2] + 1, 
            ]

            # Create a grid of coordinates
            x = torch.linspace(bound[0][0], bound[0][1], int(voxel_num[0]))
            y = torch.linspace(bound[1][0], bound[1][1], int(voxel_num[1]))
            z = torch.linspace(bound[2][0], bound[2][1], int(voxel_num[2]))

            # Create a 3D grid of coordinates
            x, y, z = torch.meshgrid(x, y, z, indexing='ij')

            # Stack the coordinates into a single tensor
            voxel_coordinates = torch.stack([x, y, z], dim=-1)
            return voxel_coordinates.reshape(1, -1, 3)
        def soft_clamp(x, cval=3.):
            return x.div(cval).tanh_().mul(cval)
        def visualize_voxel(voxel_coordinates, voxel_densities, voxel_size, scene_data, density_threshold=0.03, camera_view='default'):
            import plotly.graph_objects as go
            from typing import List, Optional, Union
            def voxel2points_thresholding(voxel_coords, voxel_densities, voxelSize, density_thresh=0.1):
                voxel_coords, voxel_densities = voxel_coords[0], voxel_densities[0]
                show = voxel_densities >= density_thresh
                points = torch.cat(
                    (
                        voxel_coords[:, [0]] * voxelSize[0],
                        voxel_coords[:, [1]] * voxelSize[1],
                        voxel_coords[:, [2]] * voxelSize[2],
                    ),
                    dim=1,
                )
                points = voxel_coords
                return points[show[:,0]], torch.ones((voxel_coords.shape[0],), dtype=torch.long)[show[:,0]], voxel_coords[show[:,0]]
            # plotly
            def vis_occ_plotly(
                coords: np.array = None,
                offset = [0, 0, 0],
                # points: np.ndarray,
                vis_aabb: List[Union[int, float]] = [-50, -50, -5, 50, 50, 15],
                colors: np.array = None,
                x_ratio: float = 1.0,
                y_ratio: float = 1.0,
                z_ratio: float = 0.4, # 0.125,
                size: int = 5,
                black_bg: bool = False,
                title: str = None,
            ) -> go.Figure:  # type: ignore
                
                fig = go.Figure()  # start with an empty figure
                coords = coords + offset
                if coords is not None:
                    # Add static trace
                    static_trace = go.Scatter3d(
                        x=coords[:, 0],
                        y=coords[:, 1],
                        z=coords[:, 2],
                        mode="markers",
                        marker=dict(
                            size=size,
                            color='black',
                            symbol="square",
                        ),
                    )
                    fig.add_trace(static_trace)

                ''' background color setting '''
                title_font_color = "white" if black_bg else "black"
                if not black_bg:
                    fig.update_layout(
                        scene=dict(
                            xaxis=dict(
                                title="x",
                                showspikes=False,
                                range=[vis_aabb[0], vis_aabb[3]],
                            ),
                            yaxis=dict(
                                title="y",
                                showspikes=False,
                                range=[vis_aabb[1], vis_aabb[4]],
                            ),
                            zaxis=dict(
                                title="z",
                                showspikes=False,
                                range=[vis_aabb[2], vis_aabb[5]],
                            ),
                            aspectmode="manual",
                            aspectratio=dict(x=x_ratio, y=y_ratio, z=z_ratio),
                        ),
                        margin=dict(r=0, b=10, l=0, t=10),
                        hovermode=False,
                        title=dict(
                            text=title,
                            font=dict(color=title_font_color),
                            x=0.5,
                            y=0.95,
                            xanchor="center",
                            yanchor="top",
                        )
                        if title
                        else None,  # Title addition
                    )
                else:
                    fig.update_layout(
                        scene=dict(
                            xaxis=dict(
                                title="x",
                                showspikes=False,
                                range=[vis_aabb[0], vis_aabb[3]],
                                backgroundcolor="rgb(0, 0, 0)",
                                gridcolor="gray",
                                showbackground=True,
                                zerolinecolor="gray",
                                tickfont=dict(color="gray"),
                            ),
                            yaxis=dict(
                                title="y",
                                showspikes=False,
                                range=[vis_aabb[1], vis_aabb[4]],
                                backgroundcolor="rgb(0, 0, 0)",
                                gridcolor="gray",
                                showbackground=True,
                                zerolinecolor="gray",
                                tickfont=dict(color="gray"),
                            ),
                            zaxis=dict(
                                title="z",
                                showspikes=False,
                                range=[vis_aabb[2], vis_aabb[5]],
                                backgroundcolor="rgb(0, 0, 0)",
                                gridcolor="gray",
                                showbackground=True,
                                zerolinecolor="gray",
                                tickfont=dict(color="gray"),
                            ),
                            aspectmode="manual",
                            aspectratio=dict(x=x_ratio, y=y_ratio, z=z_ratio),
                        ),
                        margin=dict(r=0, b=10, l=0, t=10),
                        hovermode=False,
                        paper_bgcolor="black",
                        plot_bgcolor="rgba(0,0,0,0)",
                        title=dict(
                            text=title,
                            font=dict(color=title_font_color),
                            x=0.5,
                            y=0.95,
                            xanchor="center",
                            yanchor="top",
                        )
                        if title
                        else None,  # Title addition
                    )
                
                ''' set camera pose '''
                eye = np.array([-1, 0, 0.5])
                eye = eye.tolist()
                fig.update_layout(
                    scene_camera=dict(
                        eye=dict(x=eye[0], y=eye[1], z=eye[2]),
                    ),
                )
                return fig
            # open3d
            def voxel_profile(voxel, voxel_size):
                centers = torch.cat((voxel[:, :2], voxel[:, 2][:, None] - voxel_size[2] / 2), dim=1)
                wlh = torch.cat(
                    (
                        torch.tensor(voxel_size[0]).repeat(centers.shape[0])[:, None],
                        torch.tensor(voxel_size[1]).repeat(centers.shape[0])[:, None],
                        torch.tensor(voxel_size[2]).repeat(centers.shape[0])[:, None],
                    ),
                    dim=1,
                )
                yaw = torch.full_like(centers[:, 0:1], 0)
                return torch.cat((centers, wlh, yaw), dim=1)
            def my_compute_box_3d(center, size, heading_angle):
                h, w, l = size[:, 2], size[:, 0], size[:, 1]
                heading_angle = -heading_angle - math.pi / 2
                center[:, 2] = center[:, 2] + h / 2
                # R = rotz(1 * heading_angle)
                l, w, h = (l / 2).unsqueeze(1), (w / 2).unsqueeze(1), (h / 2).unsqueeze(1)
                x_corners = torch.cat([-l, l, l, -l, -l, l, l, -l], dim=1)[..., None]
                y_corners = torch.cat([w, w, -w, -w, w, w, -w, -w], dim=1)[..., None]
                z_corners = torch.cat([h, h, h, h, -h, -h, -h, -h], dim=1)[..., None]
                # corners_3d = R @ torch.vstack([x_corners, y_corners, z_corners])
                corners_3d = torch.cat([x_corners, y_corners, z_corners], dim=2)
                corners_3d[..., 0] += center[:, 0:1]
                corners_3d[..., 1] += center[:, 1:2]
                corners_3d[..., 2] += center[:, 2:3]
                return corners_3d
            def show_point_cloud(
                points: np.ndarray,
                colors=True,
                points_colors=None,
                bbox3d=None,
                voxelize=False,
                bbox_corners=None,
                linesets=None,
                vis=None,
                offset=[0, 0, 0],
                ) -> None:
                """
                :param points:
                :param colors: false 不显示点云颜色
                :param points_colors:
                :param bbox3d: voxel边, Nx7 (center, wlh, yaw=0)
                :param voxelize: false 不显示voxel边界
                :return:
                """

                # create visualizer
                import open3d as o3d
                if vis is None:
                    vis = o3d.visualization.VisualizerWithKeyCallback()
                    vis.create_window(width=1400, height=900)
                opt = vis.get_render_option()
                opt.background_color = np.asarray([1, 1, 1])

                # generate point cloud
                pcd = o3d.geometry.PointCloud()
                if isinstance(offset, list) or isinstance(offset, tuple):
                    offset = np.array(offset)
                pcd.points = o3d.utility.Vector3dVector(points + offset)

                # point cloud color
                if colors:
                    pcd.colors = o3d.utility.Vector3dVector(points_colors[:, :3])
                
                # generate voxels from point cloud
                vgd = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size=0.5)
                vis.add_geometry(vgd)
                if voxelize:
                    line_sets = o3d.geometry.LineSet()
                    line_sets.points = o3d.open3d.utility.Vector3dVector(
                        bbox_corners.reshape((-1, 3)) + offset
                    )
                    line_sets.lines = o3d.open3d.utility.Vector2iVector(linesets.reshape((-1, 2)))
                    line_sets.paint_uniform_color((0, 0, 0))
                    vis.add_geometry(line_sets)

                # Create plane meshes, x, y, z
                    # Define centers for the planes
                    #       x               y
                    # left_to_right   back_to_front
                def create_plane(center, size=10, plane='y'):
                    """ Create a plane mesh centered at 'center'. """

                    if plane == 'y': 
                        width=size 
                        height=0.01 
                        depth=size
                    elif plane == 'x':
                        width=0.01 
                        height=size 
                        depth=size
                    elif plane == 'z':
                        width=size 
                        height=size 
                        depth=0.01
                    mesh = o3d.geometry.TriangleMesh.create_box(width=width, height=height, depth=depth)
                    # mesh.translate(center - np.array([size / 2, 0, size / 2]))
                    # mesh.translate(center)
                    if plane == 'y': 
                        mesh.translate(center - np.array([size / 2, 0, size / 2]))
                    elif plane == 'x':
                        mesh.translate(center - np.array([0, size / 2, size / 2]))
                    elif plane == 'z':
                        mesh.translate(center - np.array([size / 2, size / 2, 0]))

                    return mesh
                center_x100 = np.array([100, 0, 0])
                center_xneg100 = np.array([-100, 0, 0])
                center_y100 = np.array([0, 100, 0])
                center_yneg100 = np.array([0, -100, 0])
                center_z50 = np.array([0, 0, 50])
                center_zneg50 = np.array([0, 0, -50])

                plane_x100 = create_plane(center_x100, plane='x')
                plane_xneg100 = create_plane(center_xneg100, plane='x')
                plane_y100 = create_plane(center_y100, plane='y')
                plane_yneg100 = create_plane(center_yneg100, plane='y')
                # plane_z50 = create_plane(center_z50)
                # plane_zneg50 = create_plane(center_zneg50)

                # Visualize the planes
                vis.add_geometry(plane_x100) 
                vis.add_geometry(plane_xneg100)
                vis.add_geometry(plane_y100) 
                vis.add_geometry(plane_yneg100)

                return vis

            ''' thresholding '''
            pcd, labels, occIdx = voxel2points_thresholding(
                voxel_coordinates.cpu(),
                voxel_densities.cpu(),
                voxel_size,
                density_thresh=density_threshold
            )


            ''' option 1: visualize and save using plotly'''
            # fig = vis_occ_plotly(pcd.numpy(), offset=[-50, -50, -5, 50, 50, 3][:3])
            fig = vis_occ_plotly(pcd.numpy())
            fig.write_html('./output_path.html')


            ''' option 2: visualize using open3d '''
            # create boxes
            bboxes = voxel_profile(pcd, voxel_size)
            # create box corners and edges
            bboxes_corners = my_compute_box_3d(
                bboxes[:, 0:3], bboxes[:, 3:6], bboxes[:, 6:7]
            )
            bases_ = torch.arange(0, bboxes_corners.shape[0] * 8, 8)
            edges = torch.tensor(
                [
                    [0, 1],
                    [1, 2],
                    [2, 3],
                    [3, 0],
                    [4, 5],
                    [5, 6],
                    [6, 7],
                    [7, 4],
                    [0, 4],
                    [1, 5],
                    [2, 6],
                    [3, 7],
                ]
            )  # lines along y-axis
            edges = edges.reshape((1, 12, 2)).repeat(bboxes_corners.shape[0], 1, 1)
            edges = edges + bases_[:, None, None]
            # create voxel color according to height
            norm = plt.Normalize(pcd[:,-1].min()-15, pcd[:,-1].max())  # Normalize the colors
            cmap = plt.get_cmap('gray')  # Choose the colormap
            scalar_map = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
            pcds_colors = scalar_map.to_rgba(pcd[:,-1])

            # visualize with open3d
            vis = show_point_cloud(
                points=pcd.numpy(),
                colors=True,
                points_colors=pcds_colors,
                voxelize=True,
                bbox3d=bboxes.numpy(),
                bbox_corners=bboxes_corners.numpy(),    #
                linesets=edges.numpy(),   # 
                vis=None,
                # offset=cfg.point_cloud_range[:3],
                #                                           # sample from tree
            )


            # open3d: default camera view
            if camera_view == 'default':
                vis.run()
                vis.poll_events()
                vis.update_renderer()

            # open3d: manually specify the camera view
            elif camera_view == 'manual':
                # Get the view control object
                view_control = vis.get_view_control()
                view_control.get_field_of_view()
                view_control.change_field_of_view(step=-6)

                # # Set the camera parameters
                params = view_control.convert_to_pinhole_camera_parameters()
                # view_control.change_field_of_view(-30)
                camera_matrix = np.array([[1, 0, 0, 0],
                                        [0, 0, -1,0],    # pos = upward
                                        [0, 1, 0, 0], # pos = back
                                        [0, 0, 0, 1]])
                params.extrinsic = camera_matrix
                view_control.convert_from_pinhole_camera_parameters(params, allow_arbitrary=True)
                vis.run()
                vis.poll_events()
                vis.update_renderer()

            # open3d: set 6 camera view
            elif camera_view == 'six_cam':
                
                # create image save directory
                if self.save_visualized_imgs:
                    directory = self.vis_save_directory + 'voxel'
                    if not os.path.exists(directory):
                        os.makedirs(directory)
                    num_file = len(os.listdir(directory))
                    dir_name = directory + '/{}/'.format(num_file)
                    if not os.path.exists(dir_name):
                        os.makedirs(dir_name)

                # create 6 camear poses
                # f = 0.0055
                camera_extrinsic_lst = []
                camera_intrinsic_lst = []
                for i in range(6):
                    # extrinsics
                    trans = scene_data.get("trans")[0, i].cpu().detach().numpy()
                    rots = scene_data.get("rots")[0, i].cpu().detach().numpy()
                    transform_extrinsic = np.eye(4)
                    transform_extrinsic[:3, :3] = np.linalg.inv(rots)
                    transform_extrinsic[:3, 3] = rots @ trans
                    camera_extrinsic_lst.append(transform_extrinsic)
                    # intrinsics
                    compound_intrinsics = scene_data.get("intrins")[0, i].cpu().detach().numpy()
                    camera_intrinsic_lst.append(compound_intrinsics)

                # specify the camera view
                view_control = vis.get_view_control()
                view_control.get_field_of_view()
                view_control.change_field_of_view(step=-6)
                width, height = 1400, 900
                for i in range(6):
                    params = view_control.convert_to_pinhole_camera_parameters()
                    # view_control.change_field_of_view(-30)
                    camera_extrinsic = camera_extrinsic_lst[i]
                    params.extrinsic = camera_extrinsic
                    camera_intrinsic = camera_intrinsic_lst[i]
                    params.intrinsic.set_intrinsics(
                        width=width, height=height, fx=camera_intrinsic[0][0], fy=camera_intrinsic[1][1], cx=camera_intrinsic[0][2], cy=camera_intrinsic[1][2])
                    view_control.convert_from_pinhole_camera_parameters(params, allow_arbitrary=True)
                    vis.run()
                    vis.poll_events()
                    vis.update_renderer()

                    # save the file name
                    file_name = dir_name + '/{}.png'.format(i)
                    if self.save_visualized_imgs:
                        vis.capture_screen_image(file_name)

            vis.destroy_window()
            vis.close()

        # create voxel coordinates to query the octree
        coordinates = create_voxel_coordinate()
        coordinates = coordinates.to(intermediates.octree7.device)

        # # query from octree - old code
        # sampled_features = self.model.projector.volume_renderer.sample_from_octree(coordinates, intermediates)
        # # post-process densities
        # voxel_features = sampled_features[0, :, :]                  # torch.Size([440657, 65])
        # sigma = sampled_features[..., -1:]
        # voxel_density_logits = soft_clamp(sigma, cval=10)
        # voxel_densities = torch.exp(voxel_density_logits - 1)       # torch.Size([1, 440657, 1])

        # query from octree
        out = self.model.projector.volume_renderer.collect_feats_from_octree(coordinates, intermediates)
        voxel_features = out['rgb'][0]                                  # [B, num_rays*samples_per_ray, 65]
        voxel_density_logits = out['density_logits'].unsqueeze(0)       # [num_rays*samples_per_ray, 1]
        # activation bias of -1 makes things initialize better
        if self.model.projector.config.density_activation == 'sigmoid':
            voxel_densities = F.sigmoid(voxel_density_logits - 1)
        elif self.model.projector.config.density_activation == 'softplus':
            voxel_densities = F.softplus(voxel_density_logits - 1)
        elif self.model.projector.config.density_activation == 'exp':
            voxel_densities = torch.exp(voxel_density_logits - 1)


        # visualize via different camera views
        voxel_size = (0.5, 0.5, 0.5)
        density_threshold=0.1
        camera_view = ['default', 'manual', 'six_cam'][2]
        visualize_voxel(coordinates, voxel_densities, voxel_size, scene_data, density_threshold=density_threshold, camera_view=camera_view)

        return voxel_features, coordinates, voxel_densities

    def edge_loss(self, intermediates, outputs, iteration, 
                    input_gt_lidar_depth_imgs, input_emernerf_depth_imgs, 
                    input_coarse_depth_pred, input_coarse_weight_pred, input_coarse_depth_samples,
                    input_fine_depth_pred, input_fine_weights_pred, input_fine_depth_samples,
                    target_gt_lidar_depth_imgs, target_emernerf_depth_imgs, 
                    target_pred_depth_imgs, target_weight_pred, target_depth_samples
        ):
        def visualize_depth(value, lo=4, hi=120, curve_fn=lambda x: -np.log(x + 1e-6)):
            import matplotlib.cm as cm
            color_map = cm.get_cmap("turbo")
            value, lo, hi = [curve_fn(x) for x in [value, lo, hi]]
            value = np.nan_to_num(
                    np.clip((value - np.minimum(lo, hi)) / np.abs(hi - lo), 0, 1)
                )
            colorized = color_map(value)[..., :3]
            return colorized

        output = {}
        coarse_depth_edge_mask_list = []
        fine_depth_edge_mask_list = []
        target_depth_edge_mask_list = []
        
        # detect depth edge for each camera, transform depth to rgb map for edge detection
        for cam in range(len(intermediates.coarse_depth)):
            coarse_depth_edge = cv.Canny((visualize_depth(intermediates.coarse_depth[cam].cpu().detach().numpy())*255).astype(np.uint8), 150, 200) == 255
            coarse_depth_edge_mask_list.append(coarse_depth_edge)
        for cam in range(len(intermediates.get("fine_depth"))):
            fine_depth_edge = cv.Canny((visualize_depth(intermediates.get("fine_depth")[cam].cpu().detach().numpy())*255).astype(np.uint8), 150, 200) == 255
            fine_depth_edge_mask_list.append(fine_depth_edge)
        for cam in range(len(intermediates.get("target_pred_depth_image"))):
            target_depth_edge = cv.Canny((visualize_depth(intermediates.get("target_pred_depth_image")[:,0][cam].cpu().detach().numpy())*255).astype(np.uint8), 150, 200) == 255
            target_depth_edge_mask_list.append(target_depth_edge)
        coarse_depth_edge_masks = torch.from_numpy(np.stack(coarse_depth_edge_mask_list)).unsqueeze(0).to(input_gt_lidar_depth_imgs)
        fine_depth_edge_masks = torch.from_numpy(np.stack(fine_depth_edge_mask_list)).unsqueeze(0).to(input_gt_lidar_depth_imgs)
        target_depth_edge_masks = torch.from_numpy(np.stack(target_depth_edge_mask_list)).unsqueeze(0).unsqueeze(2).to(target_gt_lidar_depth_imgs)

        # ignore pixels whose depth are not at the edge
        coarse_depth_edge_masks[torch.where(coarse_depth_edge_masks==0)] = torch.nan
        fine_depth_edge_masks[torch.where(fine_depth_edge_masks==0)] = torch.nan
        target_depth_edge_masks[torch.where(target_depth_edge_masks==0)] = torch.nan

        # ignore depths that are too far 
        coarse_depth_edge_masks[torch.where(intermediates.coarse_depth.unsqueeze(0) > self.edge_loss_distance)] = torch.nan
        fine_depth_edge_masks[torch.where(intermediates.get("fine_depth").unsqueeze(0) > self.edge_loss_distance)] = torch.nan
        target_depth_edge_masks[torch.where(intermediates.get("target_pred_depth_image").unsqueeze(0) > self.edge_loss_distance)] = torch.nan

        ''' input coarse depth edge loss '''
        # LiDAR
        output.update(
            self.coarse_depth_loss_lidar(input_coarse_depth_pred, input_gt_lidar_depth_imgs * coarse_depth_edge_masks, name="coarse_edge_depth_loss_lidar")
            )
        # EmerNerf
        output.update(
            self.coarse_depth_loss_emernerf(input_coarse_depth_pred, input_emernerf_depth_imgs * coarse_depth_edge_masks, name="coarse_edge_depth_loss_emernerf")
            )

        ''' input fine depth edge loss '''
        # LiDAR
        output.update(
            self.fine_depth_loss_lidar(input_fine_depth_pred, input_gt_lidar_depth_imgs * fine_depth_edge_masks, name="fine_edge_depth_loss_lidar")
            )
        # EmerNerf
        output.update(
            self.fine_depth_loss_emernerf(input_fine_depth_pred, input_emernerf_depth_imgs * fine_depth_edge_masks, name="fine_edge_depth_loss_emernerf")
            )              

        ''' target depth edge loss '''
        # LiDAR
        output.update(
            self.target_lidar_depth_loss(target_pred_depth_imgs, target_gt_lidar_depth_imgs * target_depth_edge_masks, name="target_edge_depth_loss_lidar")
            )
        # EmerNerf
        output.update(
            self.target_emernerf_depth_loss(target_pred_depth_imgs, target_emernerf_depth_imgs * target_depth_edge_masks, name="target_edge_depth_loss_emernerf")
            )

        ''' depth edge loss weight '''
        for key, values in output.items():
            if 'obs' not in key:
                output[key] = values * self.edge_loss_aug_weight

        ''' line of sight loss '''
        if self.enforce_edge_LoS_loss:
            if (self.enforce_LoS_loss and iteration is not None and iteration >= self.LoS_start_iter):
                # linear decy on epsilonc
                def epsilon_decay(step):
                    if step < self.LoS_start_iter:
                        return self.LoS_epsilon_start
                    elif step > self.LoS_fix_iter:
                        return self.LoS_epsilon_end
                    else:
                        return m * step + b
                m = (self.LoS_epsilon_end - self.LoS_epsilon_start) / (
                    self.LoS_fix_iter - self.LoS_start_iter
                    )
                b = self.LoS_epsilon_start - m * self.LoS_start_iter
                epsilon = epsilon_decay(iteration)

                ''' input coarse depth LoS '''
                # LiDAR
                output.update(                 # ([1, 6, 128, 228])      ([6, 286, 128, 228])     ([6, 286, 128, 228])
                    self.coarse_LoS_loss_lidar(input_gt_lidar_depth_imgs * coarse_depth_edge_masks, input_coarse_weight_pred, input_coarse_depth_samples, epsilon, name = 'coarse_edge_LoS_loss_lidar')
                    )
                # EmerNerf
                output.update(
                    self.coarse_LoS_loss_emernerf(input_emernerf_depth_imgs * coarse_depth_edge_masks, input_coarse_weight_pred, input_coarse_depth_samples, epsilon, name = 'coarse_edge_LoS_loss_emernerf')
                    )
                
                ''' input fine depth LoS '''
                # LiDAR
                output.update(
                    self.fine_LoS_loss_lidar(input_gt_lidar_depth_imgs * fine_depth_edge_masks, input_fine_weights_pred, input_fine_depth_samples, epsilon, name = 'fine_edge_LoS_loss_lidar')
                    )
                # EmerNerf
                output.update(
                    self.fine_LoS_loss_emernerf(input_emernerf_depth_imgs * fine_depth_edge_masks, input_fine_weights_pred, input_fine_depth_samples, epsilon, name = 'fine_edge_LoS_loss_emernerf')
                    )
                
                ''' target depth LoS '''
                # LiDAR
                output.update(
                    self.target_LoS_loss_lidar((target_gt_lidar_depth_imgs*target_depth_edge_masks).squeeze(2), target_weight_pred, target_depth_samples, epsilon, name = 'target_lidar_edge_LoS_loss')
                    )
                # EmerNerf
                output.update(
                    self.target_LoS_loss_emernerf((target_emernerf_depth_imgs*target_depth_edge_masks).squeeze(2), target_weight_pred, target_depth_samples, epsilon, name = 'target_emernerf_edge_LoS_loss')
                    )
                
            key_to_delete = []
            for key, values in output.items():
                if 'LoS' in key:
                    if 'obs' not in key:
                        if type(output['input_emernerf_edge_LoS_loss_obs']) is not torch.Tensor:
                            key_to_delete.append(key)
            for key in key_to_delete: 
                del output[key]
                del output[key+'_obs']

        outputs = {**outputs, **output}
        return outputs

    def appearance_loss_cal(self, outputs, recon_imgs, gt_imgs):
        # l1 loss
        if not self.novel_view_mode:
            outputs.update(self.l1_loss(recon_imgs, gt_imgs, name="rgb_l1_loss"))
        # lpips loss, used only for rgb
        if not self.render_foundation_model_feat:
            if self.lpips_loss is not None:
                lpips_dict = self.lpips_loss(recon_imgs, gt_imgs)
                lpips_name = list(lpips_dict.keys())[0]
                outputs.update({lpips_name: lpips_dict[lpips_name].mean()})
    
        return outputs

    def depth_lidar_loss_cal(self, outputs, input_gt_lidar_depth_imgs, input_coarse_depth_pred, input_fine_depth_pred, \
                             target_gt_lidar_depth_imgs, target_pred_depth_imgs):
        # coarse depth loss
        outputs.update(
            self.coarse_depth_loss_lidar(input_coarse_depth_pred, input_gt_lidar_depth_imgs, name="coarse_depth_loss_lidar")
            )
        # fine depth loss
        if self.enable_fine_depth_loss:
            outputs.update(
                self.fine_depth_loss_lidar(input_fine_depth_pred, input_gt_lidar_depth_imgs, name="fine_depth_loss_lidar")
                )
        # target depth loss
        outputs.update(
            self.target_lidar_depth_loss(target_pred_depth_imgs, target_gt_lidar_depth_imgs, name="target_depth_loss_lidar")
            )
        return outputs

    def depth_emernerf_loss_cal(self, outputs, input_emernerf_depth_imgs, input_coarse_depth_pred, input_fine_depth_pred,
                                target_emernerf_depth_imgs, target_pred_depth_imgs):
        # coarse depth loss
        outputs.update(
            self.coarse_depth_loss_emernerf(input_coarse_depth_pred, input_emernerf_depth_imgs, name="coarse_depth_loss_emernerf")
            )
        # fine depth loss
        if self.enable_fine_depth_loss:
            outputs.update(
                self.fine_emernerf_depth_loss(input_fine_depth_pred, input_emernerf_depth_imgs, name="fine_depth_loss_emernerf")
                )
        # target depth loss
        outputs.update(
            self.target_emernerf_depth_loss(target_pred_depth_imgs, target_emernerf_depth_imgs, name="target_depth_loss_emernerf")
            )

        return outputs

    def pretrained_depth_loss_cal(self, outputs, intermediates, gt=None, input_coarse_depth_pred=None, target_pred_depth_imgs=None):
        # coarse depth loss
        outputs.update(
            self.depthanything_depth_loss(input_coarse_depth_pred, gt)
            )
        
        # midas depth loss
        if intermediates.has("target_midas_depth_pred"):
            # midas depth supervision
            outputs.update({"midas_depth_imgs": intermediates.get("target_midas_depth_pred_vis").clamp(0,1)})
            target_midas_depth_pred = intermediates.get("target_midas_depth_pred")

            target_midas_depth_pred[target_midas_depth_pred==0] = float('nan')
            target_midas_depth_pred = F.interpolate(target_midas_depth_pred, (target_pred_depth_imgs.shape[-2], target_pred_depth_imgs.shape[-1]), mode='bilinear')

            outputs.update(
                self.midas_depth_loss(target_pred_depth_imgs, target_midas_depth_pred.unsqueeze(0))
            )

        return outputs

    def weight_entropy_loss_cal(self, outputs, input_coarse_weight_pred, target_weight_pred):
        # coarse weight entropy loss
        outputs.update(
            self.coarse_weight_entropy_loss(input_coarse_weight_pred, name='coarse_nerf_weight_entropy_loss')
            )
        # target weight entropy loss
        outputs.update(
            self.target_weight_entropy_loss(
                target_weight_pred,
                name='target_nerf_weight_entropy_loss'
                )
            )
        return outputs

    def opacity_loss_cal(self, outputs, input_coarse_weight_pred, input_fine_weights_pred, target_weight_pred):
        # coarse opacity
        outputs.update(
            self.coarse_opacity_loss(input_coarse_weight_pred, name='coarse_opacity_loss')
            )
        # fine opacity loss
        outputs.update(
            self.fine_opacity_loss(input_fine_weights_pred, name='fine_opacity_loss')
            )
        # target opacity loss
        outputs.update(
            self.target_opacity_loss(target_weight_pred, name='target_opacity_loss')
            )
        return outputs

    def line_of_sight_loss_cal(self, outputs, iteration, 
                               input_gt_lidar_depth_imgs, input_emernerf_depth_imgs, 
                               input_coarse_weight_pred, input_coarse_depth_samples,
                               input_fine_weights_pred, input_fine_depth_samples,
                               target_gt_lidar_depth_imgs, target_emernerf_depth_imgs, 
                               target_weight_pred, target_depth_samples
                                ):
        if (self.enforce_LoS_loss and iteration is not None and iteration >= self.LoS_start_iter):
            def epsilon_decay(step):
                if step < self.LoS_start_iter:
                    return self.LoS_epsilon_start
                elif step > self.LoS_fix_iter:
                    return self.LoS_epsilon_end
                else:
                    return m * step + b
            # linear decy on epsilonc
            m = (self.LoS_epsilon_end - self.LoS_epsilon_start) / (
                self.LoS_fix_iter - self.LoS_start_iter
            )
            b = self.LoS_epsilon_start - m * self.LoS_start_iter
            epsilon = epsilon_decay(iteration)
            
            ''' input coarse depth '''
            # lidar gt
            outputs.update(                 # ([1, 6, 128, 228])      ([6, 286, 128, 228])     ([6, 286, 128, 228])
                self.coarse_LoS_loss_lidar(input_gt_lidar_depth_imgs, input_coarse_weight_pred, input_coarse_depth_samples, epsilon, name = 'coarse_LoS_loss_lidar')
            )
            # emernerf gt
            outputs.update(
                self.coarse_LoS_loss_emernerf(input_emernerf_depth_imgs, input_coarse_weight_pred, input_coarse_depth_samples, epsilon, name = 'coarse_LoS_loss_emernerf')
            )

            ''' input fine depth '''
            # outputs.update(
                # self.fine_LoS_loss_lidar(input_gt_lidar_depth_imgs, input_fine_weights_pred, input_fine_depth_samples, epsilon, name = 'fine_LoS_loss_lidar'))
            # outputs.update(
                # self.fine_LoS_loss_emernerf(input_emernerf_depth_imgs, input_fine_weights_pred, input_fine_depth_samples, epsilon, name = 'fine_LoS_loss_emernerf')
            # )

            ''' target depth '''
            outputs.update(
                self.target_LoS_loss_lidar(target_gt_lidar_depth_imgs.squeeze(2), target_weight_pred, target_depth_samples, epsilon, name = 'target_LoS_loss_lidar')
                )
            outputs.update(
                self.target_LoS_loss_emernerf(target_emernerf_depth_imgs.squeeze(2), target_weight_pred, target_depth_samples, epsilon, name = 'target_LoS_loss_emernerf')
                )

        return outputs

    def recoloring(self, recon_imgs, gt_imgs, N_cams):
        for i in range(N_cams):
            # Flatten, add offset dimension
            X = recon_imgs[:,:,i].flatten().unsqueeze(1)  # i-th channel in recon_imgs
            X_with_bias = torch.cat([X, torch.ones_like(X)], dim=1)  # add constant 1 as offsets
            
            Y = gt_imgs[:,:,i].flatten().unsqueeze(1)  # i-th channel in gt_imgs
            
            # solve Ax = B where A is recon_imgs and B is GT to find the mapping a * recon_imgs + b = GT
            sol = torch.linalg.lstsq(X_with_bias, Y).solution
            a = sol[0]  # scaling factor
            b = sol[1]  # offset
            
            # apply the transform
            recon_imgs[:,:,i] = torch.clip(recon_imgs[:,:,i] * a + b, 0, 1)
        return recon_imgs
        
    def input_depth_evaluation_cal(self, outputs, is_val, input_gt_lidar_depth_imgs, input_emernerf_depth_imgs, input_coarse_depth_pred):
        ''' input coarse depth evaluation '''
        # ''' lidar depth error '''
        input_coarse_lidar_abs_rel, input_coarse_lidar_sq_rel, input_coarse_lidar_rmse, input_coarse_lidar_rmse_log, input_coarse_lidar_a1, input_coarse_lidar_a2, input_coarse_lidar_a3 \
            = self.compute_depth_errors(input_gt_lidar_depth_imgs, input_coarse_depth_pred)
        # ''' emerenrf depth error '''
        input_coarse_emernerf_abs_rel, input_coarse_emernerf_sq_rel, input_coarse_emernerf_rmse, input_coarse_emernerf_rmse_log, input_coarse_emernerf_a1, input_coarse_emernerf_a2, input_coarse_emernerf_a3 \
            = self.compute_depth_errors(input_emernerf_depth_imgs, input_coarse_depth_pred)
        # ''' emernerf inner depth error '''
        input_emernerf_inner_abs_rel, _, _, _, _, _, _ = self.compute_depth_errors(input_emernerf_depth_imgs, input_coarse_depth_pred, 0.001, self.inner_range)
        # default log
        outputs.update(
            {
                "coarse_depth_lidar_error_abs_rel": input_coarse_lidar_abs_rel,
                "coarse_depth_emernerf_error_abs_rel": input_coarse_emernerf_abs_rel,
                "coarse_depth_emernerf_inner_error_abs_rel": input_emernerf_inner_abs_rel
            }
        )
        # evaluation log
        if is_val:
            outputs.update(
                    {
                        "coarse_depth_lidar_abs_rel": input_coarse_lidar_abs_rel,
                        "coarse_depth_lidar_sq_rel": input_coarse_lidar_sq_rel,
                        "coarse_depth_lidar_rmse": input_coarse_lidar_rmse,
                        "coarse_depth_lidar_rmse_log": input_coarse_lidar_rmse_log,
                        "coarse_depth_lidar_a1": input_coarse_lidar_a1,
                        "coarse_depth_lidar_a2": input_coarse_lidar_a2,
                        "coarse_depth_lidar_a3": input_coarse_lidar_a3
                    }
                )
            outputs.update(
                    {
                        "coarse_depth_emernerf_abs_rel": input_coarse_emernerf_abs_rel,
                        "coarse_depth_emernerf_sq_rel": input_coarse_emernerf_sq_rel,
                        "coarse_depth_emernerf_rmse": input_coarse_emernerf_rmse,
                        "coarse_depth_emernerf_rmse_log": input_coarse_emernerf_rmse_log,
                        "coarse_depth_emernerf_a1": input_coarse_emernerf_a1,
                        "coarse_depth_emernerf_a2": input_coarse_emernerf_a2,
                        "coarse_depth_emernerf_a3": input_coarse_emernerf_a3
                    }
                )
            

        return outputs
    

    def target_evaluation_cal(self, outputs, is_val, gt_imgs, recon_imgs, target_gt_lidar_depth_imgs, target_emernerf_depth_imgs, target_pred_depth_imgs, \
                                  training_target_smart_sample, add_next_frame_for_eval):
            
        ''' organize gt targets and predictions '''
        if (training_target_smart_sample) or (add_next_frame_for_eval):
            # original_view + other_view
            if training_target_smart_sample: assert not is_val, "training_target_smart_sample should not be in the evaluation mode"
            if add_next_frame_for_eval: assert is_val, "add_next_frame_for_eval should only happen in the evaluation mode" 
            assert not (training_target_smart_sample and add_next_frame_for_eval), "training_target_smart_sample and add_next_frame_for_eval cannot be True simultaneously"
            half_cam = int(gt_imgs.shape[1] / 2)
            # original view
            original_view_gt_imgs = gt_imgs[:, half_cam:]
            original_view_recon_imgs = recon_imgs[:, half_cam:]
            original_view_gt_lidar_depth = target_gt_lidar_depth_imgs[:, half_cam:]
            original_view_gt_emernerf_depth = target_emernerf_depth_imgs[:, half_cam:]
            original_view_depth_pred = target_pred_depth_imgs[:, half_cam:]
        else:
            original_view_gt_imgs = gt_imgs
            original_view_recon_imgs = recon_imgs
            original_view_gt_lidar_depth = target_gt_lidar_depth_imgs
            original_view_gt_emernerf_depth = target_emernerf_depth_imgs
            original_view_depth_pred = target_pred_depth_imgs

        ''' original view evaluation '''
        # psnr
        psnr = (-10 * torch.log10(F.mse_loss(original_view_gt_imgs, original_view_recon_imgs))).item()
        outputs.update({"psnr": psnr})
        # ssim
        ssim_val_lst = []
        for onebatch_gt_imgs, onebatch_pred_imgs in zip(original_view_gt_imgs.cpu().detach().numpy(), original_view_recon_imgs.cpu().detach().numpy()):
            for oneimg_gt_imgs, oneimg_pred_imgs in zip(onebatch_gt_imgs, onebatch_pred_imgs):
                if self.render_foundation_model_feat:       # foundation model targets
                    ssim_val_lst.append(ssim(oneimg_gt_imgs, oneimg_pred_imgs, data_range=1.0, channel_axis=-1))
                else:
                    ssim_val_lst.append(ssim(oneimg_gt_imgs, oneimg_pred_imgs, data_range=1.0, channel_axis=0))
        outputs.update({"ssim": sum(ssim_val_lst) / len(ssim_val_lst)})
        # lidar depth error
        target_lidar_depth_abs_rel, target_lidar_depth_sq_rel, target_lidar_depth_rmse, target_lidar_depth_rmse_log, target_lidar_depth_a1, target_lidar_depth_a2, target_lidar_depth_a3 \
            = self.compute_depth_errors(original_view_gt_lidar_depth, original_view_depth_pred)
        # emerenrf depth error
        target_emernerf_depth_abs_rel, target_emernerf_depth_sq_rel, target_emernerf_depth_rmse, target_emernerf_depth_rmse_log, target_emernerf_depth_a1, target_emernerf_depth_a2, target_emernerf_depth_a3 \
            = self.compute_depth_errors(original_view_gt_emernerf_depth, original_view_depth_pred)
        # emernerf inner depth error
        target_emernerf_inner_depth_abs_rel, target_emernerf_inner_depth_sq_rel, target_emernerf_inner_depth_rmse, target_emernerf_inner_depth_rmse_log, target_emernerf_inner_depth_a1, target_emernerf_inner_depth_a2, target_emernerf_inner_depth_a3 \
            = self.compute_depth_errors(original_view_gt_emernerf_depth, original_view_depth_pred, 0.001, self.inner_range)
        
        # simplified depth error log during training
        outputs.update(
            {
                "target_depth_lidar_error_abs_rel": target_lidar_depth_abs_rel,
                "target_depth_emernerf_error_abs_rel": target_emernerf_depth_abs_rel,
                "target_depth_emernerf_inner_error_abs_rel": target_emernerf_inner_depth_abs_rel
            }
        )
        # complete depth error log during evaluation
        if is_val:
            # lidar depth error
            outputs.update(
                {
                    "target_lidar_abs_rel": target_lidar_depth_abs_rel,
                    "target_lidar_sq_rel": target_lidar_depth_sq_rel,
                    "target_lidar_rmse": target_lidar_depth_rmse,
                    "target_lidar_rmse_log": target_lidar_depth_rmse_log,
                    "target_lidar_a1": target_lidar_depth_a1,
                    "target_lidar_a2": target_lidar_depth_a2,
                    "target_lidar_a3": target_lidar_depth_a3
                }
            )
            # emernerf depth error
            outputs.update(
                    {
                        "target_emernerf_abs_rel": target_emernerf_depth_abs_rel,
                        "target_emernerf_sq_rel": target_emernerf_depth_sq_rel,
                        "target_emernerf_rmse": target_emernerf_depth_rmse,
                        "target_emernerf_rmse_log": target_emernerf_depth_rmse_log,
                        "target_emernerf_a1": target_emernerf_depth_a1,
                        "target_emernerf_a2": target_emernerf_depth_a2,
                        "target_emernerf_a3": target_emernerf_depth_a3
                    }
                )
            # emernerf depth error for inner voxel
            outputs.update(
                    {
                        "target_emernerf_inner_abs_rel": target_emernerf_inner_depth_abs_rel,
                        "target_emernerf_inner_sq_rel": target_emernerf_inner_depth_sq_rel,
                        "target_emernerf_inner_rmse": target_emernerf_inner_depth_rmse,
                        "target_emernerf_inner_rmse_log": target_emernerf_inner_depth_rmse_log,
                        "target_emernerf_inner_a1": target_emernerf_inner_depth_a1,
                        "target_emernerf_inner_a2": target_emernerf_inner_depth_a2,
                        "target_emernerf_inner_a3": target_emernerf_inner_depth_a3
                    }
                )
            
        ''' special check: error between emernerf and lidar'''
        # lidar_emernerf_abs_rel, lidar_emernerf_sq_rel, lidar_emernerf_rmse, lidar_emernerf_rmse_log, lidar_emernerf_a1, lidar_emernerf_a2, lidar_emernerf_a3 \
            #  = self.compute_depth_errors(original_view_gt_lidar_depth, original_view_gt_emernerf_depth)
        # outputs.update(
                # {
                    # "lidar_emernerf_abs_rel": lidar_emernerf_abs_rel,
                    # "lidar_emernerf_sq_rel": lidar_emernerf_sq_rel,
                    # "lidar_emernerf_rmse": lidar_emernerf_rmse,
                    # "lidar_emernerf_rmse_log": lidar_emernerf_rmse_log,
                    # "lidar_emernerf_a1": lidar_emernerf_a1,
                    # "lidar_emernerf_a2": lidar_emernerf_a2,
                    # "lidar_emernerf_a3": lidar_emernerf_a3
                # }
            # )
        
        ''' optional other view evaluation '''
        if (training_target_smart_sample) or (add_next_frame_for_eval):
            # other view
            other_view_gt_imgs = gt_imgs[:, :half_cam]
            other_view_recon_imgs = recon_imgs[:, :half_cam]
            other_view_gt_lidar_depth = target_gt_lidar_depth_imgs[:, :half_cam]
            other_view_gt_emernerf_depth = target_emernerf_depth_imgs[:, :half_cam]
            other_view_depth_pred = target_pred_depth_imgs[:, :half_cam]
            
            if training_target_smart_sample: other_view_name = 'aux'
            if add_next_frame_for_eval: other_view_name = 'next'

            # psnr
            other_psnr = (-10 * torch.log10(F.mse_loss(other_view_gt_imgs, other_view_recon_imgs))).item()
            outputs.update({"{}_psnr".format(other_view_name): other_psnr})
            # ssim
            other_ssim_val_lst = []
            for onebatch_gt_imgs, onebatch_pred_imgs in zip(other_view_gt_imgs.cpu().detach().numpy(), other_view_recon_imgs.cpu().detach().numpy()):
                for oneimg_gt_imgs, oneimg_pred_imgs in zip(onebatch_gt_imgs, onebatch_pred_imgs):
                    if self.render_foundation_model_feat:       # foundation model targets
                        other_ssim_val_lst.append(ssim(oneimg_gt_imgs, oneimg_pred_imgs, data_range=1.0, channel_axis=-1))
                    else:
                        other_ssim_val_lst.append(ssim(oneimg_gt_imgs, oneimg_pred_imgs, data_range=1.0, channel_axis=0))
            outputs.update({"{}_ssim".format(other_view_name): sum(other_ssim_val_lst) / len(other_ssim_val_lst)})
            # lidar depth error
            other_target_lidar_abs_rel, other_target_lidar_sq_rel, other_target_lidar_rmse, other_target_lidar_rmse_log, other_target_lidar_a1, other_target_lidar_a2, other_target_lidar_a3 \
                        = self.compute_depth_errors(other_view_gt_lidar_depth, other_view_depth_pred)
            # emerenrf depth error
            other_target_emernerf_abs_rel, other_target_emernerf_sq_rel, other_target_emernerf_rmse, other_target_emernerf_rmse_log, other_target_emernerf_a1, other_target_emernerf_a2, other_target_emernerf_a3 \
                        = self.compute_depth_errors(other_view_gt_emernerf_depth, other_view_depth_pred)
            # emernerf inner depth error
            other_target_inner_emernerf_abs_rel, other_target_inner_emernerf_sq_rel, other_target_inner_emernerf_rmse, other_target_inner_emernerf_rmse_log, other_target_inner_emernerf_a1, other_target_inner_emernerf_a2, other_target_inner_emernerf_a3 \
                        = self.compute_depth_errors(other_view_gt_emernerf_depth, other_view_depth_pred, 0.001, self.inner_range)
            
            # simplified depth error log during training
            outputs.update(
                {
                    "{}_target_depth_lidar_error_abs_rel".format(other_view_name): other_target_lidar_abs_rel,
                    "{}_target_depth_emernerf_error_abs_rel".format(other_view_name): other_target_emernerf_abs_rel,
                    "{}_target_depth_emernerf_inner_error_abs_rel".format(other_view_name): other_target_inner_emernerf_abs_rel
                }
            )
            # complete depth error log during evaluation
            if is_val:
                # lidar depth error
                outputs.update(
                    {
                        "{}_target_lidar_abs_rel".format(other_view_name): other_target_lidar_abs_rel,
                        "{}_target_lidar_sq_rel".format(other_view_name): other_target_lidar_sq_rel,
                        "{}_target_lidar_rmse".format(other_view_name): other_target_lidar_rmse,
                        "{}_target_lidar_rmse_log".format(other_view_name): other_target_lidar_rmse_log,
                        "{}_target_lidar_a1".format(other_view_name): other_target_lidar_a1,
                        "{}_target_lidar_a2".format(other_view_name): other_target_lidar_a2,
                        "{}_target_lidar_a3".format(other_view_name): other_target_lidar_a3
                    }
                )
                # emernerf depth error
                outputs.update(
                        {
                            "{}_target_emernerf_abs_rel".format(other_view_name): other_target_emernerf_abs_rel,
                            "{}_target_emernerf_sq_rel".format(other_view_name): other_target_emernerf_sq_rel,
                            "{}_target_emernerf_rmse".format(other_view_name): other_target_emernerf_rmse,
                            "{}_target_emernerf_rmse_log".format(other_view_name): other_target_emernerf_rmse_log,
                            "{}_target_emernerf_a1".format(other_view_name): other_target_emernerf_a1,
                            "{}_target_emernerf_a2".format(other_view_name): other_target_emernerf_a2,
                            "{}_target_emernerf_a3".format(other_view_name): other_target_emernerf_a3
                        }
                    )
                # emernerf depth error for inner voxel
                outputs.update(
                        {
                            "{}_target_emernerf_inner_abs_rel".format(other_view_name): other_target_inner_emernerf_abs_rel,
                            "{}_target_emernerf_inner_sq_rel".format(other_view_name): other_target_inner_emernerf_sq_rel,
                            "{}_target_emernerf_inner_rmse".format(other_view_name): other_target_inner_emernerf_rmse,
                            "{}_target_emernerf_inner_rmse_log".format(other_view_name): other_target_inner_emernerf_rmse_log,
                            "{}_target_emernerf_inner_a1".format(other_view_name): other_target_inner_emernerf_a1,
                            "{}_target_emernerf_inner_a2".format(other_view_name): other_target_inner_emernerf_a2,
                            "{}_target_emernerf_inner_a3".format(other_view_name): other_target_inner_emernerf_a3
                        }
                    )
                
        return outputs
            

    def image_log(self, outputs, scene_data, input_gt_lidar_depth_imgs, input_emernerf_depth_imgs, input_coarse_depth_pred, 
                         recon_imgs, target_pred_depth_imgs, target_gt_lidar_depth_imgs, target_emernerf_depth_imgs):
        ''' save input coarse depth images'''
        outputs.update(
            {
                "input_imgs": scene_data.get("imgs")[0].clamp(0, 1),
                "input_gt_lidar_depth_imgs": (torch.nan_to_num(input_gt_lidar_depth_imgs, 0) / self.depth_bounds[1]).clamp(0, 1),
                "input_emernerf_depth_imgs": (torch.nan_to_num(input_emernerf_depth_imgs, 0) / self.depth_bounds[1]).clamp(0, 1),
                "input_coarse_depth_pred_imgs": (input_coarse_depth_pred.detach() / self.depth_bounds[1]).clamp(0, 1), 
             
             
                "target_imgs": scene_data.get("target_imgs")[0].clamp(0, 1),
                "recon_imgs": recon_imgs[0].clamp(0, 1),
                "target_gt_lidar_depth_imgs": (torch.nan_to_num(target_gt_lidar_depth_imgs, 0) / self.depth_bounds[1]).clamp(0, 1),
                "target_emernerf_depth_imgs": (torch.nan_to_num(target_emernerf_depth_imgs, 0) / self.depth_bounds[1]).clamp(0, 1),
                "target_pred_depth_imgs": (torch.nan_to_num(target_pred_depth_imgs.detach(), 0) / self.depth_bounds[1]).clamp(0,1),
            }
        )

        return outputs
    
    def octree_log(self, outputs, intermediates):
        # octree
        outputs.update({"octree_point_hierarchies7": intermediates.get("octree_point_hierarchy7")})      
        outputs.update({"octree_pyramid7": intermediates.get("octree_pyramid7")})
        outputs.update({"octree_feats7": intermediates.get("octree_feats7")})
        outputs.update({"octree_point_hierarchies9": intermediates.get("octree_point_hierarchy9")})      
        outputs.update({"octree_pyramid9": intermediates.get("octree_pyramid9")})
        outputs.update({"octree_feats9": intermediates.get("octree_feats9")})

        outputs.update({"octree9": intermediates.get("octree9")})
        outputs.update({"octree_prefix9": intermediates.get("octree_prefix9")})
        outputs.update({"octree7": intermediates.get("octree7")})
        outputs.update({"octree_prefix7": intermediates.get("octree_prefix7")})
    
    def check_gt_depth_distribution(self, input_gt_lidar_depth_imgs, input_emernerf_depth_imgs, target_gt_lidar_depth_imgs):
        ''' calculating the gt depth distribution '''
        rounded_tensor = torch.round(input_gt_lidar_depth_imgs)  # Similar to rounding in the NumPy example
        unique_values, counts = np.unique(rounded_tensor.cpu().numpy(), return_counts=True)
        frequency = dict(zip(unique_values, counts))
        print(frequency)

        rounded_tensor = torch.round(input_emernerf_depth_imgs)  # Similar to rounding in the NumPy example
        unique_values, counts = np.unique(rounded_tensor.cpu().numpy(), return_counts=True)
        frequency = dict(zip(unique_values, counts))
        print(frequency)
        
        rounded_tensor = torch.round(target_gt_lidar_depth_imgs)  # Similar to rounding in the NumPy example
        unique_values, counts = np.unique(rounded_tensor.cpu().numpy(), return_counts=True)
        frequency = dict(zip(unique_values, counts))
        print(frequency)

    def evaluate_depth_error(self, input_coarse_depth_pred, target_pred_depth_imgs, scene_data, outputs, N_cams=6):
        
        ''' input depth error '''
        # input gt lidar depth 
        lidar_depths = scene_data.get("lidar_depths")
        lidar_depth_loc2ds = scene_data.get("lidar_depth_loc2ds")
        lidar_depth_masks = scene_data.get("lidar_depth_masks")
        # interpolate depth prediction
        pred_depth_infer = F.grid_sample(   # N, 1, 1, n ---reshape---> N, n
            input_coarse_depth_pred.permute(1,0,2,3),      # N, 1, H, W
            lidar_depth_loc2ds.squeeze(2).permute(1,0,2,3) * 2 - 1,    # N, 1, n, 2
            mode='bilinear', 
            padding_mode='border', 
            align_corners=True).reshape(N_cams, -1)
        # calculate error
        input_cams_depth_errors = []
        for cam in range(N_cams):
            gt_depth_infer = lidar_depths[0, cam, 0][lidar_depth_masks[0, cam, 0]]
            tmp_depth_infer = pred_depth_infer[cam][lidar_depth_masks[0, cam, 0]]
            assert len(tmp_depth_infer) == len(gt_depth_infer)
            depth_errors = self.compute_depth_errors(gt_depth_infer, tmp_depth_infer)
            input_cams_depth_errors.append(np.array(depth_errors))
        input_depth_error = np.stack(input_cams_depth_errors).mean(0)
        outputs.update(
            {
                "input_depth_error": input_depth_error
            }
        )


        ''' target depth error '''
        # target gt lidar depth 
        target_lidar_depths = scene_data.get("target_lidar_depths")
        target_lidar_depth_loc2ds = scene_data.get("target_lidar_depth_loc2ds")
        target_lidar_depth_masks = scene_data.get("target_lidar_depth_masks")
        target_pred_depth_imgs_cur_pose = target_pred_depth_imgs[0, N_cams:]    # N, 1, H, W
        target_lidar_depths_cur_pose = target_lidar_depths[0, N_cams:]    # N, 1, n, 2
        target_lidar_depth_loc2ds_cur_pose = target_lidar_depth_loc2ds[0, N_cams:]    # N, 1, n, 2
        target_lidar_depth_masks_cur_pose = target_lidar_depth_masks[0, N_cams:]    # N, 1, n, 2
        # interpolate depth prediction
        target_pred_depth_infer = F.grid_sample(   # N, 1, 1, n ---reshape---> N, n
            target_pred_depth_imgs_cur_pose,      # N, 1, H, W
            target_lidar_depth_loc2ds_cur_pose * 2 - 1,    # N, 1, n, 2
            mode='bilinear', 
            padding_mode='border', 
            align_corners=True).reshape(N_cams, -1)
        # calculate error
        target_cams_depth_errors = []
        for cam in range(N_cams):
            gt_depth_infer = target_lidar_depths_cur_pose[cam, 0][target_lidar_depth_masks_cur_pose[cam, 0]]
            tmp_depth_infer = target_pred_depth_infer[cam][target_lidar_depth_masks_cur_pose[cam, 0]]
            assert len(tmp_depth_infer) == len(gt_depth_infer)
            depth_errors = self.compute_depth_errors(gt_depth_infer, tmp_depth_infer)
            target_cams_depth_errors.append(np.array(depth_errors))
        target_depth_error = np.stack(target_cams_depth_errors).mean(0)
        outputs.update(
            {
                "target_depth_error": target_depth_error
            }
        )

        return outputs

    def forward_train(self, imgs: torch.Tensor, img_metas, force_same_seq=False, is_val=False, all_prev_frames=False, \
            training_target_smart_sample=False, add_next_frame_for_eval=False, iteration=None, virtual_img_target_smart_sample=False, **kwargs):
        ''' create dimensions and buffers '''
        super(DistillNerfModelWrapper, self).forward_train(imgs, img_metas, **kwargs)
        B, _, C, H, W = imgs.shape
        N_cams = self.num_camera
        intermediates = Container()
        outputs = dict()

        ''' prepare data'''
        scene_data = self.prepare_data_in_scene_format(imgs, N_cams, is_val=is_val, force_same_seq=force_same_seq, all_prev_frames=all_prev_frames, \
                    training_target_smart_sample=training_target_smart_sample, add_next_frame_for_eval=add_next_frame_for_eval, \
                    virtual_img_target_smart_sample=virtual_img_target_smart_sample, **kwargs)
        
        ''' run the DistillNerf model '''
        intermediates = self.model(scene_data)

        ''' organize gt and predictions '''
        # 1. input depth
        # # input foundation features
        # if self.render_foundation_model_feat:       # foundation model targets
        #     input_fm_feat = rearrange(scene_data.get("fm_feat"), 'b n h w c -> (b n) c h w')
        #     input_fm_feat = F.interpolate(input_fm_feat, (recon_imgs.shape[2], recon_imgs.shape[3]), mode='bilinear')   # # [b, n, 1, w, h]
        #     input_fm_feat_imgs = rearrange(input_fm_feat, '(b n) c h w -> b n h w c', b=B)
        # input gt lidar depth
        input_gt_lidar_depth_imgs = scene_data.get("depths").squeeze(2)                                 # [b, n, h, w]
        input_gt_lidar_depth_imgs = F.interpolate(input_gt_lidar_depth_imgs, (H, W), mode='bilinear')   # [b, n, h, w]
        # input emernerf depth
        input_emernerf_depth_imgs = scene_data.get("emernerf_depth_img")                                # [b, n, 1, h, w]
        input_emernerf_depth_imgs = F.interpolate(input_emernerf_depth_imgs, (H, W), mode='bilinear')   # [b, n, h, w]

        # 2. coarse depth 
        # coarse depth prediction
        input_coarse_depth_pred = intermediates.coarse_depth
        input_coarse_depth_pred = rearrange(input_coarse_depth_pred, '(b n) h w -> b n h w', b=B)       # [b, n, h, w]
        # coarse depth samples
        input_coarse_depth_samples = intermediates.coarse_depth_samples
        # coarse weight prediction
        input_coarse_weight_pred = intermediates.coarse_depth_weights 

        # 2. fine depth
        # fine depth prediction
        input_fine_depth_pred = intermediates.fine_depth
        input_fine_depth_pred = rearrange(input_fine_depth_pred, '(b n) h w -> b n h w', b=B)           # [b, n, h, w]
        # fine depth samples
        input_fine_depth_samples = intermediates.fine_depth_candidates_mid
        # fine weight prediction
        input_fine_weights_pred = intermediates.fine_depth_candidates_mid_weights

        # 3. target image/depth
        # target rendered img
        if self.render_foundation_model_feat:       # foundation model targets
            recon_imgs = intermediates.get("recons")
        else:                                   # rgb image
            recon_imgs = intermediates.get("recons")
            recon_imgs = rearrange(recon_imgs, '(b n) c h w -> b n c h w', b=B, c=3)
            if self.rgb_clamp_0_1: recon_imgs = torch.clamp(recon_imgs, 0, 1)
        # target rendered target depth
        target_pred_depth_imgs = intermediates.get("target_pred_depth_image")
        target_pred_depth_imgs = rearrange(target_pred_depth_imgs, '(b n) 1 h w -> b n 1 h w', b=B)
        # target depth samples
        target_depth_samples = intermediates.get("target_sample_depths_all")
        # target weight prediction
        target_weight_pred = intermediates.get("target_weights_all")
        # target gt img
        if self.render_foundation_model_feat:       # foundation model targets
            target_fm_feat = rearrange(scene_data.get("target_fm_feat"), 'b n h w c -> (b n) c h w')
            target_fm_feat = F.interpolate(target_fm_feat, (recon_imgs.shape[2], recon_imgs.shape[3]), mode='bilinear')   # # [b, n, 1, w, h]
            gt_imgs = rearrange(target_fm_feat, '(b n) c h w -> b n h w c', b=B)
        else:
            gt_imgs = scene_data.get("target_imgs")
        # target gt lidar depth
        target_gt_lidar_depth_imgs = scene_data.get("target_depth_imgs")  # [b, n, 1, w, h]
        # target emernerf depth
        H_pred_depth, W_pred_depth = target_pred_depth_imgs.shape[-2], target_pred_depth_imgs.shape[-1]
        target_emernerf_depth_imgs = F.interpolate(scene_data.get("target_emernerf_depth_img"), (H_pred_depth, W_pred_depth), mode='bilinear').unsqueeze(2)   # # [b, n, 1, w, h]
        

        ''' loss calculation '''
        # 1. appearance loss
        outputs = self.appearance_loss_cal(outputs, recon_imgs, gt_imgs)
            
        # 2. lidar depth loss
        outputs = self.depth_lidar_loss_cal(outputs, input_gt_lidar_depth_imgs, input_coarse_depth_pred, \
                                input_fine_depth_pred, target_gt_lidar_depth_imgs, target_pred_depth_imgs)

        # 3. emernerf depth loss
        outputs = self.depth_emernerf_loss_cal(outputs, input_emernerf_depth_imgs, input_coarse_depth_pred, \
                                input_fine_depth_pred, target_emernerf_depth_imgs, target_pred_depth_imgs)
        
        # 4. pretrained depth loss (Optional)
        # outputs = self.pretrained_depth_loss_cal(outputs, intermediates, gt=None, \
        #                         input_coarse_depth_pred=input_coarse_depth_pred, target_pred_depth_imgs=target_pred_depth_imgs)

        # 5. weight entropy loss
        outputs = self.weight_entropy_loss_cal(outputs, input_coarse_weight_pred, target_weight_pred)

        # 6. opacity loss
        outputs = self.opacity_loss_cal(outputs, input_coarse_weight_pred, input_fine_weights_pred, target_weight_pred)

        # 7. line of sight loss (Optional)
        outputs = self.line_of_sight_loss_cal(outputs, iteration, input_gt_lidar_depth_imgs, input_emernerf_depth_imgs, \
                    input_coarse_weight_pred, input_coarse_depth_samples, input_fine_weights_pred, input_fine_depth_samples, \
                    target_gt_lidar_depth_imgs, target_emernerf_depth_imgs, target_weight_pred, target_depth_samples
                    )
        
        # 8. depth edge loss (Optional)
        if self.enable_edge_loss:
            if (iteration is not None and iteration >= self.edge_loss_start_iter) or iteration is None:
                outputs = self.edge_loss(intermediates, outputs, iteration, 
                    input_gt_lidar_depth_imgs, input_emernerf_depth_imgs, 
                    input_coarse_depth_pred, input_coarse_weight_pred, input_coarse_depth_samples,
                    input_fine_depth_pred, input_fine_weights_pred, input_fine_depth_samples,
                    target_gt_lidar_depth_imgs, target_emernerf_depth_imgs, 
                    target_pred_depth_imgs, target_weight_pred, target_depth_samples
                )

        # 9. Object Detection Head (Optional)
        if self.with_pts_bbox:
            scene_data.set("get_coord_level", self.dense_voxel_level)
            self.detection_model(intermediates, is_val, kwargs, img_metas, outputs)
       
        # 10. EmerNerf Regularization Loss (TODO: on feature space, maybe rgb space is more direct?)
        if self.use_emernerf_reg:
            self.emernerf_regularization(kwargs, outputs)


        ''' eval and log '''
        # image recoloring - a baseline in Waymo
        if self.recoloring_img:
            recon_imgs = self.recoloring(recon_imgs, gt_imgs, N_cams)

        # input depth
        self.input_depth_evaluation_cal(outputs, is_val, input_gt_lidar_depth_imgs, input_emernerf_depth_imgs, input_fine_depth_pred)

        # target image, depth
        if not self.novel_view_mode:
            self.target_evaluation_cal(outputs, is_val, gt_imgs, recon_imgs, target_gt_lidar_depth_imgs, target_emernerf_depth_imgs, target_pred_depth_imgs, \
                                  training_target_smart_sample, add_next_frame_for_eval)
        
        # save depth/rgb image
        self.image_log(outputs, scene_data, input_gt_lidar_depth_imgs, input_emernerf_depth_imgs, input_coarse_depth_pred, 
                     recon_imgs, target_pred_depth_imgs, target_gt_lidar_depth_imgs, target_emernerf_depth_imgs)
        
        # log the octree
        self.octree_log(outputs, intermediates)

        # check gt depth distribution
        # self.check_gt_depth_distribution(self, input_gt_lidar_depth_imgs, input_emernerf_depth_imgs, target_gt_lidar_depth_imgs)

        # query/visualize the 3D voxel
        if self.visualize_voxels:
            self.sample_from_octree_wrapper(intermediates, scene_data)


        # visualization
        # import pdb; pdb.set_trace()
        if self.visualize_imgs:
            self.visualize_img_depth(scene_data, intermediates, img_metas, add_next_frame_for_eval, novel_view=self.novel_view_mode)

        # a different way of depth evaluation
        # self.evaluate_depth_error(input_coarse_depth_pred, target_pred_depth_imgs, scene_data, outputs, N_cams=N_cams)

        # visualize foundation model feature images
        if self.render_foundation_model_feat and self.visualize_foundation_model_feat:
            if add_next_frame_for_eval:
                visualize_foundation_feat(recon_imgs[:, N_cams:], novel_view=self.novel_view_mode, feat=self.render_foundation_model_feat, directory=self.vis_save_directory)
            else:
                visualize_foundation_feat(recon_imgs, novel_view=self.novel_view_mode, feat=self.render_foundation_model_feat, directory=self.vis_save_directory)


        # open vocabulary query
        if self.render_foundation_model_feat and self.language_query:
            if add_next_frame_for_eval:
                language_query(recon_imgs[:, N_cams:], directory=self.vis_save_directory)
            else:
                language_query(recon_imgs, directory=self.vis_save_directory)

        # import pdb; pdb.set_trace()
        return outputs


    def simple_test(self, imgs, img_metas, **kwargs):
        with torch.no_grad():
            if "novel_view_traj" in kwargs.keys():
                outputs = self.forward_train(imgs, img_metas, force_same_seq=True, is_val=True, **kwargs)
            else:
                outputs = self.forward_train(imgs, img_metas, force_same_seq=True, is_val=True, **kwargs)
                # outputs = self.forward_train(imgs, img_metas, force_same_seq=True, is_val=True, add_next_frame_for_eval=True, **kwargs)
            results = {}
            
            for key, val in outputs.items():
                if 'is_test' in kwargs.keys():
                    ''' option 1: for testing only, save memory '''
                    key_word_lst = ['ssim', 'psnr', 'abs_rel', 'sq_rel', 'rmse', 'rmse_log', 'a1', 'a2', 'a3']
                    if any([key_word in key for key_word in key_word_lst]):
                        results[key] = val

                else:
                    ''' option 2: normal eval - save most stuff '''
                    if 'octree' in key:         # skip octree to save memory
                        continue
                    if '3d' in key:             # ignore detection log
                        results[key] = val
                    elif not 'img' in key:
                        key_word_lst = ['ssim', 'psnr', 'abs_rel', 'sq_rel', 'rmse', 'rmse_log', 'a1', 'a2', 'a3', 'error', 'target_depth_error', 'input_depth_error']
                        if any([key_word in key for key_word in key_word_lst]):     # save metrics
                            results[key] = val
                        else:
                            try:
                                results[key] = val.item()
                            except:
                                import pdb; pdb.set_trace()
                    else:
                        # images
                        results[key] = val
    
            return [results]

    def aug_test(self, imgs, img_metas, **kwargs):
        """Test function with test time augmentation."""
        raise NotImplementedError()
