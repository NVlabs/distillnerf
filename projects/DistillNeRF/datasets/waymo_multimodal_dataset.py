import os
import tempfile
from os import path as osp
import math
import mmcv
import numpy as np
import torch
from mmcv.utils import print_log
import pdb

from mmdet3d.datasets import DATASETS, WaymoDataset

from collections import defaultdict
from pyquaternion import Quaternion



@DATASETS.register_module()
class WaymoMultiModalDatasetV2(WaymoDataset):
    def __init__(self,
                 data_root,
                 ann_file,
                 split,
                 pts_prefix='velodyne',
                 pipeline=None,
                 classes=None,
                 modality=None,
                 box_type_3d='LiDAR',
                 filter_empty_gt=True,
                 test_mode=False,
                 load_interval=1,
                 pcd_limit_range=[-85, -85, -5, 85, 85, 5],
                 # below are customized args
                 subselect_camera_num=-1,        # -1 here means to use all cameras, no subselection. 
                 subselect_random=False,
                 num_prev_frames=0,
                 num_next_frames=0,
                 subselect_group_num=0,         # when subselect_group_num > 0 and subselect_camera_num = -1, subselect_camera_num does nothing
                                                # subselect_group_num controls the number of camera used
                 **kwargs):
        
        super().__init__(
            data_root=data_root,
            ann_file=ann_file,
            split=split,
            pts_prefix=pts_prefix,
            pipeline=pipeline,
            classes=classes,
            modality=modality,
            box_type_3d=box_type_3d,
            filter_empty_gt=filter_empty_gt,
            test_mode=test_mode,
            pcd_limit_range=pcd_limit_range,
            **kwargs)

        self.subselect_camera_num = subselect_camera_num
        self.subselect_random = subselect_random
        self.num_prev_frames = num_prev_frames
        self.num_next_frames = num_next_frames
        self.timestamp2idx = {}
        for idx, info in enumerate(self.data_infos):
            timestamp = math.floor(info['timestamp']/1e5)
            if timestamp not in self.timestamp2idx:
                self.timestamp2idx[timestamp] = idx
            else:
                # two adjacent timesteps with less than 1 typical sensor frequency
                if self.timestamp2idx[timestamp] == idx - 1:
                    self.timestamp2idx[timestamp+1] = idx
                else:
                    print(self.timestamp2idx[timestamp], idx)
                    import pdb; pdb.set_trace()

        self.subselect_group_num = subselect_group_num
        if self.subselect_group_num > 0:
            # cam info is given in the order of:
            # 0: front, 1: front_left, 2: front_right, 3: side_left, 4: side_right
            # following previous works, we use 3 cameras by defaultdefault clock-wise view indices:
            cam_indices = [1, 0, 2]

            self.groups = {}
            for group_ind in range(len(cam_indices)):
                start_view_idx = group_ind
                end_view_idx = group_ind + self.subselect_group_num
                if end_view_idx > len(cam_indices):
                    overflow_idx = end_view_idx - len(cam_indices)
                    self.groups[group_ind] = cam_indices[start_view_idx:] + cam_indices[:overflow_idx]
                else:
                    self.groups[group_ind] = cam_indices[start_view_idx: end_view_idx]


    def get_data_info(self, index):
        """Get data info according to the given index.

        Args:
            index (int): Index of the sample data to get.

        Returns:
            dict: Standard input_dict consists of the
                data information.

                - sample_idx (str): sample index
                - pts_filename (str): filename of point clouds
                - img_prefix (str): prefix of image files
                - img_info (dict): image info
                - lidar2img (list[np.ndarray], optional): transformations from
                    lidar to different cameras
                - ann_info (dict): annotation info
        """
        ''' TODO: second sample of one scene '''
        info = self.data_infos[index]

        ''' choose the order of camera views '''
        selected_idxs = self.groups[0]

        ''' create input_dict'''
        sample_idx = info['image']['image_idx']
        pts_filename = self._get_pts_filename(sample_idx)
        input_dict = dict(
            data_idx=index,
            sample_idx=sample_idx,
            pts_filename=pts_filename,
            #  change here to replace ground truth lidar points with pseudo lidar points
            # pts_filename=info["lidar_path"].replace('nuscenes/samples', "nuscenes_virtual_highres/samples_0_0_0_0_0_0_224_400_1").replace('LIDAR_TOP/', 'pseudo_lidar/LIDAR_TOP/'),
            timestamp=info["timestamp"] / 1e5,
            subselect_camera_num=len(selected_idxs),
        )

        ''' multiple frames '''
        def get_camera_images(info, selected_idxs, anchor_lidar2global_inv=None):

            rect = info['calib']['R0_rect'].astype(np.float32)

            image_paths = []
            intrinsics = []
            extrinsics = []
            cam2globals = []
            lidar2img = []

            for idx in selected_idxs:
                image_paths.append(self.data_root + info['image']['image_path'][idx])

                Tr_c2i = info['calib']['P{}'.format(idx)].astype(np.float32)
                intrinsics.append(Tr_c2i[:3, :3])

                
                # in waymo, the vehicle frame and the lidar frame is the same 
                Tr_v2c = info['calib']['Tr_velo_to_cam_{}'.format(idx)].astype(np.float32)
                Tr_c2v = np.linalg.inv(Tr_v2c)
                Tr_e2g = info['pose']
                if anchor_lidar2global_inv is not None:
                    Tr_c2v = anchor_lidar2global_inv @ Tr_e2g @ Tr_c2v
                extrinsics.append(Tr_c2v.astype(np.float32))
                # import pdb; pdb.set_trace()

                cam2globals.append(Tr_e2g @ Tr_c2v)

                lidar2img.append(Tr_c2i @ rect @ Tr_v2c)

            if anchor_lidar2global_inv is None:
                return [image_paths, intrinsics, extrinsics, cam2globals, lidar2img, selected_idxs], np.linalg.inv(Tr_e2g)
            else:
                return [image_paths, intrinsics, extrinsics, cam2globals, lidar2img, selected_idxs]

            ''' single frame '''
            # num_cam = len(info['image']['image_path'])
            # sample_idx = info['image']['image_idx']
            # img_filename = [os.path.join(self.data_root,
                # one_image_path) for one_image_path in info['image']['image_path']]
                    # 
            # intrinsic = [info['calib']['P{}'.format(i)].astype(np.float32) for i in range(num_cam)]
            # extrinsic = [info['calib']['Tr_velo_to_cam_{}'.format(i)].astype(np.float32) for i in range(num_cam)]

            # rect = info['calib']['R0_rect'].astype(np.float32)
            # lidar2img = [intrinsic[i] @ rect @ extrinsic[i] for i in range(num_cam)]
            # import pdb; pdb.set_trace()
            
            ''' original code in mmdetection3d for waymo '''
            # rect = info['calib']['R0_rect'].astype(np.float32)
            # Trv2c = info['calib']['Tr_velo_to_cam_0'].astype(np.float32)
            # P0 = info['calib']['P0'].astype(np.float32)
            # lidar2img = P0 @ rect @ Trv2c

        ''' collect current-frame data '''
        current_data, anchor_lidar2global_inv = get_camera_images(info, selected_idxs)

        ''' collect previous-frame data '''
        if self.num_prev_frames > 0:
            prev_index = index - 1
            prev_datas = []
            while len(prev_datas) < self.num_prev_frames:
                if prev_index < 0:
                    prev_datas.append(current_data)
                    # print("prev_index < 0")
                    # import pdb; pdb.set_trace()
                elif abs(self.data_infos[prev_index]["timestamp"] / 1e5 - input_dict["timestamp"]) > self.num_prev_frames+2:
                    # a hard-coded threshold for the time gap between two frames in the same scene
                    prev_datas.append(current_data)
                    # print("prev time gap")
                    # import pdb; pdb.set_trace()
                else:
                    prev_ret = get_camera_images(self.data_infos[prev_index], selected_idxs, anchor_lidar2global_inv=anchor_lidar2global_inv)
                    prev_datas.append(prev_ret)       
                    prev_index = prev_index - 1
                    # print("prev normal")
                    # import pdb; pdb.set_trace()
            prev_datas = prev_datas[::-1]
        else:
            prev_datas = []

        ''' collect next-frame data '''
        if self.num_next_frames > 0:
            next_index = index + 1
            next_datas = []
            while len(next_datas) < self.num_next_frames:
                if next_index >= len(self.data_infos):
                    next_datas.append(current_data)
                    # print("next over index")
                    # import pdb; pdb.set_trace()
                elif abs(self.data_infos[next_index]["timestamp"] / 1e5 - input_dict["timestamp"]) > self.num_next_frames+2:
                    # a hard-coded threshold for the time gap between two frames in the same scene
                    next_datas.append(current_data)
                    # print("next time gap")
                    # import pdb; pdb.set_trace()
                else:
                    next_ret = get_camera_images(self.data_infos[next_index], selected_idxs, anchor_lidar2global_inv=anchor_lidar2global_inv)
                    next_datas.append(next_ret)     
                    next_index = next_index + 1
                    # print("next normal")
                    # import pdb; pdb.set_trace()

        else:
            next_datas = []


        ''' combine data from different frame, and organize them '''
        datas = prev_datas + [current_data] + next_datas
        image_paths = []
        intrinsics = []
        extrinsics = []
        cam2globals = []
        lidar2imgs = []
        selected_idxs = []
        for data in datas:
            image_paths.extend(data[0])
            intrinsics.extend(data[1])
            extrinsics.extend(data[2])
            cam2globals.extend(data[3])
            lidar2imgs.extend(data[4])
            selected_idxs.extend(data[5])

        input_dict.update(
            dict(
                img_prefix=None,
                img_filename=image_paths,
                intrinsic=intrinsics,
                extrinsic=extrinsics,
                cam2global=cam2globals,
                lidar2img=lidar2imgs,
                num_cams=len(selected_idxs),
            )
        )

        # input_dict = dict(
        #     # data_idx=index,
        #     # sample_idx=sample_idx,
        #     # pts_filename=pts_filename,
        #     # timestamp=info['timestamp'] / 1e5,
        #     img_prefix=None,
        #     # img_info=dict(filename=img_filename),
        #     img_filename=img_filename,  # list of camera images paths
        #     intrinsic=info['calib']['P2'].astype(np.float32),
        #     extrinsic=info['calib']['Tr_velo_to_cam'].astype(np.float32),
        #     lidar2img=lidar2img,
        #     )

        if not self.test_mode:
            info['calib']['Tr_velo_to_cam'] = info['calib']['Tr_velo_to_cam_0']
            annos = self.get_ann_info(index)
            input_dict['ann_info'] = annos

        return input_dict

    def tempt(self):
        dict_keys([
        # 'pre_eval'
        # 'sweeps'

        # 'data_idx'
        # 'sample_idx'
        # 'pts_filename'
        # 'timestamp'
        # 'subselect_camera_num'
        # 'img_filename'
        'intrinsic'
        'extrinsic'
        'cam2global'
        'lidar2img'
        # 'num_cams'
        ])


    def evaluate(
        self,
        results,
        metric="bbox",
        logger=None,
        jsonfile_prefix=None,
        result_names=["pts_bbox"],
        show=False,
        out_dir=None,
        pipeline=None,
    ):
        assert isinstance(
            results, list
        ), f"Expect results to be list, got {type(results)}."
        assert len(results) > 0, "Expect length of results > 0."
        assert len(results) == len(self.data_infos)
        assert isinstance(
            results[0], dict
        ), f"Expect elements in results to be dict, got {type(results[0])}."

        load_pipeline = self._get_pipeline(pipeline)

        # Need to pick one GPU that will hold the eval data.
        inference_device = "cuda:0"
        lpips_loss = LPIPSLoss().to(inference_device)
        outputs = defaultdict(float)


        ''' input '''
        # images
        outputs["input_imgs"] = []
        outputs['input_coarse_depth_pred_imgs'] = []
        outputs['input_gt_lidar_depth_imgs'] = []
        outputs['input_emernerf_depth_imgs'] = []

        # loss / metrics
        outputs['coarse_depth_loss_lidar'] = 0
        outputs['coarse_depth_loss_emernerf'] = 0
        outputs['coarse_nerf_weight_entropy_loss'] = 0
        outputs['coarse_depth_lidar_error_abs_rel'] = 0
        outputs['coarse_depth_emernerf_error_abs_rel'] = 0
        outputs['coarse_depth_emernerf_inner_error_abs_rel'] = 0


        ''' target '''
        # images
        outputs['target_imgs'] = []
        outputs['recon_imgs'] = []
        outputs['target_pred_depth_imgs'] = []
        outputs['target_gt_lidar_depth_imgs'] = []
        outputs['target_emernerf_depth_imgs'] = []
        
        # loss / metrics
        outputs['target_depth_loss_lidar'] = 0
        outputs['target_depth_loss_emernerf'] = 0
        outputs['target_nerf_weight_entropy_loss'] = 0

        outputs['rgb_l1_loss'] = 0
        outputs['lpips_loss'] = 0
            
        outputs['target_depth_lidar_error_abs_rel'] = 0
        outputs['target_depth_emernerf_error_abs_rel'] = 0
        outputs['target_depth_emernerf_inner_error_abs_rel'] = 0
        outputs['psnr'] = 0
        outputs['ssim'] = 0


        ''' novel-view - next pose '''
        has_next_pose = 'next_psnr' in results[0].keys()
        if has_next_pose:
            outputs['next_target_depth_lidar_error_abs_rel'] = 0
            outputs['next_target_depth_emernerf_error_abs_rel'] = 0
            outputs['next_target_depth_emernerf_inner_error_abs_rel'] = 0

            outputs['next_psnr'] = 0
            outputs['next_ssim'] = 0


        for data_index, result in enumerate(results):
            # record images occasionally
            if data_index == 0:
                # input 
                outputs["input_imgs"] = result['input_imgs']
                outputs["input_coarse_depth_pred_imgs"] = result['input_coarse_depth_pred_imgs']
                outputs["input_gt_lidar_depth_imgs"] = result['input_gt_lidar_depth_imgs']
                outputs["input_emernerf_depth_imgs"] = result['input_emernerf_depth_imgs']
                # target 
                outputs["target_imgs"] = result['target_imgs']
                outputs["recon_imgs"] = result['recon_imgs']
                outputs["target_pred_depth_imgs"] = result['target_pred_depth_imgs']
                outputs["target_gt_lidar_depth_imgs"] = result['target_gt_lidar_depth_imgs']
                outputs["target_emernerf_depth_imgs"] = result['target_emernerf_depth_imgs']

            # record losses and metrics
            ''' coarse depth '''
            # depth loss
            outputs['coarse_depth_loss_lidar'] += result['coarse_depth_loss_lidar'] / len(results)
            outputs['coarse_depth_loss_emernerf'] += result['coarse_depth_loss_emernerf'] / len(results)
            outputs['coarse_nerf_weight_entropy_loss'] += result['coarse_nerf_weight_entropy_loss'] / len(results)
            # depth error
            outputs['coarse_depth_lidar_error_abs_rel'] += result['coarse_depth_lidar_error_abs_rel'] / len(results)
            outputs['coarse_depth_emernerf_error_abs_rel'] += result['coarse_depth_emernerf_error_abs_rel'] / len(results)
            outputs['coarse_depth_emernerf_inner_error_abs_rel'] += result['coarse_depth_emernerf_inner_error_abs_rel'] / len(results)

            ''' target depth/image '''
            # depth loss
            outputs['target_depth_loss_lidar'] += result['target_depth_loss_lidar'] / len(results)
            outputs['target_depth_loss_emernerf'] += result['target_depth_loss_emernerf'] / len(results)
            outputs['target_nerf_weight_entropy_loss'] += result['target_nerf_weight_entropy_loss'] / len(results)
            # appearance loss
            outputs['rgb_l1_loss'] += result['rgb_l1_loss'] / len(results)
            if 'lpips_loss' in result: outputs['lpips_loss'] += result['lpips_loss'] / len(results)
            # depth metricss
            if 'target_depth_lidar_error_abs_rel' in result: outputs['target_depth_lidar_error_abs_rel'] += result['target_depth_lidar_error_abs_rel'] / len(results)
            if 'target_depth_emernerf_error_abs_rel' in result: outputs['target_depth_emernerf_error_abs_rel'] += result['target_depth_emernerf_error_abs_rel'] / len(results)
            if 'target_depth_emernerf_inner_error_abs_rel' in result: outputs['target_depth_emernerf_inner_error_abs_rel'] += result['target_depth_emernerf_inner_error_abs_rel'] / len(results)
            # appearance metrics
            if 'psnr' in result: outputs['psnr'] += result['psnr'] / len(results)
            if 'ssim' in result: outputs['ssim'] += result['ssim'] / len(results)

            if has_next_pose:
                outputs['next_target_depth_lidar_error_abs_rel'] += result['next_target_depth_lidar_error_abs_rel'] / len(results)
                outputs['next_target_depth_emernerf_error_abs_rel'] += result['next_target_depth_emernerf_error_abs_rel'] / len(results)
                outputs['next_target_depth_emernerf_inner_error_abs_rel'] += result['next_target_depth_emernerf_inner_error_abs_rel'] / len(results)

                outputs['next_psnr'] += result['next_psnr'] / len(results)
                if "next_ssim" in result: outputs['next_ssim'] += result['next_ssim'] / len(results)

        # Evaluating detections.
        if 'boxes_3d' in results[0].keys():
            results_dict = super().evaluate(
                results=results,
                metric=metric,
                logger=logger,
                jsonfile_prefix=jsonfile_prefix,
                result_names=result_names,
                show=show,
                out_dir=out_dir,
                pipeline=pipeline,
            )

            # Reformatting the keys for easier readability in logging frameworks.
            outputs.update({f"det_{k.split('/')[-1]}": v for k, v in results_dict.items()})

        return dict(outputs)

