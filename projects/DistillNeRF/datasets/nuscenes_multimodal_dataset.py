from collections import defaultdict
from functools import reduce
import operator

import numpy as np
import torch
import torch.nn.functional as F
from mmdet3d.datasets import DATASETS, NuScenesDataset, NuScenesMonoDataset
from pyquaternion import Quaternion

from ..losses import LPIPSLoss


@DATASETS.register_module()
class NuScenesMultiModalDatasetV2(NuScenesDataset):
    def __init__(
        self,
        ann_file_3d,
        ann_file_2d,
        pipeline=None,
        data_root=None,
        classes=None,
        load_interval=1,
        with_velocity=True,
        modality=None,
        box_type_3d="LiDAR",
        filter_empty_gt=True,
        test_mode=False,
        eval_version="detection_cvpr_2019",
        use_valid_flag=False,
        subselect_camera_num=-1,        # -1 here means to use all cameras, no subselection. 
        subselect_random=False,
        pre_eval=False,
        num_prev_frames=0,
        num_next_frames=0,
        subselect_group_num=0,      # when subselect_group_num > 0 and subselect_camera_num = -1, subselect_camera_num does nothing
                                    # subselect_group_num controls the number of camera used
    ):
        self.load_interval = load_interval
        self.use_valid_flag = use_valid_flag
        super().__init__(
            data_root=data_root,
            ann_file=ann_file_3d,
            pipeline=pipeline,
            classes=classes,
            modality=modality,
            box_type_3d=box_type_3d,
            filter_empty_gt=filter_empty_gt,
            test_mode=test_mode,
        )

        self.with_velocity = with_velocity
        self.eval_version = eval_version
        from nuscenes.eval.detection.config import config_factory

        self.eval_detection_configs = config_factory(self.eval_version)
        if self.modality is None:
            self.modality = dict(
                use_camera=False,
                use_lidar=True,
                use_radar=False,
                use_map=False,
                use_external=False,
            )

        self.subselect_camera_num = subselect_camera_num        # usually -1
        self.subselect_random = subselect_random
        self.pre_eval = pre_eval
        self.num_prev_frames = num_prev_frames
        self.num_next_frames = num_next_frames
        self.token2ids = {}
        for idx, info in enumerate(self.data_infos):
            self.token2ids[info['token']] = idx

        if self.subselect_camera_num == 0:
            self.modality["use_camera"] = False

        if self.modality["use_camera"]:
            self.img_dataset = NuScenesMonoDataset(
                data_root=data_root,
                ann_file=ann_file_2d,
                pipeline=pipeline,
                classes=classes,
                modality=modality,
                filter_empty_gt=filter_empty_gt,
                test_mode=test_mode,
            )

            # img_dataset Frame -> Data info index mapping
            self.img_token_idx = {}
            for idx, data_info in enumerate(self.img_dataset.data_infos):
                if data_info["token"] not in self.img_token_idx:
                    self.img_token_idx[data_info["token"]] = idx

            assert len(self.img_token_idx) == len(
                self.data_infos
            ), "Are there frames missing from the associated camera dataset?"

        ''' generate different camera view order, which is named as "group" '''
        self.subselect_group_num = subselect_group_num
        if self.subselect_group_num > 0:
            # in the data, cam info is given in the order of:
            # ['CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT', 'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT']
            cam_indices = [0, 1, 5, 3, 4, 2]    # clock-wise view indices
            self.groups = {}
            for group_ind in range(len(cam_indices)):
                start_view_idx = group_ind
                end_view_idx = group_ind + self.subselect_group_num
                if end_view_idx > len(cam_indices):
                    overflow_idx = end_view_idx - len(cam_indices)
                    self.groups[group_ind] = cam_indices[start_view_idx:] + cam_indices[:overflow_idx]
                else:
                    self.groups[group_ind] = cam_indices[start_view_idx: end_view_idx]
            # different groups starts from different camera view, but are all clock-wise.
            # The 5th group is the default group, starting from the CAM_FRONT_LEFT

    def get_data_info(self, index):
        """Get data info according to the given index.

        Args:
            index (int): Index of the sample data to get.

        Returns:
            dict: Data information that will be passed to the data \
                preprocessing pipelines. It includes the following keys:

                - sample_idx (str): Sample index.
                - pts_filename (str): Filename of point clouds.
                - sweeps (list[dict]): Infos of sweeps.
                - timestamp (float): Sample timestamp.
                - img_filename (str, optional): Image filename.
                - lidar2img (list[np.ndarray], optional): Transformations \
                    from lidar to different cameras.
                - ann_info (dict): Annotation info.
        """
        ''' to be consistent with single-frame EmerNeRF, only evaluate the second sample of one scene '''
        # import pdb; pdb.set_trace()
        if self.test_mode:
            scene_index = index + 1
            scene_count = 0
            index = 0
            while scene_count < scene_index:
                info = self.data_infos[index]
                if info['prev'] == -1 or info['prev'] == '':
                    scene_count += 1
                    if scene_count == scene_index:
                        index += 1
                        info = self.data_infos[index]
                        break
                index += 1

        info = self.data_infos[index]

        ''' create input_dict'''
        input_dict = dict(
            pre_eval=self.pre_eval,
            data_idx=index,
            sample_idx=info["token"],
            pts_filename=info["lidar_path"],
            #  change here to replace ground truth lidar points with pseudo lidar points
            # pts_filename=info["lidar_path"].replace('nuscenes/samples', "nuscenes_virtual_highres/samples_0_0_0_0_0_0_224_400_1").replace('LIDAR_TOP/', 'pseudo_lidar/LIDAR_TOP/'),
            sweeps=info["sweeps"],
            timestamp=info["timestamp"] / 1e6,
        )

        ''' choose the order of camera views '''
        # select camera indices to train on
        cam_list = list(info["cams"].values())
        if self.subselect_group_num > 0:
            # selected_idxs = self.groups[np.random.choice(len(self.groups))]
            selected_idxs = self.groups[5]
            input_dict.update(
                dict(
                    subselect_camera_num=len(selected_idxs),
                )
            )
        elif self.subselect_camera_num > 0:     # usually False
            selected_idxs = (
                np.random.choice(self.subselect_camera_num)
                if self.subselect_random
                else range(self.subselect_camera_num)
            )

            input_dict.update(
                dict(
                    subselect_camera_num=self.subselect_camera_num,
                    subselect_random=self.subselect_random,
                )
            )

        else:
            selected_idxs = range(len(info["cams"]))


        def get_camera_images(info, anchor_lidar2global_inv=None):

            l2e_r = info["lidar2ego_rotation"]
            l2e_t = info["lidar2ego_translation"]
            e2g_r = info["ego2global_rotation"]
            e2g_t = info["ego2global_translation"]
            l2e_r_mat = Quaternion(l2e_r).rotation_matrix
            e2g_r_mat = Quaternion(e2g_r).rotation_matrix

            lidar2ego = np.eye(4)
            lidar2ego[:3, :3] = l2e_r_mat
            lidar2ego[:3, 3] = l2e_t

            ego2global = np.eye(4)
            ego2global[:3, :3] = e2g_r_mat
            ego2global[:3, 3] = e2g_t
            lidar2global = ego2global @ lidar2ego

            image_paths = []
            intrinsics = []
            extrinsics = []
            cam2globals = []
            lidar2img = []
            cam_list = list(info["cams"].values())

            for idx in selected_idxs:
                cam_info = cam_list[idx]
                image_paths.append(cam_info["data_path"])
                intrinsic = cam_info["cam_intrinsic"]
                viewpad = np.eye(4)
                viewpad[: intrinsic.shape[0], : intrinsic.shape[1]] = intrinsic
                # import pdb; pdb.set_trace()
                intrinsics.append(intrinsic.astype(np.float32))
                cam2lidar_rt = np.eye(4)
                cam2lidar_rt[:3, :3] = cam_info["sensor2lidar_rotation"]
                cam2lidar_rt[:3, 3] = cam_info["sensor2lidar_translation"]

                if anchor_lidar2global_inv is not None:
                    # for multi-frames, need to have a anchor frame and convert other frames w.r.t the anchor, to account for ego pose
                    cam2lidar_rt = anchor_lidar2global_inv @ lidar2global @ cam2lidar_rt

                extrinsics.append(cam2lidar_rt.astype(np.float32))

                lidar2cam = np.linalg.inv(cam2lidar_rt)
                lidar2img.append((viewpad @ lidar2cam).astype(np.float32))

                cam2globals.append((lidar2global @ cam2lidar_rt).astype(np.float32))

            if anchor_lidar2global_inv is None:
                return [image_paths, intrinsics, extrinsics, cam2globals, lidar2img, selected_idxs, cam_list], np.linalg.inv(lidar2global)
            else:
                return image_paths, intrinsics, extrinsics, cam2globals, lidar2img, selected_idxs, cam_list
        ''' collect current-frame data '''
        current_ret, anchor_lidar2global_inv = get_camera_images(info)

        ''' collect previous-frame data '''
        if self.num_prev_frames > 0:
            prev_token = info['prev']
            prev_rets = []
            while len(prev_rets) < self.num_prev_frames:
                if prev_token == '':
                    prev_rets.append(current_ret)
                    continue
                prev_info = self.data_infos[self.token2ids[prev_token]]
                prev_token = prev_info['prev']

                prev_ret = get_camera_images(prev_info, anchor_lidar2global_inv=anchor_lidar2global_inv)
                prev_rets.append(prev_ret)
            prev_rets = prev_rets[::-1]
        else:
            prev_rets = []

        ''' collect next-frame data '''
        if self.num_next_frames > 0:
            next_token = info['next']
            next_rets = []
            while len(next_rets) < self.num_next_frames:
                if next_token == '':
                    next_rets.append(current_ret)
                    continue
                next_info = self.data_infos[self.token2ids[next_token]]
                next_token = next_info['next']
                next_ret = get_camera_images(next_info, anchor_lidar2global_inv=anchor_lidar2global_inv)
                next_rets.append(next_ret)
        else:
            next_rets = []
            

        ''' combine data from different frame, and organize them '''
        rets = prev_rets + [current_ret] + next_rets

        image_paths = []
        intrinsics = []
        extrinsics = []
        cam2globals = []
        lidar2imgs = []
        selected_idxs = []
        for ret in rets:
            image_paths.extend(ret[0])
            intrinsics.extend(ret[1])
            extrinsics.extend(ret[2])
            cam2globals.extend(ret[3])
            lidar2imgs.extend(ret[4])
            selected_idxs.extend(ret[5])

        input_dict.update(
            dict(
                img_filename=image_paths,
                intrinsic=intrinsics,
                extrinsic=extrinsics,
                cam2global=cam2globals,
                lidar2img=lidar2imgs,
                num_cams=len(selected_idxs),
            )
        )

        #TODO: current implementation only supports loading annos from current frames
        if not self.test_mode:
            input_dict["ann_info"] = self.get_ann_info(index)

            if self.modality["use_camera"]:
                sample_token = self.data_infos[index]["token"]
                for idx in current_ret[-2]: # ret[-2] -> selected_idxs
                    cam_info = current_ret[-1][idx] # ret[-1] -> cam_list
                    annos_2d = self.img_dataset.get_ann_info(
                        self.img_token_idx[sample_token] + idx
                    )
                    assert cam_info["data_path"][:-3].endswith(
                        annos_2d["seg_map"][:-3]
                    ), "There is a 3D-2D annotation data mismatch!"

                    del annos_2d["gt_bboxes_3d"]
                    del annos_2d["gt_labels_3d"]

                    for key in annos_2d:
                        if key not in input_dict["ann_info"]:
                            input_dict["ann_info"][key] = [annos_2d[key]]
                        else:
                            input_dict["ann_info"][key].append(annos_2d[key])
        return input_dict

    def evaluate_simple(
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
        outputs = results[0]

        # Evaluating detections.
        import pdb; pdb.set_trace()
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

        delete_keys = []
        for k in outputs.keys():
            if k.endswith('3d'):
                delete_keys.append(k)
        for k in delete_keys:
            outputs.pop(k)

        return outputs



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


@DATASETS.register_module()
class NuScenesMultiModalDatasetV2VirtualCam(NuScenesMultiModalDatasetV2):

    def get_data_info(self, index):
        """Get data info according to the given index.

        Args:
            index (int): Index of the sample data to get.

        Returns:
            dict: Data information that will be passed to the data \
                preprocessing pipelines. It includes the following keys:

                - sample_idx (str): Sample index.
                - pts_filename (str): Filename of point clouds.
                - sweeps (list[dict]): Infos of sweeps.
                - timestamp (float): Sample timestamp.
                - img_filename (str, optional): Image filename.
                - lidar2img (list[np.ndarray], optional): Transformations \
                    from lidar to different cameras.
                - ann_info (dict): Annotation info.
        """
        # index %= 5  # TODO: BE CAREFUL!
        info = self.data_infos[index]

        input_dict = dict(
            pre_eval=self.pre_eval,
            data_idx=index,
            sample_idx=info["token"],
            pts_filename=info["lidar_path"],
            #  change here to replace ground truth lidar points with pseudo lidar points
            # pts_filename=info["lidar_path"].replace('nuscenes/samples', "nuscenes_virtual_highres/samples_0_0_0_0_0_0_224_400_1").replace('LIDAR_TOP/', 'pseudo_lidar/LIDAR_TOP/'),
            sweeps=info["sweeps"],
            timestamp=info["timestamp"] / 1e6,
        )

        # select camera indices to train on
        cam_list = list(info["cams"].values())
        if self.subselect_group_num > 0:
            # selected_idxs = self.groups[np.random.choice(len(self.groups))]
            selected_idxs = self.groups[5]
            input_dict.update(
                dict(
                    subselect_camera_num=len(selected_idxs),
                )
            )
        elif self.subselect_camera_num > 0:
            selected_idxs = (
                np.random.choice(self.subselect_camera_num)
                if self.subselect_random
                else range(self.subselect_camera_num)
            )

            input_dict.update(
                dict(
                    subselect_camera_num=self.subselect_camera_num,
                    subselect_random=self.subselect_random,
                )
            )

        else:
            selected_idxs = range(len(info["cams"]))

        def get_camera_images(info, anchor_lidar2global_inv=None, virtual_cam_pose=None):
            ''' virtual cam '''
            # add rgb postfix
            # apply virtual cam pose
            if virtual_cam_pose is not None:
                virtual_cam_right_matrix = np.eye(4)
                virtual_cam_right_matrix[0, 3] = 1.0

                virtual_cam_left_matrix = np.eye(4)
                virtual_cam_left_matrix[0, 3] = -1.0

                virtual_cam_up_matrix = np.eye(4)
                virtual_cam_up_matrix[1, 3] = -1.0

                virtual_cam_matrix_dict = {
                    "rightward": virtual_cam_right_matrix,
                    "leftward": virtual_cam_left_matrix,
                    "upward": virtual_cam_up_matrix,
                }
                virtual_cam_matrix = virtual_cam_matrix_dict[virtual_cam_pose]


                virtual_cam_postfix_dict = {
                    "rightward": 'samples_1_0_0_0_0_0',
                    "leftward": 'samples_-1_0_0_0_0_0',
                    "upward": 'samples_0_-1_0_0_0_0'
                }
                virtual_cam_postfix = virtual_cam_postfix_dict[virtual_cam_pose]

            l2e_r = info["lidar2ego_rotation"]
            l2e_t = info["lidar2ego_translation"]
            e2g_r = info["ego2global_rotation"]
            e2g_t = info["ego2global_translation"]
            l2e_r_mat = Quaternion(l2e_r).rotation_matrix
            e2g_r_mat = Quaternion(e2g_r).rotation_matrix

            lidar2ego = np.eye(4)
            lidar2ego[:3, :3] = l2e_r_mat
            lidar2ego[:3, 3] = l2e_t

            ego2global = np.eye(4)
            ego2global[:3, :3] = e2g_r_mat
            ego2global[:3, 3] = e2g_t
            lidar2global = ego2global @ lidar2ego

            image_paths = []
            intrinsics = []
            extrinsics = []
            cam2globals = []
            lidar2img = []
            cam_list = list(info["cams"].values())

            for idx in selected_idxs:
                cam_info = cam_list[idx]
                image_paths.append(cam_info["data_path"])
                intrinsic = cam_info["cam_intrinsic"]
                viewpad = np.eye(4)
                viewpad[: intrinsic.shape[0], : intrinsic.shape[1]] = intrinsic

                intrinsics.append(intrinsic.astype(np.float32))
                cam2lidar_rt = np.eye(4)
                cam2lidar_rt[:3, :3] = cam_info["sensor2lidar_rotation"]
                cam2lidar_rt[:3, 3] = cam_info["sensor2lidar_translation"]

                if anchor_lidar2global_inv is not None:
                    # for multi-frames, need to have a anchor frame and convert other frames w.r.t the anchor, to account for ego pose
                    cam2lidar_rt = anchor_lidar2global_inv @ lidar2global @ cam2lidar_rt

                if virtual_cam_pose is not None:
                    cam2lidar_rt = cam2lidar_rt @ virtual_cam_matrix

                extrinsics.append(cam2lidar_rt.astype(np.float32))

                lidar2cam = np.linalg.inv(cam2lidar_rt)
                lidar2img.append((viewpad @ lidar2cam).astype(np.float32))

                cam2globals.append((lidar2global @ cam2lidar_rt).astype(np.float32))

            if anchor_lidar2global_inv is None and virtual_cam_pose is None:
                return [image_paths, intrinsics, extrinsics, cam2globals, lidar2img, selected_idxs, cam_list], np.linalg.inv(lidar2global)
            else:
                return image_paths, intrinsics, extrinsics, cam2globals, lidar2img, selected_idxs, cam_list


        current_ret, anchor_lidar2global_inv = get_camera_images(info)

        if self.num_prev_frames > 0:
            prev_token = info['prev']
            prev_rets = []
            while len(prev_rets) < self.num_prev_frames:
                # if prev_token == '':
                #     prev_rets.append(current_ret)
                #     continue
                # prev_info = self.data_infos[self.token2ids[prev_token]]
                # prev_token = prev_info['prev']

                # prev_ret = get_camera_images(prev_info, anchor_lidar2global_inv=anchor_lidar2global_inv, )
                if len(prev_rets)==0:
                    prev_ret = get_camera_images(info, virtual_cam_pose='upward')
                else:
                    prev_ret = get_camera_images(info, virtual_cam_pose='leftward')
                prev_rets.append(prev_ret)
            prev_rets = prev_rets[::-1]
        else:
            prev_rets = []

        if self.num_next_frames > 0:
            next_token = info['next']
            next_rets = []
            while len(next_rets) < self.num_next_frames:
                if len(next_rets)==0:
                    if next_token == '':
                        next_rets.append(current_ret)
                        continue
                    next_info = self.data_infos[self.token2ids[next_token]]
                    next_token = next_info['next']
                    next_ret = get_camera_images(next_info, anchor_lidar2global_inv=anchor_lidar2global_inv)
                else:
                    next_ret = get_camera_images(info, virtual_cam_pose='rightward')
                next_rets.append(next_ret)
        else:
            next_rets = []
        rets = prev_rets + [current_ret] + next_rets

        image_paths = []
        intrinsics = []
        extrinsics = []
        cam2globals = []
        lidar2imgs = []
        selected_idxs = []
        for ret in rets:
            image_paths.extend(ret[0])
            intrinsics.extend(ret[1])
            extrinsics.extend(ret[2])
            cam2globals.extend(ret[3])
            lidar2imgs.extend(ret[4])
            selected_idxs.extend(ret[5])

        input_dict.update(
            dict(
                img_filename=image_paths,
                intrinsic=intrinsics,
                extrinsic=extrinsics,
                cam2global=cam2globals,
                lidar2img=lidar2imgs,
                num_cams=len(selected_idxs),
            )
        )

        #TODO: current implementation only supports loading annos from current frames
        if not self.test_mode:
            input_dict["ann_info"] = self.get_ann_info(index)

            if self.modality["use_camera"]:
                sample_token = self.data_infos[index]["token"]
                for idx in current_ret[-2]: # ret[-2] -> selected_idxs
                    cam_info = current_ret[-1][idx] # ret[-1] -> cam_list
                    annos_2d = self.img_dataset.get_ann_info(
                        self.img_token_idx[sample_token] + idx
                    )
                    assert cam_info["data_path"][:-3].endswith(
                        annos_2d["seg_map"][:-3]
                    ), "There is a 3D-2D annotation data mismatch!"

                    del annos_2d["gt_bboxes_3d"]
                    del annos_2d["gt_labels_3d"]

                    for key in annos_2d:
                        if key not in input_dict["ann_info"]:
                            input_dict["ann_info"][key] = [annos_2d[key]]
                        else:
                            input_dict["ann_info"][key].append(annos_2d[key])

        return input_dict
