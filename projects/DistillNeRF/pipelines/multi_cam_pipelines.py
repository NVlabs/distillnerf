import mmcv
import cv2
import numpy as np
import torch, os
import torch.nn.functional as F
from mmcv.parallel import DataContainer as DC
from mmdet3d.datasets import PIPELINES
from mmdet3d.datasets.pipelines import DefaultFormatBundle3D
from mmdet.datasets.pipelines import LoadImageFromFile, RandomFlip
import matplotlib.image as mpimg
import copy



@PIPELINES.register_module()
class DefaultFormatBundleCamInfo3D(DefaultFormatBundle3D):
    def __init__(
        self,
        volume_render_downscale_factor,
        class_names,
        with_gt=True,
        with_label=True,
        min_lidar_dist=1.0,
        max_lidar_dist=70.0,
    ):
        super(DefaultFormatBundleCamInfo3D, self).__init__(
            class_names=class_names, with_gt=with_gt, with_label=with_label
        )
        self.volume_render_downscale_factor = volume_render_downscale_factor
        self.min_lidar_dist = min_lidar_dist
        self.max_lidar_dist = max_lidar_dist

    def __call__(self, results):
        results = super(DefaultFormatBundleCamInfo3D, self).__call__(results)

        intrinsic = np.stack(results["intrinsic"], axis=0)
        results["intrinsic"] = DC(torch.from_numpy(intrinsic), stack=True)

        extrinsic = np.stack(results["extrinsic"], axis=0)
        results["extrinsic"] = DC(torch.from_numpy(extrinsic), stack=True)

        cam2global = np.stack(results["cam2global"], axis=0)
        results["cam2global"] = DC(torch.from_numpy(cam2global), stack=True)

        # Post-Homography transformations (from data augmentations).
        aug_transform = torch.diag_embed(torch.ones((results["num_cams"], 4)))
        # Image scaling (only affects x, y)
        scale_factor = np.stack(results["scale_factor"], axis=0)[..., :2]
        aug_transform[..., :2, :2] *= torch.diag_embed(torch.from_numpy(scale_factor))
        results["aug_transform"] = DC(aug_transform, stack=True)

        emernerf_depth_img = np.stack(results["emernerf_depth_img"], axis=0)
        results["emernerf_depth_img"] = DC(torch.from_numpy(emernerf_depth_img), stack=True)

        emernerf_sky_mask = np.stack(results["emernerf_sky_mask"], axis=0)
        results["emernerf_sky_mask"] = DC(torch.from_numpy(emernerf_sky_mask), stack=True)

        if "feat_img" in results:
            feat_img = np.stack(results["feat_img"], axis=0)
            results["feat_img"] = DC(torch.from_numpy(feat_img), stack=True)

        do_min_pool = True


        if "points" in results:
            # Depth Images (for supervising the generated depth_img via LIDAR).
            lidar_pts = results["points"].data.numpy()

            num_points = lidar_pts.shape[0]
            pts_4d = np.concatenate(
                [lidar_pts[:, :3], np.ones((num_points, 1), dtype=lidar_pts.dtype)],
                axis=-1,
            )
            gt_depth_imgs = []
            lidar_depths = []
            lidar_depth_loc2ds = []
            lidar_depth_masks = []
            for i, lidar2img in enumerate(results["lidar2img"]):
                ori_img_shape = results["ori_shape"][i]

                # Accounting for image flipping.
                flip_tf = np.eye(4, dtype=lidar2img.dtype)
                if results["flip"] and results["flip_direction"] == "horizontal":
                    flip_tf[1, 1] = -1

                pts_2d = pts_4d @ (flip_tf @ lidar2img.T)

                # cam_points is Tensor of Nx4 whose last column is 1
                # transform camera coordinate to image coordinate.
                pts_2d[:, 2] = np.clip(pts_2d[:, 2], a_min=1e-5, a_max=99999)
                pts_2d[:, 0] /= pts_2d[:, 2]
                pts_2d[:, 1] /= pts_2d[:, 2]

                # Accounting for image flipping.
                if results["flip"] and results["flip_direction"] == "horizontal":
                    pts_2d[:, 0] = ori_img_shape[1] - pts_2d[:, 0]

                # Remove points that are either outside or behind the camera.
                # Also, make sure points are at least self.min_lidar_dist and
                # at most self.max_lidar_dist in front of the camera to avoid
                # seeing the lidar points on the camera casing for non-keyframes
                # which are slightly out of sync.
                mask = (
                    (pts_2d[:, 2] >= self.min_lidar_dist)
                    & (pts_2d[:, 2] < self.max_lidar_dist)
                    & (pts_2d[:, 0] < ori_img_shape[1] - 0.5)
                    & (pts_2d[:, 0] >= 0)
                    & (pts_2d[:, 1] < ori_img_shape[0] - 0.5)
                    & (pts_2d[:, 1] >= 0)
                )

                lidar_depth = copy.deepcopy(pts_2d[:, 2])
                lidar_depth_loc2d = copy.deepcopy(pts_2d[:, :2])
                lidar_depth_loc2d[:, 0] = lidar_depth_loc2d[:, 0] / ori_img_shape[1]
                lidar_depth_loc2d[:, 1] = lidar_depth_loc2d[:, 1] / ori_img_shape[0]
                lidar_depth_mask = copy.deepcopy(mask)

                lidar_depths.append(torch.from_numpy(lidar_depth).unsqueeze(0))
                lidar_depth_loc2ds.append(torch.from_numpy(lidar_depth_loc2d).unsqueeze(0))
                lidar_depth_masks.append(torch.from_numpy(lidar_depth_mask).unsqueeze(0))

                pts_2d = pts_2d[mask, :3]  # [u, v, depth]

                dist = pts_2d[:, 2]
                pts_2d = np.rint(pts_2d[:, :2]).astype(int)

                if do_min_pool:
                    pt_ind = pts_2d[:, 1] * ori_img_shape[1] + pts_2d[:, 0]
                    neg_dist = -dist
                    sorted_ind = np.argsort(pt_ind)
                    sorted_dist = neg_dist[sorted_ind]
                    unique_ind, unique_ind_first_pos = np.unique(pt_ind[sorted_ind], return_index=True)
                    min_dist = -np.maximum.reduceat(sorted_dist, unique_ind_first_pos)
                    pt_ind_h = np.minimum(unique_ind // ori_img_shape[1], ori_img_shape[0]-1)
                    pt_ind_w = np.minimum(unique_ind - (pt_ind_h * ori_img_shape[1]), ori_img_shape[1]-1)

                    # Setting these to 1000 so later min-pooling will pick up non-negative values.
                    gt_depth_img = np.full(ori_img_shape[:2], fill_value=1000.0)
                    gt_depth_img[pt_ind_h, pt_ind_w] = min_dist

                else:
                    # Setting these to -1 so later max-pooling will pick up non-negative values.
                    gt_depth_img = np.full(ori_img_shape[:2], fill_value=-1.0)
                    gt_depth_img[pts_2d[:, 1], pts_2d[:, 0]] = dist

                gt_depth_imgs.append(torch.from_numpy(gt_depth_img).unsqueeze(0))

            # They're all the same shape so we can just take the first
            # element of results["pad_shape"].
            depth_img_shape = (
                results["img_shape"][0][0] // self.volume_render_downscale_factor,
                results["img_shape"][0][1] // self.volume_render_downscale_factor,
            )

            gt_depth_imgs = torch.stack(gt_depth_imgs, dim=0)

            kernel_size = (
                int(ori_img_shape[0] / depth_img_shape[0]),
                int(ori_img_shape[1] / depth_img_shape[1]),
            )

            if do_min_pool:
                gt_depth_imgs = -F.max_pool2d(-gt_depth_imgs, kernel_size, stride=kernel_size)
                gt_depth_imgs[gt_depth_imgs == 1000] = -1
            else:
                gt_depth_imgs = F.max_pool2d(gt_depth_imgs, kernel_size, stride=kernel_size)
            gt_depth_imgs = F.interpolate(gt_depth_imgs, depth_img_shape[:2])
            gt_depth_imgs = gt_depth_imgs.float()

            # Masking out no-data pixels.
            gt_depth_imgs[gt_depth_imgs < 0] = torch.nan

            lidar_depths = torch.stack(lidar_depths, dim=0)
            lidar_depth_loc2ds = torch.stack(lidar_depth_loc2ds, dim=0)
            lidar_depth_masks = torch.stack(lidar_depth_masks, dim=0)

            
            results["lidar_depths"] = DC(lidar_depths, stack=True)
            results["lidar_depth_loc2ds"] = DC(lidar_depth_loc2ds, stack=True)
            results["lidar_depth_masks"] = DC(lidar_depth_masks, stack=True)

            results["gt_depth_img"] = DC(gt_depth_imgs, stack=True)
        return results


@PIPELINES.register_module()
class LoadImagesFromFiles(LoadImageFromFile):
    def __call__(self, results):
        results["img"] = []
        results["img_shape"] = []
        results["ori_shape"] = []
        for filename in results["img_filename"]:
            img_result = {
                "img_prefix": None,
                "img_info": {
                    "filename": filename,
                },
            }
            super().__call__(img_result)

            results["img"].append(img_result["img"])
            results["img_shape"].append(img_result["img_shape"])
            results["ori_shape"].append(img_result["ori_shape"])

        results["img_fields"] = ["img"]
        return results

@PIPELINES.register_module()
class LoadClipImagesFromFiles(LoadImageFromFile):
    def __init__(self, skip_missing=False, clip_img_path="nuscenes_feat/samples_clip_vitl14_336"):
        super(LoadImageFromFile, self).__init__()
        self.skip_missing = skip_missing        # force to load a fixed file for local testing
        self.clip_img_path = clip_img_path

    def __call__(self, results):
        results["feat_img"] = []
        for filename in results["img_filename"]:
            clip_img_path = filename.replace("nuscenes/samples", self.clip_img_path).replace("png", "npy").replace("jpg", "npy")

            if os.path.exists(clip_img_path):
                clip_img = np.load(clip_img_path) 
            else:
                # import pdb; pdb.set_trace()
                if self.skip_missing:
                    clip_img = np.load("tempt_file/data/nuscenes_feat/samples_clip_vitl14_336/CAM_FRONT/n015-2018-07-24-11-22-45+0800__CAM_FRONT__1532402865162460.npy") 
                else:
                    raise ValueError(f"File not found: {clip_img_path}")

            results["feat_img"].append(clip_img)

        return results

@PIPELINES.register_module()
class LoadDINOImagesFromFiles(LoadImageFromFile):
    def __init__(self, skip_missing=False, dino_img_path="nuscenes_feat/samples_dinov2_vitb14"):
        super(LoadImageFromFile, self).__init__()
        self.skip_missing = skip_missing        # force to load a fixed file for local testing
        self.dino_img_path = dino_img_path

    def __call__(self, results):

        results["feat_img"] = []
        for filename in results["img_filename"]:
            dino_img_path = filename.replace("nuscenes/samples", self.dino_img_path).replace("png", "npy").replace("jpg", "npy")
            if os.path.exists(dino_img_path):
                dino_img = np.load(dino_img_path) 
            else:
                if self.skip_missing:
                    # dino_img = np.load("tempt_file/data/nuscenes_feat/samples_dinov2_vitb14/CAM_FRONT/n015-2018-07-24-11-22-45+0800__CAM_FRONT__1532402865162460.npy") 
                    dino_img = np.load("tempt_file/data/nuscenes_feat/samples_clip_vitl14_336/CAM_FRONT/n015-2018-07-24-11-22-45+0800__CAM_FRONT__1532402865162460.npy") 
                else:
                    raise ValueError(f"File not found: {dino_img_path}")

            results["feat_img"].append(dino_img)

        return results


@PIPELINES.register_module()
class LoadVirtualImagesFromFiles(LoadImageFromFile):
    def __init__(self, skip_missing=False, virtual_img_path="nuscenes_emernerf/"):
        super(LoadImageFromFile, self).__init__()
        self.skip_missing = skip_missing        # force to load a fixed file for local testing
        self.virtual_img_path = virtual_img_path

    def __call__(self, results):
        Force_fixed_file = False
        results["virtual_img"] = []
        results["emernerf_depth_img"] = []
        results["emernerf_sky_mask"] = []
        virtual_cam_postfix_dict = {
            "rightward": 'samples_1_0_0_0_0_0',
            "leftward": 'samples_-1_0_0_0_0_0',
            "upward": 'samples_0_-1_0_0_0_0'
        }

        cam_num = len(results["img_filename"]) / 5
        for i, filename in enumerate(results["img_filename"]):
            virtual_cam_postfix = None
            if i < cam_num * 1: virtual_cam_postfix = virtual_cam_postfix_dict['leftward']
            elif i < cam_num * 2: virtual_cam_postfix = virtual_cam_postfix_dict['upward']
            elif i >= cam_num * 4: virtual_cam_postfix = virtual_cam_postfix_dict['rightward']

            ''' load virtual rgb image '''
            if virtual_cam_postfix is not None:
                filename_virt = filename.replace("nuscenes/", self.virtual_img_path)
                filename_virt = filename_virt.replace("samples", virtual_cam_postfix+'/rgbs')
            else:   # load original camera view
                filename_virt = filename

            if os.path.exists(filename_virt) and not Force_fixed_file:
                virtual_img = mpimg.imread(filename_virt)[:, :, [2,1,0]] 
            else:
                if self.skip_missing:
                    virtual_img = mpimg.imread("tempt_file/data/nuscenes_emernerf/samples_-1_0_0_0_0_0/rgbs/CAM_FRONT_LEFT/n008-2018-08-01-15-16-36-0400__CAM_FRONT_LEFT__1533151603504799.jpg")[:, :, [2,1,0]] 
                else:
                    raise ValueError(f"File not found: {filename_virt}")
            results["virtual_img"].append(virtual_img)

            
            ''' load virtual depth image and sky mask '''
            # the emernerf sky mask is the same as the original camera view, here we load them just to run through the pipeline
            # the actual sky mask is loaded in the model wrapper
            if virtual_cam_postfix is not None:
                filename = filename.replace("nuscenes/", self.virtual_img_path)
                depth_img_path = filename.replace("samples", virtual_cam_postfix+"/depths").replace("png", "npy").replace("jpg", "npy")
                sky_mask_path = filename.replace("samples", virtual_cam_postfix+"/gt_sky_masks").replace("png", "npy").replace("jpg", "npy")
            else:   # load original camera view
                filename = filename.replace("nuscenes/", self.virtual_img_path)
                depth_img_path = filename.replace("samples", "samples_0_0_0_0_0_0/depths").replace("png", "npy").replace("jpg", "npy")
                sky_mask_path = filename.replace("samples", "samples_0_0_0_0_0_0/gt_sky_masks").replace("png", "npy").replace("jpg", "npy")

            # depth imge
            if os.path.exists(depth_img_path) and not Force_fixed_file:
                depth_img = np.load(depth_img_path) 
            else:
                if self.skip_missing:
                    depth_img = np.load("tempt_file/data/nuscenes_emernerf/samples_0_0_0_0_0_0/depths/CAM_BACK/n015-2018-07-24-11-22-45+0800__CAM_BACK__1532402927637525.npy") 
                else:
                    raise ValueError(f"File not found: {depth_img_path}")

            # sky mask
            if os.path.exists(sky_mask_path) and not Force_fixed_file:
                sky_mask = np.load(sky_mask_path)
            elif os.path.exists(sky_mask_path.replace("npy", "jpg")) and not Force_fixed_file:
                sky_mask = cv2.imread(sky_mask_path.replace("npy", "jpg"), cv2.IMREAD_GRAYSCALE) / 255
            else:
                if self.skip_missing:
                    sky_mask = np.load("tempt_file/data/nuscenes_emernerf/samples_0_0_0_0_0_0/gt_sky_masks/CAM_BACK/n015-2018-07-24-11-22-45+0800__CAM_BACK__1532402927637525.npy") 
                else:
                    raise ValueError(f"File not found: {sky_mask_path}")

            results["emernerf_depth_img"].append(depth_img)
            results["emernerf_sky_mask"].append(sky_mask)
            
        return results


@PIPELINES.register_module()
class LoadDepthImagesFromFiles(LoadImageFromFile):
    def __init__(self, virtual_cam_dir=None, virtual_cam_postfix=None, skip_missing=False, emernerf_dir="nuscenes_emernerf/"):
        super(LoadDepthImagesFromFiles, self).__init__()
        self.virtual_cam_dir = virtual_cam_dir                      # None means loading original camera view
        self.virtual_cam_postfix = virtual_cam_postfix              # None means loading original camera view
        self.emernerf_dir = emernerf_dir
        self.skip_missing = skip_missing                            # force to load a fixed file for local testing

    def __call__(self, results):
        results["emernerf_depth_img"] = []
        results["emernerf_sky_mask"] = []

        ''' load depth image '''
        for filename in results["img_filename"]:
            if self.virtual_cam_postfix is not None and self.virtual_cam_dir is not None:  # virtual camera view
                filename = filename.replace("nuscenes/", self.virtual_cam_dir)
                depth_img_path = filename.replace("samples", f"{self.virtual_cam_postfix}/depths").replace("png", "npy").replace("jpg", "npy")
                sky_mask_path = filename.replace("samples", f"{self.virtual_cam_postfix}/gt_sky_masks").replace("png", "npy").replace("jpg", "npy")
            else:   # original camera view
                filename = filename.replace("nuscenes/", self.emernerf_dir)
                depth_img_path = filename.replace("samples", "samples_0_0_0_0_0_0/depths").replace("png", "npy").replace("jpg", "npy")
                sky_mask_path = filename.replace("samples", "samples_0_0_0_0_0_0/gt_sky_masks").replace("png", "npy").replace("jpg", "npy")

            if os.path.exists(depth_img_path):
                depth_img = np.load(depth_img_path) 
            else:
                if self.skip_missing:
                    depth_img = np.load("tempt_file/data/nuscenes_emernerf/samples_0_0_0_0_0_0/depths/CAM_BACK/n015-2018-07-24-11-22-45+0800__CAM_BACK__1532402927637525.npy") 
                else:
                    raise ValueError(f"File not found: {depth_img_path}")

            ''' load sky maskdepth image '''
            if os.path.exists(sky_mask_path):
                sky_mask = np.load(sky_mask_path)
            elif os.path.exists(sky_mask_path.replace("npy", "jpg")):
                sky_mask = cv2.imread(sky_mask_path.replace("npy", "jpg"), cv2.IMREAD_GRAYSCALE) / 255
            else:
                if self.skip_missing:
                    sky_mask = np.load("tempt_file/data/nuscenes_emernerf/samples_0_0_0_0_0_0/gt_sky_masks/CAM_BACK/n015-2018-07-24-11-22-45+0800__CAM_BACK__1532402927637525.npy") 
                else:
                    raise ValueError(f"File not found: {sky_mask_path}")

            results["emernerf_depth_img"].append(depth_img)
            results["emernerf_sky_mask"].append(sky_mask)

        return results


@PIPELINES.register_module()
class LoadDepthImagesFromFilesWaymo(LoadImageFromFile):
    def __init__(self, virtual_cam_dir=None, virtual_cam_postfix=None, skip_missing=False):
        super(LoadDepthImagesFromFilesWaymo, self).__init__()
        self.virtual_cam_dir = virtual_cam_dir                      # None means loading original camera view
        self.virtual_cam_postfix = virtual_cam_postfix              # None means loading original camera view
        self.skip_missing = skip_missing                            # force to load a fixed file for local testing

    def __call__(self, results):
        results["emernerf_depth_img"] = []
        results["emernerf_sky_mask"] = []

        ''' load depth image '''
        for filename in results["img_filename"]:
            if self.virtual_cam_postfix is not None and self.virtual_cam_dir is not None:  # virtual camera view
                filename = filename.replace("nuscenes/", self.virtual_cam_dir)
                depth_img_path = filename.replace("samples", f"{self.virtual_cam_postfix}/depths").replace("png", "npy").replace("jpg", "npy")
                sky_mask_path = filename.replace("samples", f"{self.virtual_cam_postfix}/gt_sky_masks").replace("png", "npy").replace("jpg", "npy")
            else:   # original camera view
                depth_img_path = filename.replace("/image", "/depth_img").replace("png", "npy").replace("jpg", "npy")
                sky_mask_path = filename.replace("/image", "/sky_mask").replace("jpg", "png")

            if os.path.exists(depth_img_path):
                depth_img = np.load(depth_img_path) 
            else:
                if self.skip_missing:
                    # TODO: switch to waymo depth image
                    depth_img = np.load("tempt_file/data/nuscenes_emernerf/samples_0_0_0_0_0_0/depths/CAM_BACK/n015-2018-07-24-11-22-45+0800__CAM_BACK__1532402927637525.npy") 
                else:
                    raise ValueError(f"File not found: {depth_img_path}")

            ''' load sky maskdepth image '''
            if os.path.exists(sky_mask_path):
                sky_mask = cv2.imread(sky_mask_path, cv2.IMREAD_GRAYSCALE) / 255
            else:
                if self.skip_missing:
                    sky_mask = cv2.imread("tempt_file/data/waymo_emernerf/emernerf_notr/sky_mask_0/0000000.png", cv2.IMREAD_GRAYSCALE) / 255
                else:
                    raise ValueError(f"File not found: {sky_mask_path}")

            results["emernerf_depth_img"].append(depth_img)
            results["emernerf_sky_mask"].append(sky_mask)

        return results
    

@PIPELINES.register_module()
class MultiCameraRandomFlip(RandomFlip):
    def __call__(self, results):
        """Call function to flip bounding boxes, masks, semantic segmentation
        maps.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Flipped results, 'flip', 'flip_direction' keys are added \
                into result dict.
        """

        if "flip" not in results:
            if isinstance(self.direction, list):
                # None means non-flip
                direction_list = self.direction + [None]
            else:
                # None means non-flip
                direction_list = [self.direction, None]

            if isinstance(self.flip_ratio, list):
                non_flip_ratio = 1 - sum(self.flip_ratio)
                flip_ratio_list = self.flip_ratio + [non_flip_ratio]
            else:
                non_flip_ratio = 1 - self.flip_ratio
                # exclude non-flip
                single_ratio = self.flip_ratio / (len(direction_list) - 1)
                flip_ratio_list = [single_ratio] * (len(direction_list) - 1) + [
                    non_flip_ratio
                ]

            cur_dir = np.random.choice(direction_list, p=flip_ratio_list)

            results["flip"] = cur_dir is not None
        if "flip_direction" not in results:
            results["flip_direction"] = cur_dir
        if results["flip"]:
            # flip image
            for key in results.get("img_fields", ["img"]):
                if isinstance(results[key], list):
                    results[key] = [
                        mmcv.imflip(img, direction=results["flip_direction"])
                        for img in results[key]
                    ]
                else:
                    results[key] = mmcv.imflip(
                        results[key], direction=results["flip_direction"]
                    )
            # flip bboxes
            for key in results.get("bbox_fields", []):
                if isinstance(results[key], list):
                    results[key] = [
                        self.bbox_flip(
                            value, results["img_shape"][idx], results["flip_direction"]
                        )
                        for idx, value in enumerate(results[key])
                    ]
                else:
                    results[key] = self.bbox_flip(
                        results[key], results["img_shape"], results["flip_direction"]
                    )
            # flip masks
            for key in results.get("mask_fields", []):
                results[key] = results[key].flip(results["flip_direction"])

            # flip segs
            for key in results.get("seg_fields", []):
                results[key] = mmcv.imflip(
                    results[key], direction=results["flip_direction"]
                )
        return results


@PIPELINES.register_module()
class MultiCameraRandomFlip3D(MultiCameraRandomFlip):
    """Flip the points & bbox.

    If the input dict contains the key "flip", then the flag will be used,
    otherwise it will be randomly decided by a ratio specified in the init
    method.

    Args:
        sync_2d (bool, optional): Whether to apply flip according to the 2D
            images. If True, it will apply the same flip as that to 2D images.
            If False, it will decide whether to flip randomly and independently
            to that of 2D images. Defaults to True.
        flip_ratio_bev_horizontal (float, optional): The flipping probability
            in horizontal direction. Defaults to 0.0.
        flip_ratio_bev_vertical (float, optional): The flipping probability
            in vertical direction. Defaults to 0.0.
    """

    def __init__(
        self,
        sync_2d=True,
        flip_ratio_bev_horizontal=0.0,
        flip_ratio_bev_vertical=0.0,
        **kwargs,
    ):
        super(MultiCameraRandomFlip3D, self).__init__(
            flip_ratio=flip_ratio_bev_horizontal, **kwargs
        )
        self.sync_2d = sync_2d
        self.flip_ratio_bev_vertical = flip_ratio_bev_vertical
        if flip_ratio_bev_horizontal is not None:
            assert (
                isinstance(flip_ratio_bev_horizontal, (int, float))
                and 0 <= flip_ratio_bev_horizontal <= 1
            )
        if flip_ratio_bev_vertical is not None:
            assert (
                isinstance(flip_ratio_bev_vertical, (int, float))
                and 0 <= flip_ratio_bev_vertical <= 1
            )

    def random_flip_data_3d(self, input_dict, direction="horizontal"):
        """Flip 3D data randomly.

        Args:
            input_dict (dict): Result dict from loading pipeline.
            direction (str, optional): Flip direction.
                Default: 'horizontal'.

        Returns:
            dict: Flipped results, 'points', 'bbox3d_fields' keys are
                updated in the result dict.
        """
        assert direction in ["horizontal", "vertical"]
        # for semantic segmentation task, only points will be flipped.
        if "bbox3d_fields" not in input_dict:
            input_dict["points"].flip(direction)
            return
        if len(input_dict["bbox3d_fields"]) == 0:  # test mode
            input_dict["bbox3d_fields"].append("empty_box3d")
            input_dict["empty_box3d"] = input_dict["box_type_3d"](
                np.array([], dtype=np.float32)
            )
        assert len(input_dict["bbox3d_fields"]) == 1
        for key in input_dict["bbox3d_fields"]:
            if "points" in input_dict:
                input_dict["points"] = input_dict[key].flip(
                    direction, points=input_dict["points"]
                )
            else:
                input_dict[key].flip(direction)
        if "centers2d" in input_dict:
            assert (
                self.sync_2d is True and direction == "horizontal"
            ), "Only support sync_2d=True and horizontal flip with images"
            if isinstance(input_dict["centers2d"], list):
                for idx, value in enumerate(input_dict["centers2d"]):
                    w = input_dict["ori_shape"][idx][1]
                    value[..., 0] = w - value[..., 0]
                    # need to modify the horizontal position of camera center
                    # along u-axis in the image (flip like centers2d)
                    # ['intrinsic'][0][2] = c_u
                    # see more details and examples at
                    # https://github.com/open-mmlab/mmdetection3d/pull/744
                    input_dict["intrinsic"][idx][0][2] = (
                        w - input_dict["intrinsic"][idx][0][2]
                    )
            else:
                w = input_dict["ori_shape"][1]
                input_dict["centers2d"][..., 0] = w - input_dict["centers2d"][..., 0]
                # need to modify the horizontal position of camera center
                # along u-axis in the image (flip like centers2d)
                # ['intrinsic'][0][2] = c_u
                # see more details and examples at
                # https://github.com/open-mmlab/mmdetection3d/pull/744
                input_dict["intrinsic"][0][2] = w - input_dict["intrinsic"][0][2]

    def __call__(self, input_dict):
        """Call function to flip points, values in the ``bbox3d_fields`` and
        also flip 2D image and its annotations.

        Args:
            input_dict (dict): Result dict from loading pipeline.

        Returns:
            dict: Flipped results, 'flip', 'flip_direction',
                'pcd_horizontal_flip' and 'pcd_vertical_flip' keys are added
                into result dict.
        """
        # flip 2D image and its annotations
        super(MultiCameraRandomFlip3D, self).__call__(input_dict)

        if self.sync_2d:
            input_dict["pcd_horizontal_flip"] = input_dict["flip"]
            input_dict["pcd_vertical_flip"] = False
        else:
            if "pcd_horizontal_flip" not in input_dict:
                flip_horizontal = True if np.random.rand() < self.flip_ratio else False
                input_dict["pcd_horizontal_flip"] = flip_horizontal
            if "pcd_vertical_flip" not in input_dict:
                flip_vertical = (
                    True if np.random.rand() < self.flip_ratio_bev_vertical else False
                )
                input_dict["pcd_vertical_flip"] = flip_vertical

        if "transformation_3d_flow" not in input_dict:
            input_dict["transformation_3d_flow"] = []

        if input_dict["pcd_horizontal_flip"]:
            self.random_flip_data_3d(input_dict, "horizontal")
            input_dict["transformation_3d_flow"].extend(["HF"])
        if input_dict["pcd_vertical_flip"]:
            self.random_flip_data_3d(input_dict, "vertical")
            input_dict["transformation_3d_flow"].extend(["VF"])
        return input_dict

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += f"(sync_2d={self.sync_2d},"
        repr_str += f" flip_ratio_bev_vertical={self.flip_ratio_bev_vertical})"
        return repr_str
