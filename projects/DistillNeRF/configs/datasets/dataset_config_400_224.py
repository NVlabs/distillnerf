''' need to change'''
# If point cloud range is changed, the models should also change their point cloud range accordingly

# determines the filter for the point cloud and object
# point_cloud_range = [-50, -50, -5, 50, 50, 3]
point_cloud_range = [-80, -80, -6.4, 80, 80, 6.4]

# What do we want our input image size to be (into the image encoder)?

# used to reshape the camera image
desired_input_img_size = (400, 224)
# desired_input_img_size = (400, 200)

# The volume renderer will render a feature image from the feature voxels.
# How much should this image be spatially downscaled compared to the input?
# We choose half of the input image size (which will then be upscaled via StyleGANv2's decoder).

# need to correspond to the model yaml file:  input_img_width / render_width, input_img_height / render_height
# used to reshape the lidar depth image
volume_render_downscale_factor = 2

# skip missing date, only for local debugging
SKIP_MISSING = True
'''need to change'''

# What fraction of images to flip horizontally, between 0 and 1 (as an augmentation).
img_aug_flip_fraction = 0.0
# Number of LIDAR sweeps to aggregate per data sample.
lidar_sweeps_num = 10
# -1 here means to use all cameras, no subselection.
subselect_camera_num = -1


# For nuScenes we usually do 10-class detection
class_names = [
    "car",
    "truck",
    "trailer",
    "bus",
    "construction_vehicle",
    "bicycle",
    "motorcycle",
    "pedestrian",
    "traffic_cone",
    "barrier",
]
dataset_type = "NuScenesMultiModalDatasetV2"
data_root = "data/nuscenes/"
# Input modality for nuScenes dataset, this is consistent with the submission
# format which requires the information in input_modality.
input_modality = dict(
    use_lidar=True, use_camera=True, use_radar=False, use_map=False, use_external=False
)
file_client_args = dict(backend="disk")
# Uncomment the following if use ceph or other file clients.
# See https://mmcv.readthedocs.io/en/latest/api.html#mmcv.fileio.FileClient
# for more details.
# file_client_args = dict(
#     backend='petrel',
#     path_mapping=dict({
#         './data/nuscenes/': 's3://nuscenes/nuscenes/',
#         'data/nuscenes/': 's3://nuscenes/nuscenes/'
#     }))


# How much will the encoder divide the input dimensions by?
# PVTv2 stage 0 does (H/4, W/4).
# encoder_size_divisor = 4


train_pipeline = [
    dict(type="LoadImagesFromFiles"),
    dict(type="LoadDepthImagesFromFiles",
        virtual_cam_dir = "nuscenes_virtual_highres/",
        virtual_cam_postfix = "samples_0_0_0_0_0_0_224_400_1",
        skip_missing=SKIP_MISSING
    ),
    dict(
        type="LoadPointsFromFile",
        coord_type="LIDAR",
        load_dim=5,
        use_dim=5,
        file_client_args=file_client_args,
    ),
    # dict(
    #     type="LoadPointsFromMultiSweeps",
    #     sweeps_num=lidar_sweeps_num,
    #     file_client_args=file_client_args,
    # ),
    dict(
        type="LoadAnnotations3D",
        with_bbox=True,
        with_label=True,
        with_attr_label=True,
        with_bbox_3d=True,
        with_label_3d=True,
        with_bbox_depth=True,
    ),
    # dict(
    #     type="GlobalRotScaleTrans",
    #     rot_range=[-0.3925, 0.3925],
    #     scale_ratio_range=[0.95, 1.05],
    #     translation_std=[0, 0, 0],
    # ),
    dict(
        type="MultiViewWrapper",
        transforms=[
            dict(type="Resize", img_scale=desired_input_img_size, keep_ratio=False),
            # dict(type="Pad", size_divisor=encoder_size_divisor),
        ],
        collected_keys=["img_shape", "scale_factor"],
    ),
    dict(
        type="MultiCameraRandomFlip3D", flip_ratio_bev_horizontal=img_aug_flip_fraction
    ),
    dict(type="PointsRangeFilter", point_cloud_range=point_cloud_range),
    dict(type="ObjectRangeFilter", point_cloud_range=point_cloud_range),
    dict(type="ObjectNameFilter", classes=class_names),
    dict(type="PointShuffle"),
    dict(
        type="DefaultFormatBundleCamInfo3D",
        volume_render_downscale_factor=volume_render_downscale_factor,
        class_names=class_names,
        min_lidar_dist=1,
        max_lidar_dist=point_cloud_range[3]
    ),
    dict(
        type="Collect3D",
        keys=[
            "img",
            "points",
            "intrinsic",
            "extrinsic",
            # "cam2global",
            "aug_transform",
            # "ray_origins",
            # "ray_directions",
            "gt_bboxes",
            "gt_labels",
            "attr_labels",
            "gt_bboxes_3d",
            "gt_labels_3d",
            "centers2d",
            "depths",
            "gt_depth_img",
            "lidar_depths",
            "lidar_depth_loc2ds",
            "lidar_depth_masks",
            "emernerf_depth_img",
            "emernerf_sky_mask",
        ],
    ),
]
test_pipeline = [
    dict(type="LoadImagesFromFiles"),
    dict(
        type="LoadDepthImagesFromFiles",
        virtual_cam_dir="nuscenes_virtual_highres/",
        virtual_cam_postfix="samples_0_0_0_0_0_0_224_400_1",
        skip_missing=SKIP_MISSING
        ),
    dict(
        type="LoadPointsFromFile",
        coord_type="LIDAR",
        load_dim=5,
        use_dim=5,
        file_client_args=file_client_args,
    ),
    # dict(
    #     type="LoadPointsFromMultiSweeps",
    #     sweeps_num=lidar_sweeps_num,
    #     file_client_args=file_client_args,
    #     test_mode=True,
    # ),
    dict(
        type="LoadAnnotations3D",
        with_bbox=False,
        with_label=False,
        with_attr_label=False,
        with_bbox_3d=False,
        with_label_3d=False,
        with_bbox_depth=False,
    ),
    dict(
        type="MultiScaleFlipAug3D",
        img_scale=desired_input_img_size,
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            # dict(
            #     type="GlobalRotScaleTrans",
            #     rot_range=[0, 0],
            #     scale_ratio_range=[1.0, 1.0],
            #     translation_std=[0, 0, 0],
            # ),
            dict(
                type="MultiViewWrapper",
                transforms=[
                    dict(
                        type="Resize",
                        img_scale=desired_input_img_size,
                        keep_ratio=False,
                    ),
                    # dict(type="Pad", size_divisor=encoder_size_divisor),
                ],
                collected_keys=[
                    "img_shape",
                    "scale_factor",
                    # "pad_shape",
                ],
            ),
            dict(type="MultiCameraRandomFlip3D"),
            dict(type="PointsRangeFilter", point_cloud_range=point_cloud_range),
            dict(
                type="DefaultFormatBundleCamInfo3D",
                volume_render_downscale_factor=volume_render_downscale_factor,
                class_names=class_names,
                with_label=False,
                min_lidar_dist=1,
                max_lidar_dist=point_cloud_range[3]
            ),
            dict(
                type="Collect3D",
                keys=[
                    "data_idx",
                    "pre_eval",
                    "img",
                    "points",
                    "intrinsic",
                    "extrinsic",
                    # "cam2global",
                    "aug_transform",
                    # "ray_origins",
                    # "ray_directions",
                    "gt_depth_img",
                    "lidar_depths",
                    "lidar_depth_loc2ds",
                    "lidar_depth_masks",
                    "emernerf_depth_img",
                    "emernerf_sky_mask",
                ],
            ),
        ],
    ),
]
# Construct a pipeline for data and gt loading in show/dataset.evaluate functions.
# Please keep its loading function consistent with test_pipeline (e.g. client)
# We need to load the original image and GT depth images for evaluation!
eval_pipeline = [
    dict(type="LoadImagesFromFiles"),
    dict(
        type="LoadDepthImagesFromFiles",
        virtual_cam_dir="nuscenes_virtual_highres/",
        virtual_cam_postfix="samples_0_0_0_0_0_0_224_400_1",
        skip_missing=SKIP_MISSING
        ),
    dict(
        type="LoadPointsFromFile",
        coord_type="LIDAR",
        load_dim=5,
        use_dim=5,
        file_client_args=file_client_args,
    ),
    # dict(
    #     type="LoadPointsFromMultiSweeps",
    #     sweeps_num=lidar_sweeps_num,
    #     file_client_args=file_client_args,
    #     test_mode=True,
    # ),
    dict(
        type="LoadAnnotations3D",
        with_bbox=False,
        with_label=False,
        with_attr_label=False,
        with_bbox_3d=False,
        with_label_3d=False,
        with_bbox_depth=False,
    ),
    dict(
        type="MultiViewWrapper",
        transforms=[
            dict(type="Resize", img_scale=desired_input_img_size, keep_ratio=False),
            # dict(type="Pad", size_divisor=encoder_size_divisor),
        ],
        collected_keys=["img_shape", "scale_factor"],
    ),
    dict(type="MultiCameraRandomFlip3D"),
    dict(type="PointsRangeFilter", point_cloud_range=point_cloud_range),
    dict(
        type="DefaultFormatBundleCamInfo3D",
        volume_render_downscale_factor=volume_render_downscale_factor,
        class_names=class_names,
        with_label=False,
        min_lidar_dist=1,
        max_lidar_dist=point_cloud_range[3]
    ),
    dict(
        type="Collect3D",
        keys=[
            "img",
            "gt_depth_img",
            "lidar_depths",
            "lidar_depth_loc2ds",
            "lidar_depth_masks",
            "emernerf_depth_img",
            "emernerf_sky_mask"
        ],
    ),
]

data = dict(
    samples_per_gpu=1,
    workers_per_gpu=62,
    persistent_workers=True,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file_3d=data_root + "nuscenes_infos_train.pkl",
        ann_file_2d=data_root + "nuscenes_infos_train_mono3d.coco.json",
        pipeline=train_pipeline,
        classes=class_names,
        modality=input_modality,
        filter_empty_gt=False,
        test_mode=False,
        # we use box_type_3d='LiDAR' in kitti and nuscenes dataset
        # and box_type_3d='Depth' in sunrgbd and scannet dataset.
        box_type_3d="LiDAR",
        subselect_camera_num=subselect_camera_num,
        pre_eval=False,
    ),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file_3d=data_root + "nuscenes_infos_val.pkl",
        ann_file_2d=data_root + "nuscenes_infos_val_mono3d.coco.json",
        pipeline=test_pipeline,
        classes=class_names,
        modality=input_modality,
        filter_empty_gt=False,
        test_mode=True,
        box_type_3d="LiDAR",
        subselect_camera_num=subselect_camera_num,
        pre_eval=True,
    ),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file_3d=data_root + "nuscenes_infos_val.pkl",
        ann_file_2d=data_root + "nuscenes_infos_val_mono3d.coco.json",
        pipeline=test_pipeline,
        classes=class_names,
        modality=input_modality,
        filter_empty_gt=False,
        test_mode=True,
        box_type_3d="LiDAR",
        subselect_camera_num=subselect_camera_num,
        pre_eval=True,
    ),
)


# evaluation = dict(interval=1, pipeline=eval_pipeline, gpu_collect=True)
evaluation = dict(interval=1, pipeline=eval_pipeline)
