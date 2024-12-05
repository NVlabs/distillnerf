_base_ = [
    "../../configs/_base_/default_runtime.py",
    "../datasets/dataset_config_dino.py",
]

data = dict(
    train=dict(
        num_prev_frames=2,
        num_next_frames=2,
        subselect_group_num=6),
    val=dict(
        num_prev_frames=2,
        num_next_frames=2,
        subselect_group_num=6),
    test=dict(
        num_prev_frames=2,
        num_next_frames=2,
        subselect_group_num=6))
# when subselect_group_num > 0 and subselect_camera_num = -1, subselect_camera_num does nothing
# subselect_group_num controls the number of camera used
# can change the order of cameras in nuscenes_multimodal_dataset.py LINE 103

log_config = dict(
    # Overwriting the default one in _base_/default_runtime.py
    # so that we use our custom *ImageLoggerHook.
    _delete_=True,
    interval=1, # 350
    hooks=[
        dict(type="TextImageLoggerHook"),
        # dict(type="TensorboardImageLoggerHook2")
        dict(
            type="WandbImageLoggerHookV2",
            init_kwargs=dict(
                project="distillnerf"
            ),
            log_artifact=False,
        ),
    ],
)

custom_imports = dict(
    imports=[
        "projects.DistillNeRF.models",
        "projects.DistillNeRF.pipelines",
        "projects.DistillNeRF.datasets",
        "projects.DistillNeRF.losses",
        "projects.DistillNeRF.hooks",
        "projects.DistillNeRF.runners",
    ],
    allow_failed_imports=False,
)


# Different loss term inclusions (or not).
model = dict(
    type="DistillNerfModelWrapper",
    model_yaml_path="projects/DistillNeRF/configs/models/model_linearspace_foundation_feature.yaml",
    seg_model_path = "./aux_models",
    num_camera=6,
    num_input_seq=1,
    target_cam_temporal_idx=2,
    force_same_seq = True,
    all_prev_frames = False,
    # clip_pca_file_path="../aux_models/clip/clip_pca_dict.npz",
    render_foundation_model_feat='dino'
)

# optimizer settings copied from mmdetection3d/configs/_base_/schedules/schedule_2x.py
# This schedule is mainly used by models on nuScenes dataset.
optimizer = dict(type="Adam", lr=0.0002, betas=[0., 0.99], foreach=True)
# momentum_config = None
lr_config = dict(
    policy="step",
    warmup="linear",
    warmup_iters=1,
    warmup_ratio=1.0 / 1.0,
    step=[30, 60],
)

# Whether to use Automatic Mixed Precision (AMP) or not.
do_fp16: bool = True
if do_fp16:
    optimizer_config = dict(
        type="GradientCumulativeFp16OptimizerHook",
        loss_scale="dynamic",
        cumulative_iters=1,  # 16 for a batch size of 128 with 8 GPUs.
        # max_norm=10 is better for SECOND.
        grad_clip=dict(max_norm=35, norm_type=2),
    )
else:
    optimizer_config = dict(
        type="GradientCumulativeOptimizerHook",
        cumulative_iters=1,  # 16 for a batch size of 128 with 8 GPUs.
        # max_norm=10 is better for SECOND.
        grad_clip=dict(max_norm=35, norm_type=2),
    )

# Distributed parameters
find_unused_parameters = True
log_level = "INFO"

# Runtime settings
# runner = dict(type="EpochBasedRunner", max_epochs=1000)
runner = dict(type="EpochBasedRunnerValFirst", max_epochs=1000)
# runner = dict(type="EpochBasedRunnerValFirstPassIter", max_epochs=1000)



# How often to save checkpoints
# checkpoint_config = dict(interval=1)
checkpoint_config = dict(interval=1000, by_epoch=False, max_keep_ckpts=5)
# checkpoint_config = dict(interval=1000, by_epoch=False, max_keep_ckpts=5)


