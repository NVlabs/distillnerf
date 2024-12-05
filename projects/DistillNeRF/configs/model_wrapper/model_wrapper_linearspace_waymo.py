_base_ = [
    "../../configs/_base_/default_runtime.py",
    "../datasets/dataset_config_waymo.py",
]
# model_wrapper_noemernerfdepth
data = dict(
    train=dict(
        num_prev_frames=2,
        num_next_frames=2,
        subselect_group_num=3),
    val=dict(
        num_prev_frames=2,
        num_next_frames=2,
        subselect_group_num=3),
    test=dict(
        num_prev_frames=2,
        num_next_frames=2,
        subselect_group_num=3))

log_config = dict(
    # Overwriting the default one in _base_/default_runtime.py
    # so that we use our custom *ImageLoggerHook.
    _delete_=True,
    interval=1,
    hooks=[
        dict(type="TextImageLoggerHook"),
        # dict(type="TensorboardImageLoggerHook2")
        dict(
            type="WandbImageLoggerHookV2",
            init_kwargs=dict(
                project="distillnerf",
            ),
            log_artifact=False,
        ),
    ],
)
# lidar_depth_loss_coef
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
    model_yaml_path="projects/DistillNeRF/configs/models/model_linearspace_waymo.yaml",
    seg_model_path = "./aux_models",
    num_camera=3,
    num_input_seq=1,
    target_cam_temporal_idx=2,
    force_same_seq = True,
    all_prev_frames = False,
    # rgb_clamp_0_1=True
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
find_unused_parameters = False
log_level = "INFO"

# Runtime settings
# runner = dict(type="EpochBasedRunner", max_epochs=1000)
# runner = dict(type="EpochBasedRunnerAnomoly", max_epochs=1000)
runner = dict(type="EpochBasedRunnerValFirst", max_epochs=1000)


# How often to save checkpoints
checkpoint_config = dict(interval=1050, by_epoch=False, max_keep_ckpts=5)
# checkpoint_config = dict(interval=1)

