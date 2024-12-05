from pathlib import Path
from typing import Tuple

import numpy as np
import torch
from matplotlib import cm
from mmcv import Config
from mmcv.runner import HOOKS, LoggerHook, WandbLoggerHook, master_only
from torchvision.utils import make_grid
import time


def draw_depth_img(
    depth_img: torch.Tensor,
    bg_color: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    cmap_name: str = "rainbow_r",
) -> torch.Tensor:
    N, C, H, W = depth_img.shape
    assert C == 1, "Depth images can only have 1 channel"

    output_img = torch.empty((N * H * W, 3))

    # Assigning background RGB values.
    output_img[..., 0] = bg_color[0]
    output_img[..., 1] = bg_color[1]
    output_img[..., 2] = bg_color[2]

    depth_img = depth_img.flatten()
    value_mask = torch.isfinite(depth_img)
    masked_depth_img = depth_img[value_mask].cpu()
    max_val = masked_depth_img.max()

    cmap = cm.get_cmap(cmap_name)
    output_img[value_mask] = torch.from_numpy(
        cmap(masked_depth_img / max_val)[:, :3].astype(np.float32)
    )

    return output_img.view(N, H, W, 3).permute(0, 3, 1, 2)


@HOOKS.register_module()
class WandbImageLoggerHookV2(WandbLoggerHook):
    @master_only
    def before_run(self, runner) -> None:
        # Copied from LoggerHook (rather than calling super().before_run(runner)).
        for hook in runner.hooks[::-1]:
            if isinstance(hook, LoggerHook):
                hook.reset_flag = True
                break
        config = Config.fromfile(Path(runner.work_dir) / runner.meta["exp_name"])

        # Copied and modified from WandbLoggerHook.
        if self.wandb is None:
            self.import_wandb()
        # import pdb; pdb.set_trace()
        if 'name' in self.init_kwargs:
            name = self.init_kwargs['name']
        else:
            name = str(runner.meta["exp_name"]) + '_' + time.ctime().replace(' ', '_')
        if self.init_kwargs:
            # self.wandb.init(config=dict(config), name=name, **self.init_kwargs)
            self.wandb.init(config=dict(config), **self.init_kwargs)
        else:
            self.wandb.init(config=dict(config))

    @master_only
    def log(self, runner) -> None:
        tags = self.get_loggable_tags(runner)
        # train: only include loss, need to preprocess images and add metrics
        # val: inlcude loss, images, metrics, need to preprocess images

        run_mode: str = self.get_mode(runner)
        if run_mode == "train":
            outputs = runner.outputs
        else:
            outputs = runner.log_buffer.output

        grid_options = {"nrow": 3, "padding": 5}


        ''' record images during training '''
        if run_mode == "train": 

            ''' input images/depth '''
            tags[f"{run_mode}/input_imgs"] = self.wandb.Image(
                make_grid(outputs["input_imgs"], **grid_options)
            )
            # input depth: [b, n, h, w] -> [n, 1, h, w]
            tags[f"{run_mode}/input_coarse_depth_pred_imgs"] = self.wandb.Image(
                make_grid(
                    draw_depth_img(outputs["input_coarse_depth_pred_imgs"][0].unsqueeze(1)),
                    **grid_options,
                )
            )
            tags[f"{run_mode}/input_gt_lidar_depth_imgs"] = self.wandb.Image(
                make_grid(
                    draw_depth_img(outputs["input_gt_lidar_depth_imgs"][0].unsqueeze(1)),
                    **grid_options,
                )
            )
            tags[f"{run_mode}/input_emernerf_depth_imgs"] = self.wandb.Image(
                make_grid(
                    draw_depth_img(outputs["input_emernerf_depth_imgs"][0].unsqueeze(1)),
                    **grid_options,
                )
            )
            tags[f"{run_mode}/input_lidar_depth_abs_diff_imgs"] = self.wandb.Image(
                make_grid(
                    draw_depth_img(
                        torch.abs(outputs["input_gt_lidar_depth_imgs"][0].unsqueeze(1) - outputs["input_coarse_depth_pred_imgs"][0].unsqueeze(1)),
                        cmap_name="plasma",
                    ),
                    **grid_options,
                )
            )
            tags[f"{run_mode}/input_emernerf_depth_abs_diff_imgs"] = self.wandb.Image(
                make_grid(
                    draw_depth_img(
                        torch.abs(outputs["input_emernerf_depth_imgs"][0].unsqueeze(1) - outputs["input_coarse_depth_pred_imgs"][0].unsqueeze(1)),
                        cmap_name="plasma",
                    ),
                    **grid_options,
                )
            )

            ''' target images/depth '''
            if 'recon_imgs' in outputs:
                if outputs["recon_imgs"].shape[1] == 3:
                    # only visulize rgb here, skip foundation model feature image
                    # fm feature: torch.Size([1, 112, 200, 64])
                    # rgb: torch.Size([1, 3, 128, 228])
                    tags[f"{run_mode}/recon_imgs"] = self.wandb.Image(
                        make_grid(outputs["recon_imgs"].to(dtype=torch.float32), **grid_options)
                    )

                tags[f"{run_mode}/target_pred_depth_imgs"] = self.wandb.Image(
                    make_grid(
                        draw_depth_img(outputs["target_pred_depth_imgs"][0]),
                        **grid_options,
                    )
                )
                tags[f"{run_mode}/target_gt_lidar_depth_imgs"] = self.wandb.Image(
                    make_grid(
                        draw_depth_img(outputs["target_gt_lidar_depth_imgs"][0]),
                        **grid_options,
                    )
                )

                tags[f"{run_mode}/target_emernerf_depth_imgs"] = self.wandb.Image(
                    make_grid(
                        draw_depth_img(outputs["target_emernerf_depth_imgs"][0]),
                        **grid_options,
                    )
                )
                # import pdb; pdb.set_trace()

                tags[f"{run_mode}/target_lidar_depth_abs_diff_imgs"] = self.wandb.Image(
                    make_grid(
                        draw_depth_img(
                            torch.abs(outputs["target_gt_lidar_depth_imgs"][0] - outputs["target_pred_depth_imgs"][0]),
                            cmap_name="plasma",
                        ),
                        **grid_options,
                    )
                )

                tags[f"{run_mode}/target_emernerf_depth_abs_diff_imgs"] = self.wandb.Image(
                    make_grid(
                        draw_depth_img(
                            torch.abs(outputs["target_emernerf_depth_imgs"][0] - outputs["target_pred_depth_imgs"][0]),
                            cmap_name="plasma",
                        ),
                        **grid_options,
                    )
                )

        # record images
        if run_mode == "val": 
            for i in range(1):
                post_fix = '' if i == 0 else f"{i}"

                if "input_imgs" in outputs and len(outputs["input_imgs"]) > 0:

                    tags[f"{run_mode}/input_imgs"] = self.wandb.Image(
                        make_grid(outputs["input_imgs"], **grid_options)
                    )

                    tags[f"{run_mode}/input_coarse_depth_pred_imgs"] = self.wandb.Image(
                        make_grid(
                            draw_depth_img(outputs["input_coarse_depth_pred_imgs"][0].unsqueeze(1)),
                            **grid_options,
                        )
                    )
                    tags[f"{run_mode}/input_gt_lidar_depth_imgs"] = self.wandb.Image(
                        make_grid(
                            draw_depth_img(outputs["input_gt_lidar_depth_imgs"][0].unsqueeze(1)),
                            **grid_options,
                        )
                    )

                    tags[f"{run_mode}/input_emernerf_depth_imgs"] = self.wandb.Image(
                        make_grid(
                            draw_depth_img(outputs["input_emernerf_depth_imgs"][0].unsqueeze(1)),
                            **grid_options,
                        )
                    )


                    tags[f"{run_mode}/input_lidar_depth_abs_diff_imgs"] = self.wandb.Image(
                        make_grid(
                            draw_depth_img(
                                torch.abs(outputs["input_gt_lidar_depth_imgs"][0].unsqueeze(1) - outputs["input_coarse_depth_pred_imgs"][0].unsqueeze(1)),
                                cmap_name="plasma",
                            ),
                            **grid_options,
                        )
                    )

                    tags[f"{run_mode}/input_emernerf_depth_abs_diff_imgs"] = self.wandb.Image(
                        make_grid(
                            draw_depth_img(
                                torch.abs(outputs["input_emernerf_depth_imgs"][0].unsqueeze(1) - outputs["input_coarse_depth_pred_imgs"][0].unsqueeze(1)),
                                cmap_name="plasma",
                            ),
                            **grid_options,
                        )
                    )


                if 'recon_imgs' in outputs and f"recon_imgs" in outputs and len(outputs["recon_imgs"]) > 0:
                    if outputs["recon_imgs"].shape[1] == 3:
                        # only visulize rgb here, skip foundation model feature image
                        # fm feature: torch.Size([1, 112, 200, 64])
                        # rgb: torch.Size([1, 3, 128, 228])
                        tags[f"{run_mode}/recon_imgs"] = self.wandb.Image(
                            make_grid(outputs["recon_imgs"].to(dtype=torch.float32), **grid_options)
                        )

                    tags[f"{run_mode}/target_pred_depth_imgs"] = self.wandb.Image(
                        make_grid(
                            draw_depth_img(outputs["target_pred_depth_imgs"][0]),
                            **grid_options,
                        )
                    )
                    tags[f"{run_mode}/target_gt_lidar_depth_imgs"] = self.wandb.Image(
                        make_grid(
                            draw_depth_img(outputs["target_gt_lidar_depth_imgs"][0]),
                            **grid_options,
                        )
                    )

                    tags[f"{run_mode}/target_emernerf_depth_imgs"] = self.wandb.Image(
                        make_grid(
                            draw_depth_img(outputs["target_emernerf_depth_imgs"][0]),
                            **grid_options,
                        )
                    )

                    tags[f"{run_mode}/target_lidar_depth_abs_diff_imgs"] = self.wandb.Image(
                        make_grid(
                            draw_depth_img(
                                torch.abs(outputs["target_gt_lidar_depth_imgs"][0] - outputs["target_pred_depth_imgs"][0]),
                                cmap_name="plasma",
                            ),
                            **grid_options,
                        )
                    )

                    tags[f"{run_mode}/target_emernerf_depth_abs_diff_imgs"] = self.wandb.Image(
                        make_grid(
                            draw_depth_img(
                                torch.abs(outputs["target_emernerf_depth_imgs"][0] - outputs["target_pred_depth_imgs"][0]),
                                cmap_name="plasma",
                            ),
                            **grid_options,
                        )
                    )
                
                
        # add metrics
        if run_mode == "train":
            tags[f"{run_mode}/coarse_depth_lidar_error_abs_rel"] = outputs['coarse_depth_lidar_error_abs_rel']
            tags[f"{run_mode}/coarse_depth_emernerf_error_abs_rel"] = outputs['coarse_depth_emernerf_error_abs_rel']
            tags[f"{run_mode}/coarse_depth_emernerf_inner_error_abs_rel"] = outputs['coarse_depth_emernerf_inner_error_abs_rel']

            if 'psnr' in outputs: 
                tags[f"{run_mode}/psnr"] = outputs['psnr']
                tags[f"{run_mode}/ssim"] = outputs['ssim']
                tags[f"{run_mode}/target_depth_lidar_error_abs_rel"] = outputs['target_depth_lidar_error_abs_rel']
                tags[f"{run_mode}/target_depth_emernerf_error_abs_rel"] = outputs['target_depth_emernerf_error_abs_rel']
                tags[f"{run_mode}/target_depth_emernerf_inner_error_abs_rel"] = outputs['target_depth_emernerf_inner_error_abs_rel']
            if 'aux_psnr' in outputs:
                tags[f"{run_mode}/aux_psnr"] = outputs['aux_psnr']
                tags[f"{run_mode}/aux_ssim"] = outputs['aux_ssim']
                tags[f"{run_mode}/aux_target_mono_depth_lidar_error_abs_rel"] = outputs['aux_target_mono_depth_lidar_error_abs_rel']
                tags[f"{run_mode}/aux_target_mono_depth_emernerf_error_abs_rel"] = outputs['aux_target_mono_depth_emernerf_error_abs_rel']
                tags[f"{run_mode}/aux_target_mono_depth_emernerf_inner_error_abs_rel"] = outputs['aux_target_mono_depth_emernerf_inner_error_abs_rel']

        if tags:
            if self.with_step:
                self.wandb.log(tags, step=self.get_iter(runner), commit=self.commit)
            else:
                tags["global_step"] = self.get_iter(runner)
                self.wandb.log(tags, commit=self.commit)
