import argparse
import mmdet
import torch
from mmcv import Config, DictAction
from mmcv.parallel import MMDataParallel
from mmcv.runner import load_checkpoint, wrap_fp16_model
from mmdet3d.datasets import build_dataloader, build_dataset
from mmdet3d.models import build_model
from mmdet.datasets import replace_ImageToTensor
if mmdet.__version__ > "2.23.0":
    # If mmdet version > 2.23.0, setup_multi_processes would be imported and
    # used from mmdet instead of mmdet3d.
    from mmdet.utils import setup_multi_processes
else:
    from mmdet3d.utils import setup_multi_processes

try:
    # If mmdet version > 2.23.0, compat_cfg would be imported and
    # used from mmdet instead of mmdet3d.
    from mmdet.utils import compat_cfg
except ImportError:
    from mmdet3d.utils import compat_cfg

import torch

def create_eval_dict():

    eval_dict = {
        'coarse_depth_lidar_error_abs_rel': [],
        'coarse_depth_emernerf_error_abs_rel': [],
        'coarse_depth_emernerf_inner_error_abs_rel': [],
        'coarse_depth_lidar_abs_rel': [],
        'coarse_depth_lidar_sq_rel': [],
        'coarse_depth_lidar_rmse': [],
        'coarse_depth_lidar_rmse_log': [],
        'coarse_depth_lidar_a1': [],
        'coarse_depth_lidar_a2': [],
        'coarse_depth_lidar_a3': [],
        'coarse_depth_emernerf_abs_rel': [],
        'coarse_depth_emernerf_sq_rel': [],
        'coarse_depth_emernerf_rmse': [],
        'coarse_depth_emernerf_rmse_log': [],
        'coarse_depth_emernerf_a1': [],
        'coarse_depth_emernerf_a2': [],
        'coarse_depth_emernerf_a3': [],
        'psnr': [],
        'ssim': [],
        'target_depth_lidar_error_abs_rel': [],
        'target_depth_emernerf_error_abs_rel': [],
        'target_depth_emernerf_inner_error_abs_rel': [],
        'target_lidar_abs_rel': [],
        'target_lidar_sq_rel': [],
        'target_lidar_rmse': [],
        'target_lidar_rmse_log': [],
        'target_lidar_a1': [],
        'target_lidar_a2': [],
        'target_lidar_a3': [],
        'target_emernerf_abs_rel': [],
        'target_emernerf_sq_rel': [],
        'target_emernerf_rmse': [],
        'target_emernerf_rmse_log': [],
        'target_emernerf_a1': [],
        'target_emernerf_a2': [],
        'target_emernerf_a3': [],
        'target_emernerf_inner_abs_rel': [],
        'target_emernerf_inner_sq_rel': [],
        'target_emernerf_inner_rmse': [],
        'target_emernerf_inner_rmse_log': [],
        'target_emernerf_inner_a1': [],
        'target_emernerf_inner_a2': [],
        'target_emernerf_inner_a3': []
    }
    return eval_dict

def parse_args():
    parser = argparse.ArgumentParser(description="MMDet test (and eval) a model")
    parser.add_argument("config", help="test config file path")
    parser.add_argument("checkpoint", help="checkpoint file")
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    args = parser.parse_args()

    return args


def main():
    args = parse_args()

    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    cfg = compat_cfg(cfg)

    # set multi-process settings
    setup_multi_processes(cfg)

    # set cudnn_benchmark
    if cfg.get("cudnn_benchmark", False):
        torch.backends.cudnn.benchmark = True

    # cfg.model.pretrained = None
    cfg.gpu_ids = [0]

    test_dataloader_default_args = dict(
        samples_per_gpu=1, workers_per_gpu=2, dist=False, shuffle=False
    )

    # in case the test dataset is concatenated
    if isinstance(cfg.data.test, dict):
        cfg.data.test.test_mode = True
        if cfg.data.test_dataloader.get("samples_per_gpu", 1) > 1:
            # Replace 'ImageToTensor' to 'DefaultFormatBundle'
            cfg.data.test.pipeline = replace_ImageToTensor(cfg.data.test.pipeline)
    elif isinstance(cfg.data.test, list):
        for ds_cfg in cfg.data.test:
            ds_cfg.test_mode = True
        if cfg.data.test_dataloader.get("samples_per_gpu", 1) > 1:
            for ds_cfg in cfg.data.test:
                ds_cfg.pipeline = replace_ImageToTensor(ds_cfg.pipeline)

    test_loader_cfg = {
        **test_dataloader_default_args,
        **cfg.data.get("test_dataloader", {}),
    }

    # build the dataloader
    dataset = build_dataset(cfg.data.test)
    # dataset = build_dataset(cfg.data.train)
    test_loader_cfg['workers_per_gpu'] = 0
    data_loader = build_dataloader(dataset, **test_loader_cfg)

    # build the model and load checkpoint
    cfg.model.train_cfg = None
    model = build_model(cfg.model, test_cfg=cfg.get("test_cfg"))
    fp16_cfg = cfg.get("fp16", None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    # load the checkpoint
    checkpoint = torch.load(args.checkpoint)
    if args.checkpoint != 'None': 
        checkpoint = load_checkpoint(model, args.checkpoint, map_location="cpu")
        # old versions did not save class info in checkpoints, this walkaround is
        # for backward compatibility
        if "CLASSES" in checkpoint.get("meta", {}):
            model.CLASSES = checkpoint["meta"]["CLASSES"]
        else:
            model.CLASSES = dataset.CLASSES
        # palette for visualization in segmentation tasks
        if "PALETTE" in checkpoint.get("meta", {}):
            model.PALETTE = checkpoint["meta"]["PALETTE"]
        elif hasattr(dataset, "PALETTE"):
            # segmentation dataset has `PALETTE` attribute
            model.PALETTE = dataset.PALETTE

    model = MMDataParallel(model, device_ids=cfg.gpu_ids)

    # import pdb; pdb.set_trace()
    PSNR_list = []
    SSIM_list = []
    eval_dict = create_eval_dict()
    for i, data in enumerate(data_loader):

        data['is_test'] = True
        print(i)
        with torch.no_grad():
            result_dict = model(return_loss=False, rescale=True, **data)[0]
            # raise

        PSNR_list.append(result_dict['psnr'])
        SSIM_list.append(result_dict['ssim'])
        for key in result_dict.keys():
            if key in eval_dict.keys():
                eval_dict[key].append(result_dict[key])
            else:
                eval_dict[key] = [result_dict[key]]
    print('PSNR:', sum(PSNR_list)/len(PSNR_list))
    print('SSIM:', sum(SSIM_list)/len(SSIM_list))


if __name__ == "__main__":
    main()
