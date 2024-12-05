# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import warnings

import mmcv
import torch
from mmcv import Config, DictAction
from mmcv.cnn import fuse_conv_bn
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import (get_dist_info, init_dist, load_checkpoint,
                         wrap_fp16_model)

import mmdet
from mmdet3d.apis import single_gpu_test
from mmdet3d.datasets import build_dataloader, build_dataset
from mmdet3d.models import build_model
from mmdet.apis import multi_gpu_test, set_random_seed
from mmdet.datasets import replace_ImageToTensor
import numpy as np

if mmdet.__version__ > '2.23.0':
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


def parse_args():
    parser = argparse.ArgumentParser(
        description='MMDet test (and eval) a model')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('--out', help='output result file in pickle format')
    parser.add_argument(
        '--fuse-conv-bn',
        action='store_true',
        help='Whether to fuse conv and bn, this will slightly increase'
        'the inference speed')
    parser.add_argument(
        '--gpu-ids',
        type=int,
        nargs='+',
        help='(Deprecated, please use --gpu-id) ids of gpus to use '
        '(only applicable to non-distributed training)')
    parser.add_argument(
        '--gpu-id',
        type=int,
        default=0,
        help='id of gpu to use '
        '(only applicable to non-distributed testing)')
    parser.add_argument(
        '--format-only',
        action='store_true',
        help='Format the output results without perform evaluation. It is'
        'useful when you want to format the result to a specific format and '
        'submit it to the test server')
    parser.add_argument(
        '--eval',
        type=str,
        nargs='+',
        help='evaluation metrics, which depends on the dataset, e.g., "bbox",'
        ' "segm", "proposal" for COCO, and "mAP", "recall" for PASCAL VOC')
    parser.add_argument('--show', action='store_true', help='show results')
    parser.add_argument(
        '--show-dir', help='directory where results will be saved')
    parser.add_argument(
        '--gpu-collect',
        action='store_true',
        help='whether to use gpu to collect results.')
    parser.add_argument(
        '--tmpdir',
        help='tmp directory used for collecting results from multiple '
        'workers, available when gpu-collect is not specified')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument(
        '--deterministic',
        action='store_true',
        help='whether to set deterministic options for CUDNN backend.')
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
    parser.add_argument(
        '--options',
        nargs='+',
        action=DictAction,
        help='custom options for evaluation, the key-value pair in xxx=yyy '
        'format will be kwargs for dataset.evaluate() function (deprecate), '
        'change to --eval-options instead.')
    parser.add_argument(
        '--eval-options',
        nargs='+',
        action=DictAction,
        help='custom options for evaluation, the key-value pair in xxx=yyy '
        'format will be kwargs for dataset.evaluate() function')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    if args.options and args.eval_options:
        raise ValueError(
            '--options and --eval-options cannot be both specified, '
            '--options is deprecated in favor of --eval-options')
    if args.options:
        warnings.warn('--options is deprecated in favor of --eval-options')
        args.eval_options = args.options
    return args


def main():
    args = parse_args()

    assert args.out or args.eval or args.format_only or args.show \
        or args.show_dir, \
        ('Please specify at least one operation (save/eval/format/show the '
         'results / save the results) with the argument "--out", "--eval"'
         ', "--format-only", "--show" or "--show-dir"')

    if args.eval and args.format_only:
        raise ValueError('--eval and --format_only cannot be both specified')

    if args.out is not None and not args.out.endswith(('.pkl', '.pickle')):
        raise ValueError('The output file must be a pkl file.')

    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    cfg = compat_cfg(cfg)

    # set multi-process settings
    setup_multi_processes(cfg)

    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    # cfg.model.pretrained = None

    if args.gpu_ids is not None:
        cfg.gpu_ids = args.gpu_ids[0:1]
        warnings.warn('`--gpu-ids` is deprecated, please use `--gpu-id`. '
                      'Because we only support single GPU mode in '
                      'non-distributed testing. Use the first GPU '
                      'in `gpu_ids` now.')
    else:
        cfg.gpu_ids = [args.gpu_id]

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)

    test_dataloader_default_args = dict(
        samples_per_gpu=1, workers_per_gpu=2, dist=distributed, shuffle=False)

    # in case the test dataset is concatenated
    if isinstance(cfg.data.test, dict):
        cfg.data.test.test_mode = True
        if cfg.data.test_dataloader.get('samples_per_gpu', 1) > 1:
            # Replace 'ImageToTensor' to 'DefaultFormatBundle'
            cfg.data.test.pipeline = replace_ImageToTensor(
                cfg.data.test.pipeline)
    elif isinstance(cfg.data.test, list):
        for ds_cfg in cfg.data.test:
            ds_cfg.test_mode = True
        if cfg.data.test_dataloader.get('samples_per_gpu', 1) > 1:
            for ds_cfg in cfg.data.test:
                ds_cfg.pipeline = replace_ImageToTensor(ds_cfg.pipeline)

    test_loader_cfg = {
        **test_dataloader_default_args,
        **cfg.data.get('test_dataloader', {})
    }

    # set random seeds
    if args.seed is not None:
        set_random_seed(args.seed, deterministic=args.deterministic)

    # build the dataloader
    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(dataset, **test_loader_cfg)

    # build the model and load checkpoint
    cfg.model.train_cfg = None
    model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')
    if args.fuse_conv_bn:
        model = fuse_conv_bn(model)
    # old versions did not save class info in checkpoints, this walkaround is
    # for backward compatibility
    if 'CLASSES' in checkpoint.get('meta', {}):
        model.CLASSES = checkpoint['meta']['CLASSES']
    else:
        model.CLASSES = dataset.CLASSES
    # palette for visualization in segmentation tasks
    if 'PALETTE' in checkpoint.get('meta', {}):
        model.PALETTE = checkpoint['meta']['PALETTE']
    elif hasattr(dataset, 'PALETTE'):
        # segmentation dataset has `PALETTE` attribute
        model.PALETTE = dataset.PALETTE
    import pdb; pdb.set_trace()
    if not distributed:
        model = MMDataParallel(model, device_ids=cfg.gpu_ids)
        outputs = single_gpu_test(model, data_loader, args.show, args.show_dir)
    else:
        model = MMDistributedDataParallel(
            model.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False)
        outputs = multi_gpu_test(model, data_loader, args.tmpdir,
                                 args.gpu_collect)

    rank, _ = get_dist_info()
    if rank == 0:
        if args.out:
            print(f'\nwriting results to {args.out}')
            mmcv.dump(outputs, args.out)
        kwargs = {} if args.eval_options is None else args.eval_options
        if args.format_only:
            dataset.format_results(outputs, **kwargs)
        psnr_lst = []
        ssim_lst = []
        next_psnr_lst = []
        next_ssim_lst = []
        input_depth_error_lst = []
        target_depth_error_lst = []
        for one_output in outputs:
            psnr_lst.append(one_output['psnr'])
            ssim_lst.append(one_output['ssim'])
            if 'next_psnr' in one_output: next_psnr_lst.append(one_output['next_psnr'])
            if 'next_ssim' in one_output: next_ssim_lst.append(one_output['next_ssim'])
            if 'input_depth_error' in one_output: input_depth_error_lst.append(one_output['input_depth_error'])
            if 'target_depth_error' in one_output: target_depth_error_lst.append(one_output['target_depth_error'])
        print(f'psnr: {sum(psnr_lst) / len(psnr_lst)}') 
        print(f'ssim: {sum(ssim_lst) / len(ssim_lst)}')
        if 'next_psnr' in one_output: print(f'next psnr: {sum(next_psnr_lst) / len(next_psnr_lst)}') 
        if 'next_ssim' in one_output: print(f'next ssim: {sum(next_ssim_lst) / len(next_ssim_lst)}')
        if 'next_psnr' in one_output: print("input_depth_error_lst: ", np.mean(np.stack(input_depth_error_lst), 0))
        if 'next_ssim' in one_output: print("target_depth_error_lst: ", np.mean(np.stack(target_depth_error_lst), 0))


        if 'target_lidar_abs_rel' in one_output.keys():
            target_lidar_abs_rel_lst = []
            target_lidar_sq_rel_lst = []
            target_lidar_rmse_lst = []
            target_lidar_rmse_log_lst = []
            target_lidar_a1_lst = []
            target_lidar_a2_lst = []
            target_lidar_a3_lst = []

            input_lidar_abs_rel_lst = []
            input_lidar_sq_rel_lst = []
            input_lidar_rmse_lst = []
            input_lidar_rmse_log_lst = []
            input_lidar_a1_lst = []
            input_lidar_a2_lst = []
            input_lidar_a3_lst = []

            target_emernerf_abs_rel_lst = []
            target_emernerf_sq_rel_lst = []
            target_emernerf_rmse_lst = []
            target_emernerf_rmse_log_lst = []
            target_emernerf_a1_lst = []
            target_emernerf_a2_lst = []
            target_emernerf_a3_lst = []

            input_emernerf_abs_rel_lst = []
            input_emernerf_sq_rel_lst = []
            input_emernerf_rmse_lst = []
            input_emernerf_rmse_log_lst = []
            input_emernerf_a1_lst = []
            input_emernerf_a2_lst = []
            input_emernerf_a3_lst = []

            lidar_emernerf_abs_rel_lst = []
            lidar_emernerf_sq_rel_lst = []
            lidar_emernerf_rmse_lst = []
            lidar_emernerf_rmse_log_lst = []
            lidar_emernerf_a1_lst = []
            lidar_emernerf_a2_lst = []
            lidar_emernerf_a3_lst = []
            
            for one_output in outputs:
                target_lidar_abs_rel_lst.append(one_output['target_lidar_abs_rel'])
                target_lidar_sq_rel_lst.append(one_output['target_lidar_sq_rel'])
                target_lidar_rmse_lst.append(one_output['target_lidar_rmse'])
                target_lidar_rmse_log_lst.append(one_output['target_lidar_rmse_log'])
                target_lidar_a1_lst.append(one_output['target_lidar_a1'])
                target_lidar_a2_lst.append(one_output['target_lidar_a2'])
                target_lidar_a3_lst.append(one_output['target_lidar_a3'])

                input_lidar_abs_rel_lst.append(one_output['input_lidar_abs_rel'])
                input_lidar_sq_rel_lst.append(one_output['input_lidar_sq_rel'])
                input_lidar_rmse_lst.append(one_output['input_lidar_rmse'])
                input_lidar_rmse_log_lst.append(one_output['input_lidar_rmse_log'])
                input_lidar_a1_lst.append(one_output['input_lidar_a1'])
                input_lidar_a2_lst.append(one_output['input_lidar_a2'])
                input_lidar_a3_lst.append(one_output['input_lidar_a3'])

                target_emernerf_abs_rel_lst.append(one_output['target_emernerf_abs_rel'])
                target_emernerf_sq_rel_lst.append(one_output['target_emernerf_sq_rel'])
                target_emernerf_rmse_lst.append(one_output['target_emernerf_rmse'])
                target_emernerf_rmse_log_lst.append(one_output['target_emernerf_rmse_log'])
                target_emernerf_a1_lst.append(one_output['target_emernerf_a1'])
                target_emernerf_a2_lst.append(one_output['target_emernerf_a2'])
                target_emernerf_a3_lst.append(one_output['target_emernerf_a3'])

                input_emernerf_abs_rel_lst.append(one_output['input_emernerf_abs_rel'])
                input_emernerf_sq_rel_lst.append(one_output['input_emernerf_sq_rel'])
                input_emernerf_rmse_lst.append(one_output['input_emernerf_rmse'])
                input_emernerf_rmse_log_lst.append(one_output['input_emernerf_rmse_log'])
                input_emernerf_a1_lst.append(one_output['input_emernerf_a1'])
                input_emernerf_a2_lst.append(one_output['input_emernerf_a2'])
                input_emernerf_a3_lst.append(one_output['input_emernerf_a3'])

                lidar_emernerf_abs_rel_lst.append(one_output['lidar_emernerf_abs_rel'])
                lidar_emernerf_sq_rel_lst.append(one_output['lidar_emernerf_sq_rel'])
                lidar_emernerf_rmse_lst.append(one_output['lidar_emernerf_rmse'])
                lidar_emernerf_rmse_log_lst.append(one_output['lidar_emernerf_rmse_log'])
                lidar_emernerf_a1_lst.append(one_output['lidar_emernerf_a1'])
                lidar_emernerf_a2_lst.append(one_output['lidar_emernerf_a2'])
                lidar_emernerf_a3_lst.append(one_output['lidar_emernerf_a3'])

            print(f'target_lidar_abs_rel_lst: {sum(target_lidar_abs_rel_lst) / len(target_lidar_abs_rel_lst)}') 
            print(f'target_lidar_sq_rel_lst: {sum(target_lidar_sq_rel_lst) / len(target_lidar_sq_rel_lst)}') 
            print(f'target_lidar_rmse_lst: {sum(target_lidar_rmse_lst) / len(target_lidar_rmse_lst)}') 
            print(f'target_lidar_rmse_log_lst: {sum(target_lidar_rmse_log_lst) / len(target_lidar_rmse_log_lst)}') 
            print(f'target_lidar_a1_lst: {sum(target_lidar_a1_lst) / len(target_lidar_a1_lst)}') 
            print(f'target_lidar_a2_lst: {sum(target_lidar_a2_lst) / len(target_lidar_a2_lst)}') 
            print(f'target_lidar_a3_lst: {sum(target_lidar_a3_lst) / len(target_lidar_a3_lst)}') 

            print()
            print(f'input_lidar_abs_rel_lst: {sum(input_lidar_abs_rel_lst) / len(input_lidar_abs_rel_lst)}') 
            print(f'input_lidar_sq_rel_lst: {sum(input_lidar_sq_rel_lst) / len(input_lidar_sq_rel_lst)}') 
            print(f'input_lidar_rmse_lst: {sum(input_lidar_rmse_lst) / len(input_lidar_rmse_lst)}') 
            print(f'input_lidar_rmse_log_lst: {sum(input_lidar_rmse_log_lst) / len(input_lidar_rmse_log_lst)}') 
            print(f'input_lidar_a1_lst: {sum(input_lidar_a1_lst) / len(input_lidar_a1_lst)}') 
            print(f'input_lidar_a2_lst: {sum(input_lidar_a2_lst) / len(input_lidar_a2_lst)}') 
            print(f'input_lidar_a3_lst: {sum(input_lidar_a3_lst) / len(input_lidar_a3_lst)}') 

            print()
            print(f'target_emernerf_abs_rel_lst: {sum(target_emernerf_abs_rel_lst) / len(target_emernerf_abs_rel_lst)}')
            print(f'target_emernerf_sq_rel_lst: {sum(target_emernerf_sq_rel_lst) / len(target_emernerf_sq_rel_lst)}')
            print(f'target_emernerf_rmse_lst: {sum(target_emernerf_rmse_lst) / len(target_emernerf_rmse_lst)}')
            print(f'target_emernerf_rmse_log_lst: {sum(target_emernerf_rmse_log_lst) / len(target_emernerf_rmse_log_lst)}')
            print(f'target_emernerf_a1_lst: {sum(target_emernerf_a1_lst) / len(target_emernerf_a1_lst)}')
            print(f'target_emernerf_a2_lst: {sum(target_emernerf_a2_lst) / len(target_emernerf_a2_lst)}')
            print(f'target_emernerf_a3_lst: {sum(target_emernerf_a3_lst) / len(target_emernerf_a3_lst)}')

            print()
            print(f'input_emernerf_abs_rel_lst: {sum(input_emernerf_abs_rel_lst) / len(input_emernerf_abs_rel_lst)}')
            print(f'input_emernerf_sq_rel_lst: {sum(input_emernerf_sq_rel_lst) / len(input_emernerf_sq_rel_lst)}')
            print(f'input_emernerf_rmse_lst: {sum(input_emernerf_rmse_lst) / len(input_emernerf_rmse_lst)}')
            print(f'input_emernerf_rmse_log_lst: {sum(input_emernerf_rmse_log_lst) / len(input_emernerf_rmse_log_lst)}')
            print(f'input_emernerf_a1_lst: {sum(input_emernerf_a1_lst) / len(input_emernerf_a1_lst)}')
            print(f'input_emernerf_a2_lst: {sum(input_emernerf_a2_lst) / len(input_emernerf_a2_lst)}')
            print(f'input_emernerf_a3_lst: {sum(input_emernerf_a3_lst) / len(input_emernerf_a3_lst)}')


            print()
            print(f'lidar_emernerf_abs_rel_lst: {sum(lidar_emernerf_abs_rel_lst) / len(lidar_emernerf_abs_rel_lst)}')
            print(f'lidar_emernerf_sq_rel_lst: {sum(lidar_emernerf_sq_rel_lst) / len(lidar_emernerf_sq_rel_lst)}')
            print(f'lidar_emernerf_rmse_lst: {sum(lidar_emernerf_rmse_lst) / len(lidar_emernerf_rmse_lst)}')
            print(f'lidar_emernerf_rmse_log_lst: {sum(lidar_emernerf_rmse_log_lst) / len(lidar_emernerf_rmse_log_lst)}')
            print(f'lidar_emernerf_a1_lst: {sum(lidar_emernerf_a1_lst) / len(lidar_emernerf_a1_lst)}')
            print(f'lidar_emernerf_a2_lst: {sum(lidar_emernerf_a2_lst) / len(lidar_emernerf_a2_lst)}')
            print(f'lidar_emernerf_a3_lst: {sum(lidar_emernerf_a3_lst) / len(lidar_emernerf_a3_lst)}')
            
        import pdb; pdb.set_trace()
        # if args.eval:
        #     eval_kwargs = cfg.get('evaluation', {}).copy()
        #     # hard-code way to remove EvalHook args
        #     for key in [
        #             'interval', 'tmpdir', 'start', 'gpu_collect', 'save_best',
        #             'rule'
        #     ]:
        #         eval_kwargs.pop(key, None)
        #     eval_kwargs.update(dict(metric=args.eval, **kwargs))
        #     print(dataset.evaluate(outputs, **eval_kwargs))


if __name__ == '__main__':
    main()
