import argparse
import os
import matplotlib.pyplot as plt
import numpy as np
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

import os

import numpy as np
import torch


def create_translation_novel_view(delta_distance = 1, sequence_num = 3, init_dist = 0, axis = 'y'):
    if axis == 'y': # forward
        translation_matrix = [ np.array([
                                    [1, 0, 0, 0],
                                    [0, 1, 0, delta_distance*i+init_dist],
                                    [0, 0, 1, 0],
                                    [0, 0, 0, 1]
                                ]) for i in range(sequence_num)
                            ]
    if axis == 'x': # right
        translation_matrix = [ np.array([
                                    [1, 0, 0, delta_distance*i+init_dist],
                                    [0, 1, 0, 0],
                                    [0, 0, 1, 0],
                                    [0, 0, 0, 1]
                                ]) for i in range(sequence_num)
                            ]
    if axis == 'z':  # up
        translation_matrix = [ np.array([
                                    [1, 0, 0, 0],
                                    [0, 1, 0, 0],
                                    [0, 0, 1, delta_distance*i+init_dist],
                                    [0, 0, 0, 1]
                                ]) for i in range(sequence_num)
                            ]
    
    return translation_matrix


def create_circle_novel_view(index=0):
    step_size = 0.1
    radius = 1
    
    # Define the points for a circular trajectory
    traj = []
    for i in range(int(radius//step_size)+1):
        traj.append([i*step_size, 0, 0])
    for i in range(0, 31):
        angle = i * np.pi / 30
        x = radius * np.cos(angle)
        z = radius * np.sin(angle)
        traj.append([x, 0, z])
    for i in range(int(radius//step_size)):
        traj.append([-radius+(i+1)*step_size, 0, 0])
    
    transform_matrices = []
    start = index
    end = min(index + 3, len(traj))
    
    for point in traj[start:end]:
        # Calculate the translation matrix
        translation_matrix = np.array([
            [1, 0, 0, point[0]],
            [0, 1, 0, point[1]],
            [0, 0, 1, point[2]],
            [0, 0, 0, 1]
        ])

        transform_matrices.append(translation_matrix)
    
    return transform_matrices


def create_left_right_novel_view(index=0):
    step_size = 0.1
    radius = 1.3
    left_traj = []
    
    # Define the points for a left trajectory
    for i in range(int(radius//step_size)+1):
        left_traj.append([i*step_size, 0, 0])
    left_traj.extend(left_traj[::-1])

    # Define the points for a right trajectory
    right_traj = []
    for i in range(int(radius//step_size)+1):
        right_traj.append([-i*step_size, 0, 0])
    right_traj.extend(right_traj[::-1])

    traj = []
    traj.extend(left_traj)
    traj.extend(right_traj)
    
    transform_matrices = []
    start = index
    end = min(index + 3, len(traj))
    for point in traj[start:end]:
        # Calculate the translation matrix
        translation_matrix = np.array([
            [1, 0, 0, point[0]],
            [0, 1, 0, point[1]],
            [0, 0, 1, point[2]],
            [0, 0, 0, 1]
        ])

        transform_matrices.append(translation_matrix)

    return transform_matrices


def parse_args():
    parser = argparse.ArgumentParser(description="MMDet test (and eval) a model")
    parser.add_argument("config", help="test config file path")
    parser.add_argument("checkpoint", help="checkpoint file")
    parser.add_argument("index", nargs="?", type=int, default=0, help="Index (optional, default: 0)")
    # parser.add_argument("index", default=0, help="to generate novel views automatically")
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
    # import pdb; pdb.set_trace()
    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(dataset, **test_loader_cfg)

    # build the model and load checkpoint
    cfg.model.train_cfg = None
    # import pdb; pdb.set_trace()
    model = build_model(cfg.model, test_cfg=cfg.get("test_cfg"))
    fp16_cfg = cfg.get("fp16", None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
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


    for i, data in enumerate(data_loader):

        with torch.no_grad():
            ''' translation in one direction '''
            # direction = 'z'
            # init_dist = 9       # 0, 3, 6, 9
            # translation_matrix = create_translation_novel_view(init_dist=init_dist, axis=direction)

            ''' draw a circle around  '''
            # direction = 'circle'
            # init_dist = args.index       # 0, 3, 6, 9, 12
            # translation_matrix = create_circle_novel_view(index=init_dist)

            ''' move left and right   '''
            init_dist = args.index       # 0, 3, 6, 9, 12
            translation_matrix = create_left_right_novel_view(index=init_dist)
            
            data['novel_view_traj'] = translation_matrix
            result_dict = model(return_loss=False, rescale=True, **data)[0]

            raise


if __name__ == "__main__":
    main()