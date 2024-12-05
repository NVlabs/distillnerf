# Copyright (c) 2019,20 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
from kaolin.ops.spc.points import quantize_points, points_to_morton, morton_to_points, unbatched_points_to_octree
from kaolin.rep.spc import Spc

def custom_unbatched_pointcloud_to_spc(pointcloud, level, features=None, mode='mean'):
    r"""This function takes as input a single point-cloud - a set of continuous coordinates in 3D,
    and coverts it into a :ref:`Structured Point Cloud (SPC)<spc>`, a compressed octree representation where
    the point cloud coordinates are quantized to integer coordinates.

    Point coordinates are expected to be normalized to the range :math:`[-1, 1]`.
    If a point is out of the range :math:`[-1, 1]` it will be clipped to it.

    If ``features`` are specified, the current implementation will average features
    of points that inhabit the same quantized bucket.

    Args:
        pointclouds (torch.Tensor):
            An unbatched pointcloud, of shape :math:`(\text{num_points}, 3)`.
            Coordinates are expected to be normalized to the range :math:`[-1, 1]`.
        level (int):
            Maximum number of levels to use in octree hierarchy.
        features (optional, torch.Tensor):
            Feature vector containing information per point, of shape
            :math:`(\text{num_points}, \text{feat_dim})`.

    Returns:
        (kaolin.rep.Spc):
        A Structured Point Cloud (SPC) object, holding a single-item batch.
    """
    # pointcloud: [-1, 1]
    # points: index in the octree
    points = quantize_points(pointcloud.contiguous(), level)

    # Avoid duplications if cells occupy more than one point
    unique, unique_keys, unique_counts = torch.unique(points.contiguous(), dim=0,
                                                      return_inverse=True, return_counts=True)

    # Create octree hierarchy
    morton, keys = torch.sort(points_to_morton(unique.contiguous()).contiguous())
    points = morton_to_points(morton.contiguous())
    octree = unbatched_points_to_octree(points, level, sorted=True)

    # Organize features for octree leaf nodes
    # Feature fusion of multiple points sharing the same cell is consolidated here, assumes mean averaging
    # Promote to double precision dtype to avoid rounding errors
    feat_dtype = features.dtype
    is_fp = features.is_floating_point()
    feat = torch.zeros(unique.shape[0], features.shape[1], device=features.device).double()
    if mode == 'mean':
        feat = feat.index_add_(0, unique_keys, features.double()) / unique_counts[..., None].double()
    elif mode == 'max':
        print(f"Taking MAX in SPC")
        feat = feat.index_reduce_(0, unique_keys, features.double(), 'amax', include_self=False)
    if not is_fp:
        feat = torch.round(feat)
    feat = feat.to(feat_dtype)
    feat = feat[keys]
    
    # A full SPC requires octree hierarchy + auxilary data structures
    lengths = torch.tensor([len(octree)], dtype=torch.int32)   # Single entry batch
    return Spc(octrees=octree, lengths=lengths, features=feat)