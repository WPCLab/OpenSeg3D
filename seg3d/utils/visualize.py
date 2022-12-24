import os

import torch
import numpy as np

import open3d as o3d
from open3d import geometry

from seg3d.utils.pointops_utils import get_voxel_centers


def draw_points(palette, data_dict, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    pcd = geometry.PointCloud()

    points = data_dict['points']
    if isinstance(points, torch.Tensor):
        points = points.cpu().numpy()

    points = points.copy()
    pcd.points = o3d.utility.Vector3dVector(points[:, :3])

    point_labels = data_dict['point_labels']
    if isinstance(point_labels, torch.Tensor):
        point_labels = point_labels.cpu().numpy()

    points_colors = np.zeros((point_labels.shape[0], 3), dtype=np.float32)
    for i in range(point_labels.shape[0]):
        if point_labels[i] == 255:
            continue
        points_colors[i] = np.array(palette[point_labels[i]])

    # normalize to [0, 1] for open3d drawing
    if not ((points_colors >= 0.0) & (points_colors <= 1.0)).all():
        points_colors /= 255.0
    pcd.colors = o3d.utility.Vector3dVector(points_colors)

    output_file = os.path.join(output_dir, data_dict['filename'] + '.pcd')
    o3d.io.write_point_cloud(output_file, pcd)


def draw_voxels(palette, voxel_size, pc_cloud_range, data_dict, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    pcd = geometry.PointCloud()

    voxel_coords = data_dict['voxel_coords']
    voxel_centers = get_voxel_centers(voxel_coords, 1.0, voxel_size, pc_cloud_range)

    points = voxel_centers.numpy().copy()
    pcd.points = o3d.utility.Vector3dVector(points[:, :3])

    voxel_labels = data_dict['voxel_labels']
    if isinstance(voxel_labels, torch.Tensor):
        voxel_labels = voxel_labels.cpu().numpy()

    voxel_colors = np.zeros((voxel_labels.shape[0], 3), dtype=np.float32)
    for i in range(voxel_labels.shape[0]):
        if voxel_labels[i] == 255:
            continue
        voxel_colors[i] = np.array(palette[voxel_labels[i]])

    # normalize to [0, 1] for open3d drawing
    if not ((voxel_colors >= 0.0) & (voxel_colors <= 1.0)).all():
        voxel_colors /= 255.0
    pcd.colors = o3d.utility.Vector3dVector(voxel_colors)

    output_file = os.path.join(output_dir, data_dict['filename'] + '.pcd')
    o3d.io.write_point_cloud(output_file, pcd)
