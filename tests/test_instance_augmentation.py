import os
import glob

from tqdm import tqdm
import numpy as np

import open3d as o3d
from open3d import geometry

from seg3d.datasets.transforms.instance_augmentation import InstanceAugmentation


def load_points(lidar_file):
    points = np.load(lidar_file)[:, :6]
    return points


def load_label(label_file):
    labels = np.load(label_file)[:, 1]
    labels -= 1
    labels[labels == -1] = 255
    return labels


if __name__ == '__main__':
    data_dir = '/nfs/dataset-dtai-common/waymo_open_dataset_v_1_3_2/training'
    instance_aug = InstanceAugmentation(
        instance_path=os.path.join(data_dir, 'instances/lidar_instances_with_height.pkl'))

    label_files = glob.glob(os.path.join(data_dir, 'label/*.npy'))
    for label_file in tqdm(label_files):
        lidar_file = label_file.replace('label', 'lidar')
        points = load_points(lidar_file)
        labels = load_label(label_file)
        points, labels = instance_aug(points, None, labels)

        pcd = geometry.PointCloud()
        points = points.copy()
        pcd.points = o3d.utility.Vector3dVector(points[:, :3])

        points_colors = np.zeros((labels.shape[0], 3), dtype=np.float32)
        points_colors[labels == 255] = np.array([1.0, 0, 0])
        pcd.colors = o3d.utility.Vector3dVector(points_colors)

        filename = os.path.splitext(os.path.basename(lidar_file))[0]
        output_file = os.path.join(data_dir, 'visualization', filename + '.pcd')
        o3d.io.write_point_cloud(output_file, pcd)
