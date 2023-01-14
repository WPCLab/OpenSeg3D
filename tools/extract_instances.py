import os
import glob
import pickle

import numpy as np
from tqdm import tqdm

from sklearn.cluster import DBSCAN

TARGET_LABEL_ID = 3  # other-vehicle:3, motorcyclist:4, cone:10
TARGET_MIN_POINT_NUM = 120  # other-vehicle:120, motorcyclist:30, cone:30


def load_points(lidar_file):
    points = np.load(lidar_file)[:, :6]
    return points


def load_label(label_file):
    labels = np.load(label_file)[:, 1]
    labels -= 1
    labels[labels == -1] = 255
    return labels


def get_instance_radius(points_xyz, center):
    """compute radius of instance points"""
    if points_xyz.ndim == 1:
        dist = np.linalg.norm(points_xyz[np.newaxis, :] - center, axis=1)
    else:
        dist = np.linalg.norm(points_xyz - center, axis=1)
    radius = np.max(dist)
    return radius


if __name__ == '__main__':
    ground_label_dict = {17: 0, 18: 1, 19: 2, 20: 3, 21: 4}
    data_dir = '/nfs/dataset-dtai-common/waymo_open_dataset_v_1_3_2/training'
    label_files = glob.glob(os.path.join(data_dir, 'label/*.npy'))
    lidar_instances = []
    for label_file in tqdm(label_files):
        lidar_file = label_file.replace('label', 'lidar')
        points = load_points(lidar_file)
        labels = load_label(label_file)

        ground_points = []
        for i in range(points.shape[0]):
            if labels[i] in ground_label_dict:
                ground_points.append(points[i, :3])
        ground_points = np.stack(ground_points)

        target_points = points[labels == TARGET_LABEL_ID]
        if target_points.shape[0] < TARGET_MIN_POINT_NUM:
            continue

        clustering = DBSCAN(eps=0.25, min_samples=TARGET_MIN_POINT_NUM).fit(target_points[:, :2])
        cluster_ids = clustering.labels_

        # Number of clusters in labels, ignoring noise if present.
        unique_cluster_ids = set(cluster_ids)
        for cluster_id in unique_cluster_ids:
            if cluster_id == -1:
                continue

            cluster_points = target_points[cluster_ids == cluster_id]
            cluster_center = np.mean(cluster_points[:, :3], axis=0)
            cluster_radius = get_instance_radius(cluster_points[:, :3], cluster_center)

            distance = np.linalg.norm(ground_points - cluster_center, axis=1)
            indices = (distance < 1.2 * cluster_radius)
            ground_points_within_radius = ground_points[indices]
            if ground_points_within_radius.shape[0] > 0:
                distance_within_radius = distance[indices]
                min_idx = np.argmin(distance_within_radius)
                cluster_height = cluster_center[2] - ground_points_within_radius[min_idx][2]
                lidar_instances.append({'cluster_height': cluster_height, 'cluster_points': cluster_points})

    fp = open('lidar_instances.pkl', 'wb')
    pickle.dump(lidar_instances, fp)
    fp.close()
