import os
import glob
import pickle

import numpy as np
from tqdm import tqdm

from sklearn.cluster import DBSCAN

TARGET_LABEL_ID = 3 # other-vehicle:3, motorcyclist:4, cone: 10
TARGET_MIN_POINT_NUM = 500 # other-vehicle:500, motorcyclist:100, cone: 50


if __name__ == '__main__':
    data_dir = '/nfs/dataset-dtai-common/waymo_open_dataset_v_1_3_2/training'
    label_files = glob.glob(os.path.join(data_dir, 'label/*.npy'))
    lidar_instances = []
    for label_file in tqdm(label_files):
        lidar_file = label_file.replace('label', 'lidar')
        lidar = np.load(lidar_file)[:, :6]
        label = np.load(label_file)[:, 1]
        target_lidar = lidar[label == TARGET_LABEL_ID]

        if target_lidar.shape[0] < TARGET_MIN_POINT_NUM:
            continue

        clustering = DBSCAN(eps=0.2, min_samples=TARGET_MIN_POINT_NUM).fit(target_lidar[:, :3])
        cluster_ids = clustering.labels_

        # Number of clusters in labels, ignoring noise if present.
        unique_cluster_ids = set(cluster_ids)
        for cluster_id in unique_cluster_ids:
            if cluster_id == -1:
                continue

            lidar_cluster = target_lidar[cluster_ids == cluster_id]
            lidar_instances.append(lidar_cluster)

    fp = open('lidar_instances.pkl', 'wb')
    pickle.dump(lidar_instances, fp)
    fp.close()
