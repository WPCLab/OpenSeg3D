import os
import glob

from tqdm import tqdm
import numpy as np

from seg3d.datasets.transforms.instance_augmentation import InstanceAugmentation

if __name__ == '__main__':
    data_dir = '/nfs/dataset-dtai-common/waymo_open_dataset_v_1_3_2'
    split = 'training'
    instance_aug = InstanceAugmentation(instance_path=os.path.join(data_dir, split, 'instances/lidar_instances.pkl'))

    label_files = glob.glob(os.path.join(data_dir, split, 'label/*.npy'))
    for label_file in tqdm(label_files):
        lidar_file = label_file.replace('label', 'lidar')
        lidar = np.load(lidar_file)[:, :6]
        label = np.load(label_file)[:, 1]
        points, labels = instance_aug(lidar, label)