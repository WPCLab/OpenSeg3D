import os
import glob
from collections import defaultdict

import numpy as np
from torch.utils.data import Dataset

from seg3d.core import VoxelGenerator
from seg3d.datasets.transforms import transforms
from seg3d.datasets.transforms.instance_augmentation import InstanceAugmentation
from seg3d.datasets.transforms.polarmix import PolarMix
from seg3d.utils.pointops_utils import cart2polar


class WaymoDataset(Dataset):
    def __init__(self, cfg, data_root, mode='training'):
        assert mode in ['training', 'validation', 'testing']
        self.cfg = cfg
        self.data_root = data_root
        self.mode = mode

        all_filenames = self.get_filenames('lidar')
        self.file_idx_to_name = self.build_file_idx_to_name(all_filenames)

        if self.mode == 'testing':
            self.filenames = self.get_testing_filenames(all_filenames)
        else:
            self.filenames = self.get_filenames('label')

        self.voxel_generator = VoxelGenerator(voxel_size=cfg.DATASET.VOXEL_SIZE,
                                              point_cloud_range=cfg.DATASET.POINT_CLOUD_RANGE)

        self.grid_size = self.voxel_generator.grid_size
        self.voxel_size = self.voxel_generator.voxel_size
        self.point_cloud_range = self.voxel_generator.point_cloud_range

        self.polar_mix = PolarMix(instance_classes=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
                                  rot_angle_range=[np.random.random() * np.pi * 2 / 3,
                                                   (np.random.random() + 1) * np.pi * 2 / 3])

        self.instance_aug = InstanceAugmentation(
            instance_path=os.path.join(self.data_root, 'instances/lidar_instances_with_height.pkl'))

        self.transforms = transforms.Compose([transforms.RandomGlobalRotation(cfg.DATASET.AUG_ROT_RANGE),
                                              transforms.RandomGlobalScaling(cfg.DATASET.AUG_SCALE_RANGE),
                                              transforms.RandomGlobalTranslation(cfg.DATASET.AUG_TRANSLATE_STD),
                                              transforms.RandomFlip(),
                                              transforms.PointShuffle(),
                                              transforms.PointSample(cfg.DATASET.AUG_SAMPLE_RATIO,
                                                                     cfg.DATASET.AUG_SAMPLE_RANGE)])

    @property
    def dim_point(self):
        return self.cfg.DATASET.DIM_POINT

    @property
    def use_multi_sweeps(self):
        return self.cfg.DATASET.USE_MULTI_SWEEPS

    @property
    def use_cylinder(self):
        return self.cfg.DATASET.USE_CYLINDER

    @property
    def num_classes(self):
        return self.cfg.DATASET.NUM_CLASSES

    @property
    def class_names(self):
        class_names = self.cfg.DATASET.CLASS_NAMES
        return class_names

    @property
    def class_weight(self):
        return self.cfg.DATASET.CLASS_WEIGHT

    @property
    def palette(self):
        return self.cfg.DATASET.PALETTE

    @property
    def use_image_feature(self):
        return self.cfg.DATASET.USE_IMAGE_FEATURE

    @property
    def dim_image_feature(self):
        return self.cfg.DATASET.DIM_IMAGE_FEATURE

    @property
    def ignore_index(self):
        return self.cfg.DATASET.IGNORE_INDEX

    @staticmethod
    def parse_filename(filename):
        splits = filename.split('-')
        file_idx = splits[0]
        timestamp = np.int64(splits[1])
        frame_idx = int(splits[2])
        return file_idx, frame_idx, timestamp

    def get_filenames(self, dir_name):
        return [os.path.splitext(os.path.basename(path))[0] for path in
                glob.glob(os.path.join(self.data_root, dir_name, '*.npy'))]

    def get_testing_filenames(self, filenames):
        testing_frames = dict()
        with open(os.path.join(self.data_root, '3d_semseg_test_set_frames.txt'), 'r') as fp:
            lines = fp.read().splitlines()
            for line in lines:
                splits = line.split(',')
                file_idx = splits[0]
                timestamp = np.int64(splits[1])
                testing_frames[(file_idx, timestamp)] = True

        testing_filenames = []
        for filename in filenames:
            file_idx, frame_idx, timestamp = self.parse_filename(filename)
            if (file_idx, timestamp) in testing_frames:
                testing_filenames.append(filename)
        return testing_filenames

    def build_file_idx_to_name(self, filenames):
        file_idx_to_name = dict()
        for filename in filenames:
            file_idx, frame_idx, timestamp = self.parse_filename(filename)
            file_idx_to_name[(file_idx, frame_idx)] = filename
        return file_idx_to_name

    def get_lidar_path(self, filename):
        lidar_file = os.path.join(self.data_root, 'lidar', filename + '.npy')
        return lidar_file

    def get_image_feature_path(self, filename):
        image_feature_file = os.path.join(self.data_root, 'image_feature', filename + '.npy')
        return image_feature_file

    def get_pose_path(self, filename):
        pose_file = os.path.join(self.data_root, 'pose', filename + '.txt')
        return pose_file

    def get_label_path(self, filename):
        label_file = os.path.join(self.data_root, 'label', filename + '.npy')
        return label_file

    def load_pose(self, filename):
        pose_file = self.get_pose_path(filename)
        sensor2local_matrix = np.loadtxt(pose_file)
        return sensor2local_matrix

    def load_image_features(self, num_points, filename):
        # load image feature
        image_feature_file = self.get_image_feature_path(filename)
        image_feature = np.load(image_feature_file, allow_pickle=True).item()

        # assemble point image features
        point_image_features = np.zeros((num_points, self.dim_image_feature), dtype=np.float32)
        for k in image_feature:
            point_image_features[k] = image_feature[k]
        return point_image_features

    def load_points(self, filename):
        lidar_file = self.get_lidar_path(filename)
        # (x, y, z, range, intensity, elongation, 6-dim camera project, range col, row and index): [N, 15]
        lidar_points = np.load(lidar_file)

        # set range value to be zero
        lidar_points[:, 3] = 0
        # normalize intensity
        lidar_points[:, 4] = np.tanh(lidar_points[:, 4])
        return lidar_points

    def load_points_from_sweeps(self, filename, num_sweeps=3, max_num_sweeps=5, pad_empty_sweeps=False):
        # current frame
        file_idx, frame_idx, timestamp = self.parse_filename(filename)
        points = self.load_points(filename)
        points[:, 3] = 0
        cur_point_indices = np.arange(points.shape[0])
        ts = timestamp / 1e6
        transform_matrix = self.load_pose(filename)

        # history sweep filenames
        history_sweep_filenames = []
        for i in range(0, max_num_sweeps - 1):
            sweep_frame_idx = frame_idx - i - 1
            if sweep_frame_idx >= 0:
                sweep_filename = self.file_idx_to_name[(file_idx, sweep_frame_idx)]
                history_sweep_filenames.append(sweep_filename)

        history_num_sweeps = num_sweeps - 1
        sweep_points_list = [points]
        if pad_empty_sweeps and len(history_sweep_filenames) == 0:
            for i in range(history_num_sweeps):
                sweep_points_list.append(points)
        else:
            if len(history_sweep_filenames) <= history_num_sweeps:
                choices = np.arange(len(history_sweep_filenames))
            elif self.mode == 'training':
                choices = np.random.choice(
                    len(history_sweep_filenames), history_num_sweeps, replace=False)
            else:
                choices = np.arange(history_num_sweeps)

            for idx in choices:
                sweep_filename = history_sweep_filenames[idx]
                points_sweep = self.load_points(sweep_filename)
                timestamp = self.parse_filename(sweep_filename)[-1]
                sweep_ts = timestamp / 1e6
                sweep_transform_matrix = self.load_pose(sweep_filename)
                sensor2lidar = np.linalg.inv(transform_matrix) @ sweep_transform_matrix
                sensor2lidar_rotation = sensor2lidar[0:3, 0:3]
                sensor2lidar_translation = sensor2lidar[0:3, 3]
                points_sweep[:, :3] = points_sweep[:, :3] @ sensor2lidar_rotation.T
                points_sweep[:, :3] += sensor2lidar_translation
                points_sweep[:, 3] = ts - sweep_ts
                sweep_points_list.append(points_sweep)

        points = np.concatenate(sweep_points_list, axis=0)
        return points, cur_point_indices

    def load_label(self, filename):
        label_file = self.get_label_path(filename)
        semantic_labels = np.load(label_file)[:, 1]  # (N, 1)

        # convert unlabeled to ignored label (0 to 255)
        semantic_labels -= 1
        semantic_labels[semantic_labels == -1] = 255
        return semantic_labels

    def prepare_voxel_labels(self, data_dict):
        assert self.ignore_index == 255

        point_voxel_ids = data_dict.get('point_voxel_ids', None)
        cur_point_indices = data_dict.get('cur_point_indices', None)
        if cur_point_indices is not None:
            cur_point_voxel_ids = point_voxel_ids[cur_point_indices]
        else:
            cur_point_voxel_ids = point_voxel_ids
        point_labels = data_dict.get('point_labels', None)
        voxel_coords = data_dict.get('voxel_coords', None)
        assert point_voxel_ids is not None and point_labels is not None and voxel_coords is not None

        label_size = 256
        voxel_label_counter = dict()
        for i in range(cur_point_voxel_ids.shape[0]):
            voxel_id = cur_point_voxel_ids[i]
            label = point_labels[i]
            if voxel_id != -1:
                if voxel_id not in voxel_label_counter:
                    counter = np.zeros((label_size,), dtype=np.uint16)
                    counter[label] += 1
                    voxel_label_counter[voxel_id] = counter
                else:
                    counter = voxel_label_counter[voxel_id]
                    counter[label] += 1
                    voxel_label_counter[voxel_id] = counter

        voxel_labels = np.ones(voxel_coords.shape[0], dtype=np.uint8) * self.ignore_index
        for voxel_id in voxel_label_counter:
            counter = voxel_label_counter[voxel_id]
            voxel_labels[voxel_id] = np.argmax(counter)

        data_dict['voxel_labels'] = voxel_labels

    def prepare_data(self, data_dict):
        """
        Args:
            data_dict:
                points: optional, (N, ndim)
                labels: optional, (N)
        Returns:
            data_dict:
                points: (N, ndim)
                point_labels: optional, (N)
                voxel_coords: optional, (num_voxels, 3)
                point_voxel_ids: optional, (N)
                voxel_labels: optional, (num_voxels)
        """
        if self.mode == 'training' and self.cfg.DATASET.AUG_DATA:
            data_dict = self.transforms(data_dict)

        if self.cfg.DATASET.USE_MULTI_SWEEPS:
            data_dict['cur_point_count'] = data_dict['cur_point_indices'].shape[0]
        else:
            data_dict['cur_point_count'] = data_dict['points'].shape[0]

        if self.cfg.DATASET.USE_CYLINDER:
            points = data_dict['points']
            polar_points = cart2polar(points)
            data_dict['points'] = np.concatenate((polar_points, points[:, :2], points[:, 3:]), axis=1)

        voxel_coords, point_voxel_ids = self.voxel_generator.generate(data_dict['points'])
        data_dict['voxel_coords'] = voxel_coords
        data_dict['point_voxel_ids'] = point_voxel_ids

        return data_dict

    def __getitem__(self, index):
        filename = self.filenames[index]

        input_dict = {
            'filename': filename
        }

        if self.cfg.DATASET.USE_MULTI_SWEEPS:
            points, cur_point_indices = self.load_points_from_sweeps(filename, self.cfg.DATASET.NUM_SWEEPS,
                                                                     self.cfg.DATASET.MAX_NUM_SWEEPS)
            input_dict['cur_point_indices'] = cur_point_indices
        else:
            points = self.load_points(filename)

        input_dict['points'] = points[:, :self.dim_point]

        if self.cfg.DATASET.USE_IMAGE_FEATURE:
            if self.cfg.DATASET.USE_MULTI_SWEEPS:
                input_dict['point_image_features'] = self.load_image_features(input_dict['cur_point_indices'].shape[0],
                                                                              filename)
            else:
                input_dict['point_image_features'] = self.load_image_features(input_dict['points'].shape[0], filename)

        if self.mode != 'testing':
            input_dict['point_labels'] = self.load_label(filename)

        if self.mode == 'training' and self.cfg.DATASET.AUG_DATA and not self.cfg.DATASET.USE_MULTI_SWEEPS:
            filename2 = self.filenames[np.random.randint(len(self.filenames))]
            points2 = self.load_points(filename2)[:, :self.dim_point]
            labels2 = self.load_label(filename2)
            if self.cfg.DATASET.USE_IMAGE_FEATURE:
                point_images_features2 = self.load_image_features(points2.shape[0], filename2)
                input_dict['points'], input_dict['point_image_features'], input_dict['point_labels'] = \
                    self.polar_mix(input_dict['points'], input_dict['point_image_features'], input_dict['point_labels'],
                                   points2, point_images_features2, labels2)
                input_dict['points'], input_dict['point_image_features'], input_dict['point_labels'] = \
                    self.instance_aug(input_dict['points'], input_dict['point_image_features'],
                                      input_dict['point_labels'])
            else:
                input_dict['points'], input_dict['point_labels'] = \
                    self.polar_mix(input_dict['points'], None, input_dict['point_labels'], points2, None, labels2)
                input_dict['points'], input_dict['point_labels'] = \
                    self.instance_aug(input_dict['points'], None, input_dict['point_labels'])

        if self.mode == 'testing':
            if self.cfg.DATASET.USE_MULTI_SWEEPS:
                input_dict['points_ri'] = points[input_dict['cur_point_indices']][:, -3:].astype(np.int32)
            else:
                input_dict['points_ri'] = points[:, -3:].astype(np.int32)

        data_dict = self.prepare_data(data_dict=input_dict)

        if self.mode != 'testing':
            self.prepare_voxel_labels(data_dict)

        return data_dict

    @staticmethod
    def collate_batch(batch_list, _unused=False):
        data_dict = defaultdict(list)
        for cur_sample in batch_list:
            for key, val in cur_sample.items():
                data_dict[key].append(val)

        ret = {}
        for key, val in data_dict.items():
            if key in ['points', 'voxel_coords']:
                coors = []
                for i, coor in enumerate(val):
                    coor_pad = np.pad(coor, ((0, 0), (1, 0)), mode='constant', constant_values=i)
                    coors.append(coor_pad)
                ret[key] = np.concatenate(coors, axis=0)
            elif key in ['points_ri', 'point_image_features', 'point_labels', 'voxel_labels']:
                ret[key] = np.concatenate(val, axis=0)
            elif key in ['filename']:
                ret[key] = val

        voxel_id_offset, count = [], 0
        point_voxel_ids_list = data_dict['point_voxel_ids']
        for i, point_voxel_ids in enumerate(point_voxel_ids_list):
            point_voxel_ids[point_voxel_ids != -1] += count
            count += data_dict['voxel_coords'][i].shape[0]
            voxel_id_offset.append(count)
        ret['point_voxel_ids'] = np.concatenate(point_voxel_ids_list, axis=0)
        ret['voxel_id_offset'] = np.array(voxel_id_offset)

        point_id_offset, count = [], 0
        cur_point_count_list = data_dict['cur_point_count']
        for cur_point_count in cur_point_count_list:
            count += cur_point_count
            point_id_offset.append(count)
        ret['point_id_offset'] = np.array(point_id_offset)

        batch_size = len(batch_list)
        ret['batch_size'] = batch_size
        return ret

    def __len__(self):
        return len(self.filenames)


if __name__ == '__main__':
    from seg3d.utils.config import cfg
    from seg3d.utils.visualize import draw_points, draw_voxels

    cfg.DATASET.PALETTE = [[0, 0, 142], [0, 0, 70], [0, 60, 100], [61, 133, 198], [180, 0, 0], [255, 0, 0],
                           [220, 20, 60], [246, 178, 107], [250, 170, 30], [153, 153, 153], [230, 145, 56],
                           [119, 11, 32], [0, 0, 230], [70, 70, 70], [107, 142, 35], [190, 153, 153], [196, 196, 196],
                           [128, 64, 128], [234, 209, 220], [217, 210, 233], [81, 0, 81], [244, 35, 232]]

    data_dir = '/nfs/dataset-dtai-common/waymo_open_dataset_v_1_3_2/validation'
    dataset = WaymoDataset(cfg, data_dir, mode='validation')
    for step, sample in enumerate(dataset):
        print(step, sample['points'].shape, sample['point_labels'].shape)

        if cfg.DATASET.VISUALIZE:
            draw_points(dataset.palette, sample, os.path.join(data_dir, 'visualization/points'))
            draw_voxels(dataset.palette, dataset.voxel_size, dataset.point_cloud_range, sample,
                        os.path.join(data_dir, 'visualization/voxels'))
