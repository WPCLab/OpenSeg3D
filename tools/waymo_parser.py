import argparse
import multiprocessing
import os

import numpy as np
import cv2

import tensorflow as tf
from tqdm import tqdm

from waymo_open_dataset.utils import frame_utils
from waymo_open_dataset import dataset_pb2 as open_dataset

TOP_LIDAR_ROW_NUM = 64
TOP_LIDAR_COL_NUM = 2650

class WaymoParser(object):
    def __init__(self,
                 tfrecord_list_file,
                 save_dir,
                 num_workers,
                 test_mode=False):
        self.tfrecord_list_file = tfrecord_list_file
        self.save_dir = save_dir
        self.num_workers = num_workers
        self.test_mode = test_mode

        with open(self.tfrecord_list_file, 'r') as fp:
            self.tfrecord_pathnames = fp.read().splitlines()

        self.label_save_dir = f'{self.save_dir}/label'
        self.image_save_dir = f'{self.save_dir}/image'
        self.calib_save_dir = f'{self.save_dir}/calib'
        self.point_cloud_save_dir = f'{self.save_dir}/lidar'
        self.pose_save_dir = f'{self.save_dir}/pose'

        self.create_folder()

    @staticmethod
    def get_file_id(frame):
        context_name = frame.context.name
        timestamp = frame.timestamp_micros
        file_id = context_name + '-' + str(timestamp) + '-'
        return file_id

    def parse(self):
        print('======Parse Started!======')
        pool = multiprocessing.Pool(self.num_workers)
        gen = list(tqdm(pool.imap(self.parse_one, range(len(self))), total=len(self)))
        pool.close()
        pool.join()
        print('======Parse Finished!======')

    def parse_one(self, index):
        """Convert action for single file.
        Args:
            index (int): Index of the file to be converted.
        """
        pathname = self.tfrecord_pathnames[index]

        try:
            dataset = tf.data.TFRecordDataset(pathname, compression_type='')
            for frame_idx, data in enumerate(dataset):
                frame = open_dataset.Frame()
                frame.ParseFromString(bytearray(data.numpy()))
                file_id = self.get_file_id(frame)

                self.save_image(frame, file_id, frame_idx)
                self.save_calib(frame, file_id, frame_idx)
                self.save_lidar_label(frame, file_id, frame_idx)
                self.save_pose(frame, file_id, frame_idx)

        except Exception as e:
            print('Failed to parse: %s, error msg: %s' % (pathname, str(e)))

        return pathname

    def __len__(self):
        """Length of the filename list."""
        return len(self.tfrecord_pathnames)

    @staticmethod
    def convert_range_image_to_point_cloud_ri(frame,
                                              range_images,
                                              ri_index=0):
        calibrations = sorted(frame.context.laser_calibrations, key=lambda c: c.name)
        points_ri = []
        for c in calibrations:
            range_image = range_images[c.name][ri_index]
            range_image_tensor = tf.reshape(
                tf.convert_to_tensor(range_image.data), range_image.shape.dims)
            range_image_mask = range_image_tensor[..., 0] > 0

            if c.name == open_dataset.LaserName.TOP:
                xgrid, ygrid = np.meshgrid(range(TOP_LIDAR_COL_NUM), range(TOP_LIDAR_ROW_NUM))
                col_row_inds_top = np.stack([xgrid, ygrid], axis=-1)
                sl_points_ri = col_row_inds_top[np.where(range_image_mask)]
                sl_points_ri = np.column_stack((sl_points_ri, ri_index * np.ones((sl_points_ri.shape[0], 1))))
            else:
                num_valid_point = tf.math.reduce_sum(tf.cast(range_image_mask, tf.int32))
                sl_points_ri = tf.ones([num_valid_point, 3], dtype=tf.int32) * -1

            points_ri.append(sl_points_ri)
        return points_ri

    @staticmethod
    def convert_range_image_to_point_cloud_labels(frame,
                                                  range_images,
                                                  segmentation_labels,
                                                  ri_index=0):
        """Convert segmentation labels from range images to point clouds.

        Args:
          frame: open dataset frame
          range_images: A dict of {laser_name, [range_image_first_return,
             range_image_second_return]}.
          segmentation_labels: A dict of {laser_name, [range_image_first_return,
             range_image_second_return]}.
          ri_index: 0 for the first return, 1 for the second return.

        Returns:
          point_labels: {[N, 2]} list of 3d lidar points's segmentation labels. 0 for
            points that are not labeled.
        """
        calibrations = sorted(frame.context.laser_calibrations, key=lambda c: c.name)
        point_labels = []
        for c in calibrations:
            range_image = range_images[c.name][ri_index]
            range_image_tensor = tf.reshape(
                tf.convert_to_tensor(range_image.data), range_image.shape.dims)
            range_image_mask = range_image_tensor[..., 0] > 0

            if c.name in segmentation_labels:
                sl = segmentation_labels[c.name][ri_index]
                sl_tensor = tf.reshape(tf.convert_to_tensor(sl.data), sl.shape.dims)
                sl_points_tensor = tf.gather_nd(sl_tensor, tf.where(range_image_mask))
            else:
                num_valid_point = tf.math.reduce_sum(tf.cast(range_image_mask, tf.int32))
                sl_points_tensor = tf.zeros([num_valid_point, 2], dtype=tf.int32)

            point_labels.append(sl_points_tensor.numpy())
        return point_labels

    def save_image(self, frame, file_id, frame_idx):
        """Parse and save the images in png format.
        Args:
            frame (:obj:`Frame`): Open dataset frame proto.
            file_id (str): Current file id.
            frame_idx (int): Current frame index.
        """
        for img in frame.images:
            img_path = f'{self.image_save_dir}/{str(img.name - 1)}/{file_id}' + \
                       f'{str(frame_idx).zfill(3)}.png'
            img = tf.image.decode_jpeg(img.image)
            cv2.imwrite(img_path, img.numpy())

    def save_calib(self, frame, file_id, frame_idx):
        """Parse and save the calibration data.
        Args:
            frame (:obj:`Frame`): Open dataset frame proto.
            file_id (str): Current file id.
            frame_idx (int): Current frame index.
        """
        # waymo front camera to kitti reference camera
        T_front_cam_to_ref = np.array([[0.0, -1.0, 0.0], [0.0, 0.0, -1.0],
                                       [1.0, 0.0, 0.0]])
        camera_calibs = []
        R0_rect = [f'{i:e}' for i in np.eye(3).flatten()]
        Tr_velo_to_cams = []
        calib_context = ''

        for camera in frame.context.camera_calibrations:
            # extrinsic parameters
            T_cam_to_vehicle = np.array(camera.extrinsic.transform).reshape(
                4, 4)
            T_vehicle_to_cam = np.linalg.inv(T_cam_to_vehicle)
            Tr_velo_to_cam = \
                self.cart_to_homo(T_front_cam_to_ref) @ T_vehicle_to_cam
            if camera.name == 1:  # FRONT = 1, see dataset.proto for details
                self.T_velo_to_front_cam = Tr_velo_to_cam.copy()
            Tr_velo_to_cam = Tr_velo_to_cam[:3, :].reshape((12,))
            Tr_velo_to_cams.append([f'{i:e}' for i in Tr_velo_to_cam])

            # intrinsic parameters
            camera_calib = np.zeros((3, 4))
            camera_calib[0, 0] = camera.intrinsic[0]
            camera_calib[1, 1] = camera.intrinsic[1]
            camera_calib[0, 2] = camera.intrinsic[2]
            camera_calib[1, 2] = camera.intrinsic[3]
            camera_calib[2, 2] = 1
            camera_calib = list(camera_calib.reshape(12))
            camera_calib = [f'{i:e}' for i in camera_calib]
            camera_calibs.append(camera_calib)

        # all camera ids are saved as id-1 in the result because
        # camera 0 is unknown in the proto
        for i in range(5):
            calib_context += 'P' + str(i) + ': ' + \
                             ' '.join(camera_calibs[i]) + '\n'
        calib_context += 'R0_rect' + ': ' + ' '.join(R0_rect) + '\n'
        for i in range(5):
            calib_context += 'Tr_velo_to_cam_' + str(i) + ': ' + \
                             ' '.join(Tr_velo_to_cams[i]) + '\n'

        calib_path = f'{self.calib_save_dir}/{file_id}' + \
                     f'{str(frame_idx).zfill(3)}.txt'
        with open(calib_path, 'w+') as fp_calib:
            fp_calib.write(calib_context)
            fp_calib.close()

    def save_lidar_label(self, frame, file_id, frame_idx):
        """Parse and save the lidar data in psd format.
        Args:
            frame (:obj:`Frame`): Open dataset frame proto.
            file_id (str): Current file id.
            frame_idx (int): Current frame index.
        """
        range_images, camera_projections, segmentation_labels, range_image_top_pose = \
            frame_utils.parse_range_image_and_camera_projection(frame)

        # points of first return
        points_0, cp_points_0 = frame_utils.convert_range_image_to_point_cloud(
            frame, range_images, camera_projections, range_image_top_pose, ri_index=0, keep_polar_features=True)
        points_0, cp_points_0 = np.concatenate(points_0, axis=0), np.concatenate(cp_points_0, axis=0)

        # points of second return
        points_1, cp_points_1 = frame_utils.convert_range_image_to_point_cloud(
            frame, range_images, camera_projections, range_image_top_pose, ri_index=1, keep_polar_features=True)
        points_1, cp_points_1 = np.concatenate(points_1, axis=0), np.concatenate(cp_points_1, axis=0)

        # point cloud with 6-dim: [x, y, z, range, intensity, and elongation]
        point_cloud = np.concatenate([points_0, points_1], axis=0)
        point_cloud = point_cloud[:, [3, 4, 5, 0, 1, 2]]

        # point cloud with 12-dim: [x, y, z, range, intensity, elongation and camera projection]
        cp_cloud = np.concatenate([cp_points_0, cp_points_1], axis=0)
        point_cloud = np.concatenate([point_cloud, cp_cloud], axis=1)

        # point range index of first return
        points_ri_0 = self.convert_range_image_to_point_cloud_ri(
            frame, range_images, ri_index=0)
        points_ri_0 = np.concatenate(points_ri_0, axis=0)

        # point range index of second return
        points_ri_1 = self.convert_range_image_to_point_cloud_ri(
            frame, range_images, ri_index=1)
        points_ri_1 = np.concatenate(points_ri_1, axis=0)

        points_ri = np.concatenate([points_ri_0, points_ri_1], axis=0)
        point_cloud = np.concatenate([point_cloud, points_ri], axis=1)

        pc_path = f'{self.point_cloud_save_dir}/{file_id}' + \
                  f'{str(frame_idx).zfill(3)}'
        np.save(pc_path, point_cloud)

        # save label
        if not self.test_mode and len(segmentation_labels) > 0:
            # point labels of first return
            point_labels_0 = self.convert_range_image_to_point_cloud_labels(
                frame, range_images, segmentation_labels, ri_index=0)
            point_labels_0 = np.concatenate(point_labels_0, axis=0)

            # point labels of second return
            point_labels_1 = self.convert_range_image_to_point_cloud_labels(
                frame, range_images, segmentation_labels, ri_index=1)
            point_labels_1 = np.concatenate(point_labels_1, axis=0)
            point_labels = np.concatenate([point_labels_0, point_labels_1], axis=0)

            label_path = f'{self.label_save_dir}/{file_id}' + \
                         f'{str(frame_idx).zfill(3)}'
            np.save(label_path, point_labels)

    def save_pose(self, frame, file_id, frame_idx):
        """Parse and save the pose data.
        Note that SDC's own pose is not included in the regular training
        of KITTI dataset. KITTI raw dataset contains ego motion files
        but are not often used. Pose is important for algorithms that
        take advantage of the temporal information.
        Args:
            frame (:obj:`Frame`): Open dataset frame proto.
            file_id (str): Current file id.
            frame_idx (int): Current frame index.
        """
        pose_path = f'{self.pose_save_dir}/{file_id}' + \
                    f'{str(frame_idx).zfill(3)}.txt'
        pose = np.array(frame.pose.transform).reshape(4, 4)
        np.savetxt(pose_path, pose)

    def create_folder(self):
        """Create folder for data preprocessing."""
        if not self.test_mode:
            dir_list = [
                self.point_cloud_save_dir, self.label_save_dir,
                self.calib_save_dir, self.pose_save_dir,
                self.image_save_dir
            ]
        else:
            dir_list = [
                self.calib_save_dir, self.point_cloud_save_dir,
                self.pose_save_dir, self.image_save_dir
            ]
        for d in dir_list:
            if not os.path.exists(d):
                os.makedirs(d)
        for i in range(5):
            if not os.path.exists(f'{self.image_save_dir}/{str(i)}'):
                os.makedirs(f'{self.image_save_dir}/{str(i)}')

    def cart_to_homo(self, mat):
        """Convert transformation matrix in Cartesian coordinates to
        homogeneous format.
        Args:
            mat (np.ndarray): Transformation matrix in Cartesian.
                The input matrix shape is 3x3 or 3x4.
        Returns:
            np.ndarray: Transformation matrix in homogeneous format.
                The matrix shape is 4x4.
        """
        ret = np.eye(4)
        if mat.shape == (3, 3):
            ret[:3, :3] = mat
        elif mat.shape == (3, 4):
            ret[:3, :] = mat
        else:
            raise ValueError(mat.shape)
        return ret

def parse_args():
    parser = argparse.ArgumentParser(description='Train a 3d segmentor')
    parser.add_argument(
        '--tfrecord_list_file',
        type=str,
        help='the file with tfrecord file list'
    )

    parser.add_argument(
        '--save_dir',
        type=str,
        help='directory for saving output file'
    )

    parser.add_argument(
        '--test_mode',
        action='store_true'
    )

    parser.add_argument(
        '--num_workers',
        type=int,
        default=4
    )

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    parser = WaymoParser(args.tfrecord_list_file, args.save_dir, args.num_workers, args.test_mode)
    parser.parse()
