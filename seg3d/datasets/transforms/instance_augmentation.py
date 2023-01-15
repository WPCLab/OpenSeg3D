import os
import pickle
import numpy as np


class InstanceAugmentation(object):
    def __init__(self, instance_path, instance_label_ids=[3, 4, 10], ground_label_ids=[17, 18, 19, 20, 21],
                 add_count=5, random_rotate=True, local_transformation=True, random_flip=True):
        self.instance_label_ids = instance_label_ids
        self.ground_label_ids = ground_label_ids

        self.ground_label_map = {}
        for i, ground_label_id in enumerate(self.ground_label_ids):
            self.ground_label_map[ground_label_id] = i

        self.add_count = add_count
        self.random_rotate = random_rotate
        self.local_transformation = local_transformation
        self.random_flip = random_flip

        if os.path.exists(instance_path):
            with open(instance_path, 'rb') as f:
                self.instances = pickle.load(f)

    def __call__(self, points, point_image_features, labels):
        label_choice = np.random.choice(self.instance_label_ids, self.add_count, replace=True)
        uni_label, uni_count = np.unique(label_choice, return_counts=True)
        for label_id, count in zip(uni_label, uni_count):
            # find random instance
            instance_choice = np.random.choice(len(self.instances[label_id]), count)
            # add to current scan
            for idx in instance_choice:
                object_points = []
                ground_points = []
                for i in range(labels.shape[0]):
                    point = points[i, :3]
                    label = labels[i]
                    if label == 255:
                        continue
                    if label in self.ground_label_map:
                        ground_points.append(point)
                    else:
                        object_points.append(point)
                object_points = np.stack(object_points)
                ground_points = np.stack(ground_points)

                instance = self.instances[label_id][idx]
                instance_points = instance['cluster_points']
                instance_height = instance['cluster_height']
                instance_xyz = instance_points[:, :3]
                instance_feat = instance_points[:, 3:]
                instance_feat[:, 0] = 0
                instance_feat[:, 1] = np.tanh(instance_feat[:, 1])

                # random translation and rotation
                center = np.mean(instance_xyz, axis=0)
                if self.local_transformation:
                    instance_xyz = self.local_transform(instance_xyz, center)

                # random flip instance based on it center
                if self.random_flip:
                    # get axis
                    long_axis = [center[0], center[1]] / (center[0] ** 2 + center[1] ** 2) ** 0.5
                    short_axis = [-long_axis[1], long_axis[0]]
                    # random flip
                    flip_type = np.random.choice(5, 1)
                    if flip_type == 3:
                        instance_xyz[:, :2] = self.instance_flip(instance_xyz[:, :2], [long_axis, short_axis],
                                                                 [center[0], center[1]], flip_type)

                # need to check occlusion
                fail_flag = True
                center = np.mean(instance_xyz, axis=0)
                radius = self.get_instance_radius(instance_xyz, center)
                if self.random_rotate:
                    # random rotate
                    random_angle = np.random.random(20) * np.pi * 2
                    for r in random_angle:
                        center_r = self.rotate_origin(center[np.newaxis, ...], r)
                        # check if occluded and on ground
                        if self.check_valid(object_points, ground_points, center_r[0], min_dist=radius):
                            fail_flag = False
                            break
                    # rotate to empty space
                    if fail_flag: continue
                    instance_xyz = self.rotate_origin(instance_xyz, r)
                else:
                    # check if occluded and on ground
                    fail_flag = not self.check_valid(object_points, ground_points, center_r[0], min_dist=radius)

                if fail_flag: continue

                # adjust with saved height on ground
                instance_center = np.mean(instance_xyz, axis=0)
                instance_xyz = self.adjust_z_with_height(ground_points, instance_xyz, instance_center, instance_height)

                add_points = np.concatenate((instance_xyz, instance_feat), axis=1)
                points = np.concatenate((points, add_points), axis=0)
                add_labels = np.ones((add_points.shape[0],), dtype=labels.dtype) * label_id
                labels = np.concatenate((labels, add_labels), axis=0)
                if point_image_features is not None:
                    add_point_image_features = np.zeros((add_points.shape[0], point_image_features.shape[1]),
                                                        dtype=point_image_features.dtype)
                    point_image_features = np.concatenate((point_image_features, add_point_image_features), axis=0)

        if point_image_features is not None:
            return points, point_image_features, labels
        else:
            return points, labels

    def instance_flip(self, points, axis, center, flip_type=1):
        points = points[:] - center
        if flip_type == 1:
            # rotate 180 degree
            points = -points + center
        elif flip_type == 2:
            # flip over long axis
            a = axis[0][0]
            b = axis[0][1]
            flip_matrix = np.array([[b ** 2 - a ** 2, -2 * a * b], [-2 * a * b, a ** 2 - b ** 2]])
            points = np.matmul(flip_matrix, np.transpose(points, (1, 0)))
            points = np.transpose(points, (1, 0)) + center
        elif flip_type == 3:
            # flip over short axis
            a = axis[1][0]
            b = axis[1][1]
            flip_matrix = np.array([[b ** 2 - a ** 2, -2 * a * b], [-2 * a * b, a ** 2 - b ** 2]])
            points = np.matmul(flip_matrix, np.transpose(points, (1, 0)))
            points = np.transpose(points, (1, 0)) + center

        return points

    def check_valid(self, points_xyz_object, points_xyz_ground, center, min_dist=2):
        """check if close to a point and on ground"""
        # check no occlusion
        if points_xyz_object.ndim == 1:
            dist = np.linalg.norm(points_xyz_object[np.newaxis, :] - center, axis=1)
        else:
            dist = np.linalg.norm(points_xyz_object - center, axis=1)
        no_occlusion = np.all(dist > min_dist)

        # check on ground
        if points_xyz_ground.ndim == 1:
            dist = np.linalg.norm(points_xyz_ground[np.newaxis, :] - center, axis=1)
        else:
            dist = np.linalg.norm(points_xyz_ground - center, axis=1)
        on_ground = np.any(dist < 1.2 * min_dist)

        return no_occlusion and on_ground

    def rotate_origin(self, points_xyz, radians):
        """rotate a point around the origin"""
        x = points_xyz[:, 0]
        y = points_xyz[:, 1]
        new_xyz = points_xyz.copy()
        new_xyz[:, 0] = x * np.cos(radians) + y * np.sin(radians)
        new_xyz[:, 1] = -x * np.sin(radians) + y * np.cos(radians)
        return new_xyz

    def local_transform(self, xyz, center):
        """translate and rotate point cloud according to its center"""
        # random xyz
        loc_noise = np.random.normal(scale=0.25, size=(1, 3))
        # random angle
        rot_noise = np.random.uniform(-np.pi / 20, np.pi / 20)

        xyz = xyz - center
        xyz = self.rotate_origin(xyz, rot_noise)
        xyz = xyz + loc_noise

        return xyz + center

    def get_instance_radius(self, points_xyz, center):
        """compute radius of instance points"""
        if points_xyz.ndim == 1:
            dist = np.linalg.norm(points_xyz[np.newaxis, :] - center, axis=1)
        else:
            dist = np.linalg.norm(points_xyz - center, axis=1)
        radius = np.max(dist)
        return radius

    def adjust_z_with_height(self, ground_xyz, points_xyz, center, height):
        dist = np.linalg.norm(ground_xyz - center, axis=1)
        min_idx = np.argmin(dist)
        ground_z = ground_xyz[min_idx][2]
        est_z = ground_z + height
        points_xyz[:, 2] += (est_z - center[2])
        return points_xyz
