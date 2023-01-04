import os
import pickle
import numpy as np


class InstanceAugmentation(object):
    def __init__(self, instance_path, label_ids=[3, 4, 10], road_label_id=17, add_count=5, random_rotate=True):
        self.label_ids = label_ids
        self.road_label_id = road_label_id
        self.add_count = add_count

        self.random_rotate = random_rotate

        if os.path.exists(instance_path):
            with open(instance_path, 'rb') as f:
                self.instances = pickle.load(f)

    def __call__(self, points, point_image_features, labels):
        object_points = points[labels != self.road_label_id][:, :3]
        label_choice = np.random.choice(self.label_ids, self.add_count, replace=True)
        uni_label, uni_count = np.unique(label_choice, return_counts=True)
        for label_id, count in zip(uni_label, uni_count):
            # find random instance
            instance_choice = np.random.choice(len(self.instances[label_id]), count)
            # add to current scan
            for idx in instance_choice:
                instance_points = self.instances[label_id][idx].copy()
                instance_xyz = instance_points[:, :3]
                center = np.mean(instance_xyz, axis=0)

                # need to check occlusion
                fail_flag = True
                if self.random_rotate:
                    # random rotate
                    random_angle = np.random.random(20) * np.pi * 2
                    for r in random_angle:
                        center_r = self.rotate_origin(center[np.newaxis, ...], r)
                        # check if occluded
                        if self.check_occlusion(object_points, center_r[0]):
                            fail_flag = False
                            break
                    # rotate to empty space
                    if fail_flag: continue
                    instance_xyz = self.rotate_origin(instance_xyz, r)
                else:
                    fail_flag = not self.check_occlusion(object_points, center)
                if fail_flag: continue

                add_points = np.concatenate((instance_xyz, instance_points[:, 3:]), axis=1)
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

    def check_occlusion(self, points_xyz, center, min_dist=2):
        """check if close to a point"""
        if points_xyz.ndim == 1:
            dist = np.linalg.norm(points_xyz[np.newaxis, :] - center, axis=1)
        else:
            dist = np.linalg.norm(points_xyz - center, axis=1)
        return np.all(dist > min_dist)

    def rotate_origin(self, points_xyz, radians):
        """rotate a point around the origin"""
        x = points_xyz[:, 0]
        y = points_xyz[:, 1]
        new_xyz = points_xyz.copy()
        new_xyz[:, 0] = x * np.cos(radians) + y * np.sin(radians)
        new_xyz[:, 1] = -x * np.sin(radians) + y * np.cos(radians)
        return new_xyz
