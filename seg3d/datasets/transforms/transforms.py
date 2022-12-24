import numpy as np

from . import transform_utils


class Compose(object):
    """Composes several transforms together.
    Args:
        transforms (list of ``Transform`` objects): list of transforms to compose.
    Example:
        >>> transforms.Compose([
        >>>     transforms.RandomGlobalScaling([0.95, 1.05]),
        >>>     transforms.RandomGlobalRotation([-0.78539816, 0.78539816]),
        >>> ])
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, data_dict):
        for t in self.transforms:
            data_dict = t(data_dict)
        return data_dict

    def __repr__(self) -> str:
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += f"    {t}"
        format_string += "\n)"
        return format_string

class RandomDropPointsColor(object):
    r"""Randomly set the color of points to all zeros.

    Once this transform is executed, all the points' color will be dropped.
    Refer to `PAConv <https://github.com/CVMI-Lab/PAConv/blob/main/scene_seg/
    util/transform.py#L223>`_ for more details.

    Args:
        drop_ratio (float, optional): The probability of dropping point colors.
            Defaults to 0.2.
    """

    def __init__(self, drop_ratio=0.2):
        assert isinstance(drop_ratio, (int, float)) and 0 <= drop_ratio <= 1, \
            f'invalid drop_ratio value {drop_ratio}'
        self.drop_ratio = drop_ratio

    def __call__(self, data_dict):
        """Call function to drop point colors.

        Args:
            data_dict (dict): Result dict from loading pipeline.

        Returns:
            dict: Results after color dropping,
                'points' key is updated in the result dict.
        """
        point_image_features = data_dict.get('point_image_features', None)
        if point_image_features is not None:
            # this if-expression is a bit strange
            # `RandomDropPointsColor` is used in training 3D segmentor PAConv
            # we discovered in our experiments that, using
            # `if np.random.rand() > 1.0 - self.drop_ratio` consistently leads to
            # better results than using `if np.random.rand() < self.drop_ratio`
            # so we keep this hack in our codebase
            if np.random.rand() > 1.0 - self.drop_ratio:
                point_image_features = point_image_features * 0.0
                data_dict['point_image_features'] = point_image_features
        return data_dict

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += f'(drop_ratio={self.drop_ratio})'
        return repr_str

class RandomGlobalScaling(object):
    def __init__(self, scale_range) -> None:
        self.scale_range = scale_range

    def __call__(self, data_dict):
        if self.scale_range[1] - self.scale_range[0] < 1e-3:
            return data_dict
        noise_scale = np.random.uniform(self.scale_range[0], self.scale_range[1])
        data_dict['points'][:, :3] *= noise_scale
        return data_dict

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


class RandomGlobalRotation(object):
    def __init__(self, rot_range) -> None:
        self.rot_range = rot_range

    def __call__(self, data_dict):
        noise_rotation = np.random.uniform(self.rot_range[0], self.rot_range[1])
        points = data_dict['points']
        points = transform_utils.rotate_points_along_z(points[np.newaxis, :, :], np.array([noise_rotation]))[0]
        data_dict['points'] = points

        return data_dict

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


class RandomGlobalTranslation(object):
    def __init__(self, translate_std) -> None:
        self.translate_std = translate_std

    def __call__(self, data_dict):
        points = data_dict['points']

        for cur_axis in ['x', 'y', 'z']:
            points = getattr(transform_utils, 'random_translation_along_%s' % cur_axis)(points, self.translate_std)

        data_dict['points'] = points
        return data_dict

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


class RandomFlip(object):
    def __call__(self, data_dict):
        points = data_dict['points']

        for cur_axis in ['x', 'y']:
            points = getattr(transform_utils, 'random_flip_along_%s' % cur_axis)(points)

        data_dict['points'] = points
        return data_dict

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


class PointShuffle(object):
    def __call__(self, data_dict):
        point_indices = np.array(range(data_dict['points'].shape[0]))
        np.random.shuffle(point_indices)

        data_dict['points'] = data_dict['points'][point_indices]

        cur_point_indices = data_dict.get('cur_point_indices', None)
        point_image_features = data_dict.get('point_image_features', None)
        point_labels = data_dict.get('point_labels', None)

        if cur_point_indices is not None:
            cur_shuffled_indices, data_dict['cur_point_indices'] = self.get_shuffled_indices(cur_point_indices, point_indices)
        else:
            cur_shuffled_indices = point_indices

        if point_image_features is not None:
            data_dict['point_image_features'] = point_image_features[cur_shuffled_indices]

        if point_labels is not None:
            data_dict['point_labels'] = point_labels[cur_shuffled_indices]

        return data_dict

    @staticmethod
    def get_shuffled_indices(cur_point_indices, point_indices):
        point_to_cur_index = {}
        for i, point_index in enumerate(cur_point_indices):
            point_to_cur_index[point_index] = i

        cur_shuffled_indices = []
        glb_shuffled_indices = []
        for i, point_index in enumerate(point_indices):
            if point_index in point_to_cur_index:
                cur_shuffled_indices.append(point_to_cur_index[point_index])
                glb_shuffled_indices.append(i)
        return np.array(cur_shuffled_indices), np.array(glb_shuffled_indices)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


class PointSample(object):
    """Point sample.

    Sampling data to a certain number.

    Args:
        sample_ratio (float): Ratio of points to be sampled.
        sample_range (float, optional): The range where to sample points.
            If not None, the points with depth larger than `sample_range` are
            prior to be sampled. Defaults to None.
        replace (bool, optional): Whether the sampling is with or without
            replacement. Defaults to False.
    """

    def __init__(self, sample_ratio, sample_range=None, replace=False):
        self.sample_ratio = sample_ratio
        self.sample_range = sample_range
        self.replace = replace

    def __call__(self, data_dict):
        """Call function to sample points.

        Args:
            data_dict (dict): Result dict from loading pipeline.
        Returns:
            dict: Results after sampling, 'points', 'point_image_features', 'point_indices',
                and 'labels' keys are updated in the result dict.
        """
        points = data_dict['points']
        num_samples = int(points.shape[0] * self.sample_ratio)
        points, point_indices = transform_utils.points_random_sampling(
            points,
            num_samples,
            self.sample_range,
            return_choices=True)
        data_dict['points'] = points

        cur_point_indices = data_dict.get('cur_point_indices', None)
        point_image_features = data_dict.get('point_image_features', None)
        point_labels = data_dict.get('point_labels', None)

        if cur_point_indices is not None:
            cur_shuffled_indices, data_dict['cur_point_indices'] = self.get_shuffled_indices(cur_point_indices, point_indices)
        else:
            cur_shuffled_indices = point_indices

        if point_image_features is not None:
            data_dict['point_image_features'] = point_image_features[cur_shuffled_indices]

        if point_labels is not None:
            data_dict['point_labels'] = point_labels[cur_shuffled_indices]

        return data_dict

    @staticmethod
    def get_shuffled_indices(cur_point_indices, point_indices):
        point_to_cur_index = {}
        for i, point_index in enumerate(cur_point_indices):
            point_to_cur_index[point_index] = i

        cur_shuffled_indices = []
        glb_shuffled_indices = []
        for i, point_index in enumerate(point_indices):
            if point_index in point_to_cur_index:
                cur_shuffled_indices.append(point_to_cur_index[point_index])
                glb_shuffled_indices.append(i)
        return np.array(cur_shuffled_indices), np.array(glb_shuffled_indices)

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += f'(sample_ratio={self.sample_ratio},'
        repr_str += f' sample_range={self.sample_range},'
        repr_str += f' replace={self.replace})'

        return repr_str


if __name__ == '__main__':
    batch_dict = {'points': np.ones((100, 3))}
    transforms = Compose([RandomGlobalScaling([0.95, 1.05]),
                          RandomGlobalRotation([-0.78539816, 0.78539816]),
                          PointShuffle()])
    batch_dict = transforms(batch_dict)
    print(batch_dict)
