import numpy as np

from seg3d.datasets.transforms import transform_utils


class MultiScaleFlipAug(object):
    """Test-time augmentation with multiple scales and flipping."""
    def __init__(self, scales=None, angles=None, flip_x=False, flip_y=False):
        self.scales = scales
        self.angles = angles
        self.flip_x = [True, False] if flip_x else [False]
        self.flip_y = [True, False] if flip_y else [False]

    def __call__(self, data):
        """Call function to apply test time augment transforms on results."""
        aug_data_list = []
        for scale in self.scales:
            for angle in self.angles:
                for flip_x in self.flip_x:
                    for flip_y in self.flip_y:
                        new_data = dict()
                        for key, val in data.items():
                            if isinstance(val, np.ndarray):
                                new_data[key] = val.copy()
                            else:
                                new_data[key] = val
                        points = new_data['points'][:, 1:4]
                        points[:, :3] *= scale
                        points = transform_utils.rotate_points_along_z(points[np.newaxis, :, :], np.array([angle]))[0]
                        if flip_x:
                            points[:, 1] = -points[:, 1]
                        if flip_y:
                            points[:, 0] = -points[:, 0]
                        new_data['points'][:, 1:4] = points
                        aug_data_list.append(new_data)
        return aug_data_list

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(scales={self.scales}, '
        repr_str += f'(angles={self.angles}, '
        repr_str += f'(flip_x={self.flip_x}, '
        repr_str += f'(flip_y={self.flip_y}'
        return repr_str
