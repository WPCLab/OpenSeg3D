import torch
import numpy as np


def check_numpy_to_torch(x):
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x).float(), True
    return x, False


def rotate_points_along_z(points, angle):
    """
    Args:
        points: (B, N, 3 + C)
        angle: (B), angle along z-axis, angle increases x ==> y
    Returns:
    """
    points, is_numpy = check_numpy_to_torch(points)
    angle, _ = check_numpy_to_torch(angle)

    cosa = torch.cos(angle)
    sina = torch.sin(angle)
    zeros = angle.new_zeros(points.shape[0])
    ones = angle.new_ones(points.shape[0])
    rot_matrix = torch.stack((
        cosa, sina, zeros,
        -sina, cosa, zeros,
        zeros, zeros, ones
    ), dim=1).view(-1, 3, 3).float()
    points_rot = torch.matmul(points[:, :, 0:3], rot_matrix)
    points_rot = torch.cat((points_rot, points[:, :, 3:]), dim=-1)
    return points_rot.numpy() if is_numpy else points_rot


def random_flip_along_x(points):
    """
    Args:
        points: (M, 3 + C)
    Returns:
    """
    enable = np.random.choice([False, True], replace=False, p=[0.5, 0.5])
    if enable:
        points[:, 1] = -points[:, 1]

    return points


def random_flip_along_y(points):
    """
    Args:
        points: (M, 3 + C)
    Returns:
    """
    enable = np.random.choice([False, True], replace=False, p=[0.5, 0.5])
    if enable:
        points[:, 0] = -points[:, 0]

    return points


def random_translation_along_x(points, offset_std):
    """
    Args:
        points: (M, 3 + C),
        offset_std: float
    Returns:
    """
    offset = np.random.normal(0, offset_std, 1)
    points[:, 0] += offset
    return points


def random_translation_along_y(points, offset_std):
    """
    Args:
        points: (M, 3 + C),
        offset_std: float
    Returns:
    """
    offset = np.random.normal(0, offset_std, 1)
    points[:, 1] += offset
    return points


def random_translation_along_z(points, offset_std):
    """
    Args:
        points: (M, 3 + C),
        offset_std: float
    Returns:
    """
    offset = np.random.normal(0, offset_std, 1)
    points[:, 2] += offset
    return points


def points_random_sampling(points,
                           num_samples,
                           sample_range=None,
                           return_choices=False):
    """Points random sampling.

    Sample points to a certain number.

    Args:
        points (np.ndarray): 3D Points.
        num_samples (int): Count of samples to be sampled.
        sample_range (float, optional): Indicating the range where the
            points will be sampled. Defaults to None.
        return_choices (bool, optional): Whether return choice.
            Defaults to False.
    Returns:
        tuple[np.ndarray] | np.ndarray:
            - points (np.ndarray): 3D Points.
            - choices (np.ndarray, optional): The generated random samples.
    """
    num_points = len(points)
    num_samples = min(num_samples, num_points)
    point_range = range(num_points)
    if sample_range is not None:
        # Only sampling the near points when len(points) >= num_samples
        dist = np.linalg.norm(points[:, :2], axis=1)
        far_inds = np.where(dist >= sample_range)[0]
        near_inds = np.where(dist < sample_range)[0]
        # in case there are too many far points
        if len(far_inds) > num_samples:
            far_inds = np.random.choice(far_inds, num_samples, replace=False)
        point_range = near_inds
        num_samples -= len(far_inds)
    choices = np.random.choice(point_range, num_samples, replace=False)
    if sample_range is not None:
        choices = np.concatenate((far_inds, choices))
        # Shuffle points after sampling
        np.random.shuffle(choices)
    if return_choices:
        return points[choices], choices
    else:
        return points[choices]
