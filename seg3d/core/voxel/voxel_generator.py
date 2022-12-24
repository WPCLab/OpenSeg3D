import numba
import numpy as np


class VoxelGenerator(object):
    """Voxel generator in numpy implementation.
    Args:
        voxel_size (list[float]): Size of a single voxel
        point_cloud_range (list[float]): Range of points
    """
    def __init__(self,
                 voxel_size,
                 point_cloud_range):

        point_cloud_range = np.array(point_cloud_range, dtype=np.float32)
        voxel_size = np.array(voxel_size, dtype=np.float32)
        grid_size = (point_cloud_range[3:] - point_cloud_range[:3]) / voxel_size
        grid_size = np.round(grid_size).astype(np.int64)

        self._voxel_size = voxel_size
        self._point_cloud_range = point_cloud_range
        self._grid_size = grid_size

    def generate(self, points):
        """Generate voxels given points."""
        return points_to_voxel(points, self._voxel_size, self._point_cloud_range, True)

    @property
    def voxel_size(self):
        """list[float]: Size of a single voxel."""
        return self._voxel_size

    @property
    def point_cloud_range(self):
        """list[float]: Range of point cloud."""
        return self._point_cloud_range

    @property
    def grid_size(self):
        """np.ndarray: The size of grids."""
        return self._grid_size

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        indent = ' ' * (len(repr_str) + 1)
        repr_str += f'(voxel_size={self._voxel_size},\n'
        repr_str += indent + 'point_cloud_range='
        repr_str += f'{self._point_cloud_range.tolist()},\n'
        repr_str += indent + f'grid_size={self._grid_size.tolist()}'
        repr_str += ')'
        return repr_str


def points_to_voxel(points,
                    voxel_size,
                    coors_range,
                    reverse_index=True):
    """convert kitti points(N, >=3) to voxels.
    Args:
        points (np.ndarray): [N, ndim]. points[:, :3] contain xyz points and
            points[:, 3:] contain other information such as reflectivity.
        voxel_size (list, tuple, np.ndarray): [3] xyz, indicate voxel size
        coors_range (list[float | tuple[float] | ndarray]): Voxel range.
            format: xyzxyz, minmax
        reverse_index (bool): Whether return reversed coordinates.
            if points has xyz format and reverse_index is True, output
            coordinates will be zyx format, but points in features always
            xyz format.
    Returns:
        tuple[np.ndarray]:
            coordinates: [M, 3] int32 tensor.
            point_voxel_ids: [N] int32 tensor.
    """
    if not isinstance(voxel_size, np.ndarray):
        voxel_size = np.array(voxel_size, dtype=points.dtype)
    if not isinstance(coors_range, np.ndarray):
        coors_range = np.array(coors_range, dtype=points.dtype)
    voxelmap_shape = (coors_range[3:] - coors_range[:3]) / voxel_size
    voxelmap_shape = tuple(np.round(voxelmap_shape).astype(np.int32).tolist())
    if reverse_index:
        voxelmap_shape = voxelmap_shape[::-1]
    coor_to_voxelidx = -np.ones(shape=voxelmap_shape, dtype=np.int32)
    coors = np.zeros(shape=(np.prod(voxelmap_shape), 3), dtype=np.int32)
    point_voxel_ids = -np.ones(shape=(points.shape[0]), dtype=np.int32)
    if reverse_index:
        voxel_num = _points_to_voxel_reverse_kernel(points, voxel_size, coors_range, coor_to_voxelidx,
                                                    coors, point_voxel_ids)

    else:
        voxel_num = _points_to_voxel_kernel(points, voxel_size, coors_range, coor_to_voxelidx,
                                            coors, point_voxel_ids)

    coors = coors[:voxel_num]
    return coors, point_voxel_ids


@numba.jit(nopython=True)
def _points_to_voxel_reverse_kernel(points,
                                    voxel_size,
                                    coors_range,
                                    coor_to_voxelidx,
                                    coors,
                                    point_voxel_ids):
    """convert kitti points(N, >=3) to voxels.
    Args:
        points (np.ndarray): [N, ndim]. points[:, :3] contain xyz points and
            points[:, 3:] contain other information such as reflectivity.
        voxel_size (list, tuple, np.ndarray): [3] xyz, indicate voxel size
        coors_range (list[float | tuple[float] | ndarray]): Range of voxels.
            format: xyzxyz, minmax
        coor_to_voxelidx (np.ndarray): A voxel grid of shape (D, H, W),
            which has the same shape as the complete voxel map. It indicates
            the index of each corresponding voxel.
        coors (np.ndarray): Created coordinates of each voxel.
        point_voxel_ids (np.ndarray):  Created voxel index of each point.
    Returns:
        tuple[np.ndarray]:
            coordinates: Shape [M, 3].
            point_voxel_ids: Shape [N].
    """
    # put all computations to one loop.
    # we shouldn't create large array in main jit code, otherwise
    # reduce performance
    N = points.shape[0]
    # ndim = points.shape[1] - 1
    ndim = 3
    ndim_minus_1 = ndim - 1
    grid_size = (coors_range[3:] - coors_range[:3]) / voxel_size
    # np.round(grid_size)
    # grid_size = np.round(grid_size).astype(np.int64)(np.int32)
    grid_size = np.round(grid_size, 0, grid_size).astype(np.int32)
    coor = np.zeros(shape=(3, ), dtype=np.int32)
    voxel_num = 0
    failed = False
    for i in range(N):
        failed = False
        for j in range(ndim):
            c = np.floor((points[i, j] - coors_range[j]) / voxel_size[j])
            if c < 0 or c >= grid_size[j]:
                failed = True
                break
            coor[ndim_minus_1 - j] = c
        if failed:
            continue
        voxelidx = coor_to_voxelidx[coor[0], coor[1], coor[2]]
        if voxelidx == -1:
            voxelidx = voxel_num
            voxel_num += 1
            coor_to_voxelidx[coor[0], coor[1], coor[2]] = voxelidx
            coors[voxelidx] = coor
        point_voxel_ids[i] = voxelidx
    return voxel_num


@numba.jit(nopython=True)
def _points_to_voxel_kernel(points,
                            voxel_size,
                            coors_range,
                            coor_to_voxelidx,
                            coors,
                            point_voxel_ids):
    """convert kitti points(N, >=3) to voxels.
    Args:
        points (np.ndarray): [N, ndim]. points[:, :3] contain xyz points and
            points[:, 3:] contain other information such as reflectivity.
        voxel_size (list, tuple, np.ndarray): [3] xyz, indicate voxel size.
        coors_range (list[float | tuple[float] | ndarray]): Range of voxels.
            format: xyzxyz, minmax
        coor_to_voxelidx (np.ndarray): A voxel grid of shape (D, H, W),
            which has the same shape as the complete voxel map. It indicates
            the index of each corresponding voxel.
        coors (np.ndarray): Created coordinates of each voxel.
        point_voxel_ids (np.ndarray):  Created voxel index of each point.
    Returns:
        tuple[np.ndarray]:
            coordinates: Shape [M, 3].
            point_voxel_ids: Shape [N].
    """
    N = points.shape[0]
    # ndim = points.shape[1] - 1
    ndim = 3
    grid_size = (coors_range[3:] - coors_range[:3]) / voxel_size
    # grid_size = np.round(grid_size).astype(np.int64)(np.int32)
    grid_size = np.round(grid_size, 0, grid_size).astype(np.int32)

    # lower_bound = coors_range[:3]
    # upper_bound = coors_range[3:]
    coor = np.zeros(shape=(3, ), dtype=np.int32)
    voxel_num = 0
    failed = False
    for i in range(N):
        failed = False
        for j in range(ndim):
            c = np.floor((points[i, j] - coors_range[j]) / voxel_size[j])
            if c < 0 or c >= grid_size[j]:
                failed = True
                break
            coor[j] = c
        if failed:
            continue
        voxelidx = coor_to_voxelidx[coor[0], coor[1], coor[2]]
        if voxelidx == -1:
            voxelidx = voxel_num
            voxel_num += 1
            coor_to_voxelidx[coor[0], coor[1], coor[2]] = voxelidx
            coors[voxelidx] = coor
        point_voxel_ids[i] = voxelidx
    return voxel_num
