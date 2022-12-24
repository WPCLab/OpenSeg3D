import torch
import numpy as np

from seg3d.ops import get_inner_win_inds


@torch.no_grad()
def get_flat2win_inds(batch_win_inds, voxel_batching_lvl, batching_info):
    """
    Args:
        batch_win_inds: shape=[N, ]. Indicates which window a voxel belongs to. Window inds is unique is the whole batch.
        voxel_batching_lvl: shape=[N, ]. Indicates batching_level of the window the voxel belongs to.
        batching_info: Batching configuration for region batching.
    Returns:
        flat2window_inds_dict: contains flat2window_inds of each voxel, shape=[N,]
            Determine the voxel position in range [0, num_windows * max_tokens) of each voxel.
    """
    flat2window_inds_dict = {}

    for bl in batching_info:  # bl: short for batching level
        dl_mask = voxel_batching_lvl == bl
        if not dl_mask.any():
            continue

        conti_win_inds = make_continuous_inds(batch_win_inds[dl_mask])
        max_tokens = batching_info[bl]['max_tokens']
        inner_win_inds = get_inner_win_inds(conti_win_inds)
        flat2window_inds = conti_win_inds * max_tokens + inner_win_inds
        flat2window_inds_dict[bl] = (flat2window_inds, torch.where(dl_mask))

    return flat2window_inds_dict


def flat2window(feat, voxel_batching_lvl, flat2win_inds_dict, batching_info):
    """
    Args:
        feat: shape=[N, C], N is the voxel num in the batch.
        voxel_batching_lvl: shape=[N, ]. Indicates batching_level of the window the voxel belongs to.
        flat2win_inds_dict: Contains flat2window_inds of each voxel, shape=[N,]
        batching_info: Batching configuration for region batching.
    Returns:
        feat_3d_dict: contains feat_3d of each batching level. Shape of feat_3d is [num_windows, num_max_tokens, C].
    """
    dtype = feat.dtype
    device = feat.device
    feat_dim = feat.shape[-1]

    feat_3d_dict = {}
    for bl in batching_info:
        bl_mask = voxel_batching_lvl == bl
        if not bl_mask.any():
            continue

        feat_this_dl = feat[bl_mask]

        this_inds = flat2win_inds_dict[bl][0]
        max_tokens = batching_info[bl]['max_tokens']
        num_windows = torch.div(this_inds, max_tokens, rounding_mode='floor').max().item() + 1
        feat_3d = torch.zeros((num_windows * max_tokens, feat_dim), dtype=dtype, device=device)
        feat_3d[this_inds] = feat_this_dl
        feat_3d = feat_3d.reshape((num_windows, max_tokens, feat_dim))
        feat_3d_dict[bl] = feat_3d

    return feat_3d_dict


def window2flat(feat_3d_dict, inds_dict):
    num_all_voxel = 0
    for dl in inds_dict:
        num_all_voxel += inds_dict[dl][0].shape[0]

    dtype = feat_3d_dict[list(feat_3d_dict.keys())[0]].dtype
    device = feat_3d_dict[list(feat_3d_dict.keys())[0]].device
    feat_dim = feat_3d_dict[list(feat_3d_dict.keys())[0]].shape[-1]

    all_flat_feat = torch.zeros((num_all_voxel, feat_dim), device=device, dtype=dtype)
    for dl in feat_3d_dict:
        feat = feat_3d_dict[dl]
        feat_dim = feat.shape[-1]
        inds, flat_pos = inds_dict[dl]
        feat = feat.reshape(-1, feat_dim)
        flat_feat = feat[inds]
        all_flat_feat[flat_pos] = flat_feat

    return all_flat_feat


def get_flat2win_inds_v2(batch_win_inds, voxel_batching_lvl, batching_info):
    transform_dict = get_flat2win_inds(batch_win_inds, voxel_batching_lvl, batching_info)
    # add voxel_batching_lvl and batching_info into transform_dict for better wrapping
    transform_dict['voxel_batching_level'] = voxel_batching_lvl
    transform_dict['batching_info'] = batching_info
    return transform_dict


def window2flat_v2(feat_3d_dict, inds_dict):
    inds_v1 = {k: inds_dict[k] for k in inds_dict if not isinstance(k, str)}
    return window2flat(feat_3d_dict, inds_v1)


def flat2window_v2(feat, inds_dict):
    assert 'voxel_batching_level' in inds_dict, 'voxel_batching_level should be in inds_dict in v2 function'
    inds_v1 = {k: inds_dict[k] for k in inds_dict if not isinstance(k, str)}
    batching_info = inds_dict['batching_info']
    return flat2window(feat, inds_dict['voxel_batching_level'], inds_v1, batching_info)


@torch.no_grad()
def get_window_coors(coors, sparse_shape, window_shape, do_shift):
    if len(window_shape) == 2:
        win_shape_x, win_shape_y = window_shape
        win_shape_z = sparse_shape[-1]
    else:
        win_shape_x, win_shape_y, win_shape_z = window_shape

    sparse_shape_x, sparse_shape_y, sparse_shape_z = sparse_shape
    assert sparse_shape_z < sparse_shape_x, 'Usually holds... in case of wrong order'

    max_num_win_x = int(np.ceil((sparse_shape_x / win_shape_x)) + 1)  # plus one here to meet the needs of shift.
    max_num_win_y = int(np.ceil((sparse_shape_y / win_shape_y)) + 1)  # plus one here to meet the needs of shift.
    max_num_win_z = int(np.ceil((sparse_shape_z / win_shape_z)) + 1)  # plus one here to meet the needs of shift.
    max_num_win_per_sample = max_num_win_x * max_num_win_y * max_num_win_z

    if do_shift:
        shift_x, shift_y, shift_z = win_shape_x // 2, win_shape_y // 2, win_shape_z // 2
    else:
        shift_x, shift_y, shift_z = win_shape_x, win_shape_y, win_shape_z

    # compatibility between 2D window and 3D window
    if sparse_shape_z == win_shape_z:
        shift_z = 0

    shifted_coors_x = coors[:, 3] + shift_x
    shifted_coors_y = coors[:, 2] + shift_y
    shifted_coors_z = coors[:, 1] + shift_z

    win_coors_x = torch.div(shifted_coors_x, win_shape_x, rounding_mode='floor')
    win_coors_y = torch.div(shifted_coors_y, win_shape_y, rounding_mode='floor')
    win_coors_z = torch.div(shifted_coors_z, win_shape_z, rounding_mode='floor')

    if len(window_shape) == 2:
        assert (win_coors_z == 0).all()

    batch_win_inds = coors[:, 0] * max_num_win_per_sample + \
                     win_coors_x * max_num_win_y * max_num_win_z + \
                     win_coors_y * max_num_win_z + \
                     win_coors_z

    coors_in_win_x = shifted_coors_x % win_shape_x
    coors_in_win_y = shifted_coors_y % win_shape_y
    coors_in_win_z = shifted_coors_z % win_shape_z
    coors_in_win = torch.stack([coors_in_win_z, coors_in_win_y, coors_in_win_x], dim=-1)

    return batch_win_inds, coors_in_win


@torch.no_grad()
def make_continuous_inds(inds):
    # make batch_win_inds continuous
    dtype = inds.dtype
    device = inds.device

    unique_inds, _ = torch.sort(torch.unique(inds))
    num_valid_inds = len(unique_inds)
    max_origin_inds = unique_inds.max().item()
    canvas = -torch.ones((max_origin_inds + 1,), dtype=dtype, device=device)
    canvas[unique_inds] = torch.arange(num_valid_inds, dtype=dtype, device=device)

    conti_inds = canvas[inds]

    return conti_inds
