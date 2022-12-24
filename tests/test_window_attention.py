import torch

from seg3d.models.layers import SparseWindowPartitionLayer, WindowAttention


if __name__ == '__main__':
    batching_info = {
        0: {'max_tokens': 60, 'batching_range': (0, 60)},
        1: {'max_tokens': 120, 'batching_range': (60, 120)},
        2: {'max_tokens': 180, 'batching_range': (120, 180)},
        3: {'max_tokens': 400, 'batching_range': (180, 100000)}
    }
    window_shape = (10, 10, 10)
    sparse_shape = (400, 400, 20)
    normalize_pos = False
    pos_temperature = 10000
    window_partition = SparseWindowPartitionLayer(batching_info, window_shape, sparse_shape)

    voxel_features = torch.randn((4, 24)).cuda()
    voxel_coords = torch.zeros((4, 4)).cuda()
    voxel_info = window_partition(voxel_features, voxel_coords)

    window_attention = WindowAttention(24, 2, 0.5).cuda()
    result = window_attention(voxel_info['voxel_features'], voxel_info['pos_dict_shift0'],
                              voxel_info['flat2win_inds_shift0'], voxel_info['key_mask_shift0'])
    print(result)


