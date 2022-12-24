import torch.nn as nn

from torch_scatter import scatter


class VFE(nn.Module):
    def __init__(self, voxel_feature_channel, reduce='mean'):
        super(VFE, self).__init__()
        self._voxel_feature_channel = voxel_feature_channel
        self.reduce = reduce

    @property
    def voxel_feature_channel(self):
        return self._voxel_feature_channel

    def forward(self, features, index):
        """
        Args:
            features: (N, C)
            index: (N)
        Returns:
            vfe_features: (num_voxels, C)
        """
        mask = (index != -1)
        voxel_features = scatter(features[mask], index[mask], dim=0, reduce=self.reduce)

        return voxel_features
