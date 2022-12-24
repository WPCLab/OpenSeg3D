from functools import partial

import torch
import torch.nn as nn

import spconv.pytorch as spconv

from seg3d.models.layers import FlattenSELayer, SALayer, SparseWindowPartitionLayer
from seg3d.models.layers.point_transformer_layer import SWFormerBlock
from seg3d.utils.spconv_utils import replace_feature, ConvModule


class SparseBasicBlock(spconv.SparseModule):
    """
    Basic block implemented with submanifold sparse convolution.
    """

    expansion = 1

    def __init__(self, inplanes, planes, stride=1, with_se=False, with_sa=False,
                 norm_fn=None, act_fn=None, indice_key=None):
        super(SparseBasicBlock, self).__init__()

        assert norm_fn is not None
        bias = norm_fn is not None
        self.conv1 = spconv.SubMConv3d(
            inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=bias, indice_key=indice_key
        )
        self.bn1 = norm_fn(planes)
        self.act = act_fn
        self.conv2 = spconv.SubMConv3d(
            planes, planes, kernel_size=3, stride=stride, padding=1, bias=bias, indice_key=indice_key
        )
        self.bn2 = norm_fn(planes)

        # spatial and channel attention
        if with_se:
            self.se = FlattenSELayer(planes)
        else:
            self.se = None

        if with_sa:
            self.sa = SALayer(planes, indice_key=indice_key)
        else:
            self.sa = None

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = replace_feature(out, self.bn1(out.features))
        out = replace_feature(out, self.act(out.features))

        out = self.conv2(out)
        out = replace_feature(out, self.bn2(out.features))

        if self.se is not None:
            out = replace_feature(out, self.se(out.features, out.indices[:, 0]))

        if self.sa is not None:
            out = self.sa(out)

        out = replace_feature(out, out.features + identity.features)
        out = replace_feature(out, self.act(out.features))

        return out


class UpBlock(spconv.SparseModule):
    def __init__(self, inplanes, planes, norm_fn, act_fn, conv_type, layer_id):
        super(UpBlock, self).__init__()

        self.transform = SparseBasicBlock(inplanes, inplanes, norm_fn=norm_fn, act_fn=act_fn,
                                          indice_key='subm' + str(layer_id))

        self.bottleneck = ConvModule(2 * inplanes, inplanes, 3, padding=1, norm_fn=norm_fn, act_fn=act_fn,
                                     indice_key='subm' + str(layer_id))

        if conv_type == 'inverseconv':
            self.out = ConvModule(inplanes, planes, 3, norm_fn=norm_fn, act_fn=act_fn,
                                  conv_type=conv_type, indice_key='spconv' + str(layer_id))
        elif conv_type == 'subm':
            self.out = ConvModule(inplanes, planes, 3, padding=1, norm_fn=norm_fn, act_fn=act_fn,
                                  conv_type=conv_type, indice_key='subm' + str(layer_id))
        else:
            raise NotImplementedError

    @staticmethod
    def channel_reduction(x, out_channels):
        """
        Args:
            x: x.features (N, C1)
            out_channels: C2

        Returns:

        """
        features = x.features
        n, in_channels = features.shape
        assert (in_channels % out_channels == 0) and (in_channels >= out_channels)

        x = replace_feature(x, features.view(n, out_channels, -1).sum(dim=2))
        return x

    def forward(self, x_bottom, x_lateral):
        x_trans = self.transform(x_lateral)
        x = x_trans
        x = replace_feature(x, torch.cat([x_bottom.features, x_trans.features], dim=1))
        x_m = self.bottleneck(x)
        x = self.channel_reduction(x, x_m.features.shape[1])
        x = replace_feature(x, x_m.features + x.features)
        x = self.out(x)
        return x


class PointTransformer(nn.Module):
    def __init__(self, input_channels, output_channels, grid_size, voxel_size, point_cloud_range,
                 batching_info, window_shape, drop_path_rate, depths, num_classes):
        super(PointTransformer, self).__init__()
        self.sparse_shape = grid_size[::-1]
        self.voxel_size = voxel_size
        self.point_cloud_range = point_cloud_range

        self.batching_info = batching_info
        self.depths = depths
        self.window_shape = window_shape
        self.drop_path_rate = drop_path_rate

        self.norm_fn = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)
        self.act_fn = nn.ReLU(inplace=True)

        self.conv_input = spconv.SparseSequential(
            spconv.SubMConv3d(input_channels, 48, 3, padding=1, bias=False, indice_key='subm1'),
            self.norm_fn(48),
            self.act_fn
        )

        # stochastic depth decay rule
        dpr = [x.item() for x in torch.linspace(0, self.drop_path_rate, sum(self.depths))]

        self.swformer_block1 = nn.Sequential(SparseWindowPartitionLayer(self.batching_info[0], self.window_shape,
                                                                        self.sparse_shape[::-1]),
                                             SWFormerBlock(48, 8, depth=self.depths[0],
                                                           drop_path=dpr[sum(self.depths[:0]):sum(self.depths[:1])]))
        self.swformer_block2 = nn.Sequential(SparseWindowPartitionLayer(self.batching_info[1], self.window_shape,
                                                                        self.sparse_shape[::-1]/2),
                                             SWFormerBlock(96, 8, depth=self.depths[1],
                                                           drop_path=dpr[sum(self.depths[:1]):sum(self.depths[:2])]))
        self.swformer_block3 = nn.Sequential(SparseWindowPartitionLayer(self.batching_info[2], self.window_shape,
                                                                        self.sparse_shape[::-1]/4),
                                             SWFormerBlock(192, 8, depth=self.depths[2],
                                                           drop_path=dpr[sum(self.depths[:2]):sum(self.depths[:3])]))
        self.swformer_block4 = nn.Sequential(SparseWindowPartitionLayer(self.batching_info[3], self.window_shape,
                                                                        self.sparse_shape[::-1]/8),
                                             SWFormerBlock(384, 8, depth=self.depths[3],
                                                           drop_path=dpr[sum(self.depths[:3]):sum(self.depths[:4])]))

        # [1440, 1440, 64] -> [720, 720, 32]
        self.conv_down1 = ConvModule(48, 96, 3, norm_fn=self.norm_fn, act_fn=self.act_fn, stride=2, padding=1,
                                     conv_type='spconv', indice_key='spconv2')
        # [720, 720, 32] -> [360, 360, 16]
        self.conv_down2 = ConvModule(96, 192, 3, norm_fn=self.norm_fn, act_fn=self.act_fn, stride=2, padding=1,
                                     conv_type='spconv', indice_key='spconv3')
        # [360, 360, 16] -> [180, 180, 8]
        self.conv_down3 = ConvModule(192, 384, 3, norm_fn=self.norm_fn, act_fn=self.act_fn, stride=2, padding=1,
                                     conv_type='spconv', indice_key='spconv4')

        # [180, 180, 8] -> [360, 360, 16]
        self.up4 = UpBlock(384, 192, self.norm_fn, self.act_fn, conv_type='inverseconv', layer_id=4)
        # [360, 360, 16] -> [720, 720, 32]
        self.up3 = UpBlock(192, 96, self.norm_fn, self.act_fn, conv_type='inverseconv', layer_id=3)
        # [720, 720, 32] -> [1440, 1440, 64]
        self.up2 = UpBlock(96, 48, self.norm_fn, self.act_fn, conv_type='inverseconv', layer_id=2)
        # [1440, 1440, 64] -> [1440, 1440, 64]
        self.up1 = UpBlock(48, output_channels, self.norm_fn, self.act_fn, conv_type='subm', layer_id=1)

        self.aux_voxel_classifier = nn.Sequential(nn.Linear(384, num_classes, bias=False))

        self.voxel_classifier = nn.Sequential(nn.Linear(output_channels, num_classes, bias=False))

    def forward(self, batch_dict):
        voxel_features, voxel_coords = batch_dict['voxel_features'], batch_dict['voxel_coords']
        batch_size = batch_dict['batch_size']
        x = spconv.SparseConvTensor(
            features=voxel_features,
            indices=voxel_coords.int(),
            spatial_shape=self.sparse_shape,
            batch_size=batch_size
        )

        # encoder
        x_conv1 = self.conv_input(x)
        x_conv1 = replace_feature(x_conv1, self.swformer_block1(x_conv1))

        x_conv2 = self.conv_down1(x_conv1)
        x_conv2 = replace_feature(x_conv2, self.swformer_block2(x_conv2))

        x_conv3 = self.conv_down2(x_conv2)
        x_conv3 = replace_feature(x_conv3, self.swformer_block3(x_conv3))

        x_conv4 = self.conv_down3(x_conv3)
        x_conv4 = replace_feature(x_conv4, self.swformer_block4(x_conv4))

        # auxiliary branch
        aux_voxel_out = self.aux_voxel_classifier(x_conv4.features)
        batch_dict['aux_voxel_out'] = aux_voxel_out
        batch_dict['aux_voxel_coords'] = x_conv4.indices

        # decoder
        x_conv4 = self.up4(x_conv4, x_conv4)
        x_conv3 = self.up3(x_conv4, x_conv3)
        x_conv2 = self.up2(x_conv3, x_conv2)
        x_conv1 = self.up1(x_conv2, x_conv1)

        batch_dict['voxel_features'] = x_conv1.features
        batch_dict['voxel_coords'] = x_conv1.indices
        batch_dict['voxel_out'] = self.voxel_classifier(x_conv1.features)

        return batch_dict
