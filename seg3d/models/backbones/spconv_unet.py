from functools import partial

import torch
import torch.nn as nn

import spconv.pytorch as spconv

from seg3d.utils.spconv_utils import replace_feature, ConvModule
from seg3d.models.layers import FlattenSELayer, SALayer, OCRLayer


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


class SparseUnet(nn.Module):
    """
    Sparse Convolution based UNet for point-wise feature learning.
    Reference Paper: https://arxiv.org/abs/1907.03670 (Shaoshuai Shi, et. al)
    From Points to Parts: 3D Object Detection from Point Cloud with Part-aware and Part-aggregation Network
    """

    def __init__(self, input_channels, output_channels, grid_size, voxel_size,
                 point_cloud_range, num_classes, use_ocr=True):
        super(SparseUnet, self).__init__()
        self.sparse_shape = grid_size[::-1]
        self.voxel_size = voxel_size
        self.point_cloud_range = point_cloud_range
        self.use_ocr = use_ocr

        self.norm_fn = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)
        self.act_fn = nn.ReLU(inplace=True)

        self.conv_input = spconv.SparseSequential(
            spconv.SubMConv3d(input_channels, 48, 3, padding=1, bias=False, indice_key='subm1'),
            self.norm_fn(48),
            self.act_fn
        )

        self.conv1 = spconv.SparseSequential(
            SparseBasicBlock(48, 48, norm_fn=self.norm_fn, act_fn=self.act_fn, indice_key='subm1'),
            SparseBasicBlock(48, 48, norm_fn=self.norm_fn, act_fn=self.act_fn, indice_key='subm1')
        )

        # [1504, 1504, 72] -> [752, 752, 36]
        self.conv2 = spconv.SparseSequential(
            ConvModule(48, 96, 3, norm_fn=self.norm_fn, act_fn=self.act_fn, stride=2, padding=1,
                       conv_type='spconv', indice_key='spconv2'),
            SparseBasicBlock(96, 96, norm_fn=self.norm_fn, act_fn=self.act_fn, indice_key='subm2'),
            SparseBasicBlock(96, 96, norm_fn=self.norm_fn, act_fn=self.act_fn, indice_key='subm2'),
            SparseBasicBlock(96, 96, norm_fn=self.norm_fn, act_fn=self.act_fn, indice_key='subm2')
        )

        # [752, 752, 36] -> [376, 376, 18]
        self.conv3 = spconv.SparseSequential(
            ConvModule(96, 192, 3, norm_fn=self.norm_fn, act_fn=self.act_fn, stride=2, padding=1,
                       conv_type='spconv', indice_key='spconv3'),
            SparseBasicBlock(192, 192, norm_fn=self.norm_fn, act_fn=self.act_fn, indice_key='subm3'),
            SparseBasicBlock(192, 192, norm_fn=self.norm_fn, act_fn=self.act_fn, indice_key='subm3'),
            SparseBasicBlock(192, 192, norm_fn=self.norm_fn, act_fn=self.act_fn, with_se=True, indice_key='subm3')
        )

        # [376, 376, 18] -> [188, 188, 9]
        self.conv4 = spconv.SparseSequential(
            ConvModule(192, 384, 3, norm_fn=self.norm_fn, act_fn=self.act_fn, stride=2, padding=1,
                       conv_type='spconv', indice_key='spconv4'),
            SparseBasicBlock(384, 384, norm_fn=self.norm_fn, act_fn=self.act_fn, indice_key='subm4'),
            SparseBasicBlock(384, 384, norm_fn=self.norm_fn, act_fn=self.act_fn, indice_key='subm4'),
            SparseBasicBlock(384, 384, norm_fn=self.norm_fn, act_fn=self.act_fn, with_se=True, indice_key='subm4')
        )

        # [188, 188, 9] -> [376, 376, 18]
        self.up4 = UpBlock(384, 192, self.norm_fn, self.act_fn, conv_type='inverseconv', layer_id=4)
        # [376, 376, 18] -> [752, 752, 36]
        self.up3 = UpBlock(192, 96, self.norm_fn, self.act_fn, conv_type='inverseconv', layer_id=3)
        # [752, 752, 36] -> [1504, 1504, 72]
        self.up2 = UpBlock(96, 48, self.norm_fn, self.act_fn, conv_type='inverseconv', layer_id=2)
        # [1504, 1504, 72] -> [1504, 1504, 72]
        self.up1 = UpBlock(48, output_channels, self.norm_fn, self.act_fn, conv_type='subm', layer_id=1)

        if self.use_ocr:
            self.ocr = OCRLayer(384, 192, 96)

        self.aux_voxel_classifier = nn.Sequential(nn.Linear(384, num_classes, bias=False))

        self.voxel_classifier = nn.Sequential(nn.Linear(output_channels, num_classes, bias=False))

    def forward(self, batch_dict):
        """
        Args:
            batch_dict:
                batch_size: int
                voxel_features: (num_voxels, Cin)
                voxel_coords: (num_voxels, 4), [batch_idx, z_idx, y_idx, x_idx]
        Returns:
            batch_dict:
                voxel_features: (num_voxels, Cout)
                aux_voxel_features: (num_voxels, Cout)
        """
        voxel_features, voxel_coords = batch_dict['voxel_features'], batch_dict['voxel_coords']
        batch_size = batch_dict['batch_size']
        input_sp_tensor = spconv.SparseConvTensor(
            features=voxel_features,
            indices=voxel_coords.int(),
            spatial_shape=self.sparse_shape,
            batch_size=batch_size
        )

        # encoder
        x = self.conv_input(input_sp_tensor)
        x_conv1 = self.conv1(x)
        x_conv2 = self.conv2(x_conv1)
        x_conv3 = self.conv3(x_conv2)
        x_conv4 = self.conv4(x_conv3)

        # auxiliary branch
        aux_voxel_out = self.aux_voxel_classifier(x_conv4.features)
        batch_dict['aux_voxel_out'] = aux_voxel_out
        batch_dict['aux_voxel_coords'] = x_conv4.indices

        if self.use_ocr:
            x_conv4 = self.ocr(x_conv4, aux_voxel_out, batch_size)

        # decoder
        x_conv4 = self.up4(x_conv4, x_conv4)
        x_conv3 = self.up3(x_conv4, x_conv3)
        x_conv2 = self.up2(x_conv3, x_conv2)
        x_conv1 = self.up1(x_conv2, x_conv1)

        batch_dict['voxel_features'] = x_conv1.features
        batch_dict['voxel_coords'] = x_conv1.indices
        batch_dict['voxel_out'] = self.voxel_classifier(x_conv1.features)

        return batch_dict
