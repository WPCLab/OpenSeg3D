from collections import OrderedDict

import torch
import torch.nn as nn

from seg3d.models.voxel_encoders import VFE
from seg3d.models.backbones import PointTransformer
from seg3d.models.layers import FlattenSELayer, DeepFusionBlock
from seg3d.ops import voxel_to_point


class Segformer(nn.Module):
    def __init__(self, dataset, batching_info, window_shape, depths, drop_path_rate):
        super(Segformer, self).__init__()

        dim_point = dataset.dim_point
        if dataset.use_cylinder:
            dim_point = dim_point + 2

        self.point_feature_channel = 64
        self.point_encoder = nn.Sequential(
            nn.BatchNorm1d(dim_point),
            nn.Linear(dim_point, 64, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 128, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 256, bias=False),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Linear(256, self.point_feature_channel))

        self.use_multi_sweeps = dataset.use_multi_sweeps
        if self.use_multi_sweeps:
            self.vfe = VFE(dim_point, reduce='mean')
        else:
            self.vfe = VFE(self.point_feature_channel, reduce='max')

        self.scatter = VFE(3, reduce='mean')

        self.voxel_in_feature_channel = self.vfe.voxel_feature_channel
        self.voxel_feature_channel = 32
        self.point_transformer = PointTransformer(self.voxel_in_feature_channel, self.voxel_feature_channel,
                                                  dataset.grid_size, dataset.voxel_size, dataset.point_cloud_range,
                                                  batching_info=batching_info, window_shape=window_shape, depths=depths,
                                                  drop_path_rate=drop_path_rate, num_classes=dataset.num_classes)

        self.use_image_feature = dataset.use_image_feature
        if self.use_image_feature:
            self.image_feature_channel = dataset.dim_image_feature
            self.deep_fusion = DeepFusionBlock(self.point_feature_channel + self.voxel_feature_channel,
                                               self.image_feature_channel, 32, 16)
        else:
            self.image_feature_channel = 0

        self.fusion_feature_channel = 64
        self.fusion_encoder = nn.Sequential(
            nn.Linear(self.point_feature_channel + self.voxel_feature_channel + self.image_feature_channel, 256, bias=False),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 128, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Linear(128, self.fusion_feature_channel, bias=False),
            nn.BatchNorm1d(self.fusion_feature_channel),
            nn.ReLU(inplace=True)
        )

        self.se = FlattenSELayer(self.fusion_feature_channel)

        self.classifier = nn.Sequential(nn.Linear(self.fusion_feature_channel, 64, bias=False),
                                        nn.BatchNorm1d(64),
                                        nn.ReLU(True),
                                        nn.Dropout(0.3),
                                        nn.Linear(64, dataset.num_classes, bias=False))

        self.weight_initialization()

    def weight_initialization(self):
        for m in self.modules():
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d) or isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0)

    def forward(self, batch_dict):
        points = batch_dict['points'][:, 1:]
        point_id_offset = batch_dict['point_id_offset'].int()
        if self.use_multi_sweeps:
            cur_point_indices = (points[:, 3] == 0)
            cur_points = points[cur_point_indices]
        else:
            cur_points = points
        point_per_features = self.point_encoder(cur_points)

        # encode voxel features
        point_voxel_ids = batch_dict['point_voxel_ids']
        if self.use_multi_sweeps:
            batch_dict['voxel_features'] = self.vfe(points, point_voxel_ids)
        else:
            batch_dict['voxel_features'] = self.vfe(point_per_features, point_voxel_ids)
        batch_dict = self.point_transformer(batch_dict)

        # point features from encoded voxel feature
        if self.use_multi_sweeps:
            point_voxel_features = voxel_to_point(batch_dict['voxel_features'], point_voxel_ids[cur_point_indices])
        else:
            point_voxel_features = voxel_to_point(batch_dict['voxel_features'], point_voxel_ids)
        point_fusion_features = torch.cat([point_per_features, point_voxel_features], dim=1)

        # decorating points with pixel-level semantic score
        if self.use_image_feature:
            point_image_features = batch_dict['point_image_features']
            point_image_features = self.deep_fusion(cur_points.contiguous(), point_id_offset,
                                                    point_fusion_features, point_image_features)
            point_fusion_features = torch.cat([point_fusion_features, point_image_features], dim=1)

        # fusion features encoder
        point_fusion_features = self.fusion_encoder(point_fusion_features)

        # se block for channel attention
        if self.use_multi_sweeps:
            point_batch_indices = batch_dict['points'][:, 0][cur_point_indices]
        else:
            point_batch_indices = batch_dict['points'][:, 0]
        point_fusion_features = point_fusion_features + self.se(point_fusion_features, point_batch_indices)

        result = OrderedDict()
        point_out = self.classifier(point_fusion_features)
        result['point_out'] = point_out

        result['voxel_out'] = batch_dict['voxel_out']
        result['aux_voxel_out'] = batch_dict['aux_voxel_out']

        result['voxel_coords'] = batch_dict['voxel_coords']
        result['aux_voxel_coords'] = batch_dict['aux_voxel_coords']

        return result
