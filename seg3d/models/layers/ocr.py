import torch
import torch.nn as nn
import torch.nn.functional as F

import spconv.pytorch as spconv

from seg3d.utils.spconv_utils import replace_feature


class SpatialGatherModule(nn.Module):
    """Aggregate the context features according to the initial predicted
    probability distribution.
    Employ the soft-weighted method to aggregate the context.
    """
    def __init__(self, scale):
        super(SpatialGatherModule, self).__init__()
        self.scale = scale

    def forward(self, feats, probs, batch_size, batch_indices):
        """Forward function."""
        ocr_context = []
        for i in range(batch_size):
            # [1, n, channels]
            feat = feats[batch_indices == i].unsqueeze(0)
            # [1, num_classes, n]
            prob = probs[batch_indices == i].unsqueeze(0)
            prob = prob.permute(0, 2, 1)
            prob = F.softmax(self.scale * prob, dim=2)
            # [1, num_classes, channels]
            context = torch.matmul(prob, feat).contiguous()
            ocr_context.append(context)
        # [batch_size, num_classes, channels]
        ocr_context = torch.cat(ocr_context)
        return ocr_context


class ObjectAttentionBlock(nn.Module):
    def __init__(self, in_channels, key_channels):
        super(ObjectAttentionBlock, self).__init__()

        self.key_channels = key_channels

        self.query_project = nn.Sequential(
            nn.Linear(in_channels, self.key_channels, bias=False),
            nn.BatchNorm1d(self.key_channels),
            nn.ReLU(inplace=True)
        )

        self.key_project = nn.Sequential(
            nn.Linear(in_channels, self.key_channels, bias=False),
            nn.BatchNorm1d(self.key_channels),
            nn.ReLU(inplace=True)
        )

        self.value_project = nn.Sequential(
            nn.Linear(in_channels, self.key_channels, bias=False),
            nn.BatchNorm1d(self.key_channels),
            nn.ReLU(inplace=True)
        )

        self.bottleneck = nn.Sequential(
            nn.Linear(self.key_channels, in_channels, bias=False),
            nn.BatchNorm1d(in_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x, proxy):
        query = self.query_project(x)
        key = self.key_project(proxy)
        key = key.permute(1, 0)
        value = self.value_project(proxy)

        sim_map = torch.matmul(query, key)
        sim_map = (self.key_channels ** -.5) * sim_map
        sim_map = F.softmax(sim_map, dim=-1)

        # add bg context
        context = torch.matmul(sim_map, value)
        context = self.bottleneck(context)
        return context


class OCRLayer(nn.Module):
    def __init__(self, in_channels, mid_channels, key_channels, scale=1., drop=0.05):
        super(OCRLayer, self).__init__()

        self.scale = scale
        self.transform_input = spconv.SparseSequential(
            spconv.SubMConv3d(in_channels, mid_channels, 3, padding=1, bias=False),
            nn.BatchNorm1d(mid_channels),
            nn.ReLU(inplace=True)
        )
        self.spatial_gather_module = SpatialGatherModule(self.scale)
        self.object_context_block = ObjectAttentionBlock(mid_channels, key_channels)
        self.bottleneck = nn.Sequential(
            nn.Linear(mid_channels * 2, in_channels, bias=False),
            nn.BatchNorm1d(in_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(drop)
        )

    def forward(self, inputs, probs, batch_size):
        inputs = self.transform_input(inputs)
        feats = inputs.features
        batch_indices = inputs.indices[:, 0]
        ocr_context = self.spatial_gather_module(feats, probs, batch_size, batch_indices)
        output_feats = torch.zeros(feats.shape).to(feats.device)
        for i in range(batch_size):
            feat = feats[batch_indices == i]
            proxy_feat = ocr_context[i]
            out_feat = self.object_context_block(feat, proxy_feat)
            output_feats[batch_indices == i] = out_feat
        feats = torch.cat([output_feats, feats], dim=1)
        feats = self.bottleneck(feats)
        inputs = replace_feature(inputs, feats)
        return inputs
