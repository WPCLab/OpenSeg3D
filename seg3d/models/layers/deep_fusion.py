import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from seg3d.ops import knn_query


class DeepFusionBlock(nn.Module):
    def __init__(self, lidar_channel, image_channel, hidden_channel, n_neighbors, attn_pdrop=0.3):
        super(DeepFusionBlock, self).__init__()

        self.lidar_channel = lidar_channel
        self.image_channel = image_channel
        self.n_neighbors = n_neighbors

        self.q_embedding = nn.Linear(lidar_channel, hidden_channel)
        self.k_embedding = nn.Linear(image_channel, hidden_channel)
        self.v_embedding = nn.Linear(image_channel, hidden_channel)

        self.attn_dropout = nn.Dropout(attn_pdrop)

        self.c_proj = nn.Linear(hidden_channel, image_channel)

    def forward(self, points, point_id_offset, lidar_features, image_features):
        q = self.q_embedding(lidar_features)
        k = self.k_embedding(image_features)
        v = self.v_embedding(image_features)

        knn_ids, _ = knn_query(self.n_neighbors, points, points, point_id_offset, point_id_offset)
        k = k[knn_ids.long()]
        attn_weights = torch.einsum('nc,nkc->nk', q, k) / np.sqrt(q.shape[-1])

        invalid_mask = (torch.sum(image_features, dim=1) == 0)
        invalid_mask = invalid_mask[knn_ids.long()]
        attn_weights[invalid_mask] = float('-inf')
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = torch.nan_to_num(attn_weights)
        attn_weights = self.attn_dropout(attn_weights)

        v = v[knn_ids.long()]
        out = torch.einsum('nk,nkc->nc', attn_weights, v)
        out = self.c_proj(out)
        return out
