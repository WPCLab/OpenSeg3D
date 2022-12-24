import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from seg3d.utils.swformer_utils import get_flat2win_inds_v2, flat2window_v2, get_window_coors, window2flat_v2
from seg3d.ops import get_inner_win_inds
from .cosine_msa import CosineMultiheadAttention
from .drop import DropPath


class SparseWindowPartitionLayer(nn.Module):
    """
    There are 3 things to be done in this class:
    1. Regional Grouping: assign window indices to each voxel.
    2. Voxel batching and region batching: see our paper for detail
    3. Pre-computing the transformation information for converting flat features ([N x C]) to region features ([R, T, C]).
        R is the number of regions containing at most T tokens (voxels). See function flat2window and window2flat for details.

    Main args:
        batching_info (dict): batching configuration for region batching.
        window_shape (tuple[int]): (num_x, num_y， num_z). Each window is divided to num_x * num_y * num_z voxels (including empty voxel).
    """
    def __init__(self,
                 batching_info,
                 window_shape,
                 sparse_shape,
                 normalize_pos=False,
                 pos_temperature=1000):
        super(SparseWindowPartitionLayer, self).__init__()
        self.batching_info = batching_info
        self.sparse_shape = sparse_shape
        self.window_shape = window_shape
        self.normalize_pos = normalize_pos
        self.pos_temperature = pos_temperature

    def forward(self, x):
        """
        Args:
            x: input sparse tensor contains:
                voxel_features: shape=[N, C], N is the voxel num in the batch.
                voxel_coords: shape=[N, 4], [b, z, y, x]
        Returns:
            feat_3d_dict: contains region features (feat_3d) of each region batching level. Shape of feat_3d is [num_windows, num_max_tokens, C].
            flat2win_inds_list: two dict containing transformation information for non-shifted grouping and shifted grouping, respectively. The two dicts are used in function flat2window and window2flat.
            voxel_info: dict containing extra information of each voxel for usage in the backbone.
        """
        voxel_features = x.features
        voxel_coords = x.indices.long()

        voxel_info = self.window_partition(voxel_coords)
        voxel_info['voxel_features'] = voxel_features
        voxel_info['voxel_coords'] = voxel_coords
        voxel_info = self.batching_voxel(voxel_info, 2)  # voxel_info is updated in this function

        voxel_features = voxel_info['voxel_features']  # after batching

        for i in range(2):
            voxel_info[f'flat2win_inds_shift{i}'] = \
                get_flat2win_inds_v2(voxel_info[f'batch_win_inds_shift{i}'], voxel_info[f'voxel_batching_level_shift{i}'],
                                     self.batching_info)

            voxel_info[f'pos_dict_shift{i}'] = \
                self.get_pos_embed(voxel_info[f'flat2win_inds_shift{i}'], voxel_info[f'coors_in_win_shift{i}'],
                                   voxel_features.size(1), voxel_features.dtype)

            voxel_info[f'key_mask_shift{i}'] = \
                self.get_key_padding_mask(voxel_info[f'flat2win_inds_shift{i}'])

        return voxel_info

    def batching_single_shift(self, batch_win_inds):
        batching_info = self.batching_info
        batching_lvl_per_voxel = -torch.ones_like(batch_win_inds)
        inner_win_inds = get_inner_win_inds(batch_win_inds)
        bincount = torch.bincount(batch_win_inds)
        num_per_voxel_before_batching = bincount[batch_win_inds]
        target_num_per_voxel = torch.zeros_like(batch_win_inds)

        for bl in batching_info:
            max_tokens = batching_info[bl]['max_tokens']
            lower, upper = batching_info[bl]['batching_range']
            range_mask = (num_per_voxel_before_batching >= lower) & (num_per_voxel_before_batching < upper)
            target_num_per_voxel[range_mask] = max_tokens
            batching_lvl_per_voxel[range_mask] = bl

        keep_mask = inner_win_inds < target_num_per_voxel
        return keep_mask, batching_lvl_per_voxel

    def batching_voxel(self, voxel_info, num_shifts):
        """
        To make it clear and easy to follow, we do not use loop to process two shifts.
        """
        batch_win_inds_s0 = voxel_info['batch_win_inds_shift0']
        num_all_voxel = batch_win_inds_s0.shape[0]

        voxel_keep_inds = torch.arange(num_all_voxel, device=batch_win_inds_s0.device, dtype=torch.long)

        keep_mask_s0, batching_lvl_s0 = self.batching_single_shift(batch_win_inds_s0)

        batching_lvl_s0 = batching_lvl_s0[keep_mask_s0]
        voxel_keep_inds = voxel_keep_inds[keep_mask_s0]
        batch_win_inds_s0 = batch_win_inds_s0[keep_mask_s0]

        if num_shifts == 1:
            voxel_info['voxel_keep_inds'] = voxel_keep_inds
            voxel_info['voxel_batching_level_shift0'] = batching_lvl_s0
            voxel_info['batch_win_inds_shift0'] = batch_win_inds_s0
            return voxel_info

        batch_win_inds_s1 = voxel_info['batch_win_inds_shift1']
        batch_win_inds_s1 = batch_win_inds_s1[keep_mask_s0]

        keep_mask_s1, batching_lvl_s1 = self.batching_single_shift(batch_win_inds_s1)

        # batching data in first shift again
        batching_lvl_s0 = batching_lvl_s0[keep_mask_s1]
        voxel_keep_inds = voxel_keep_inds[keep_mask_s1]
        batch_win_inds_s0 = batch_win_inds_s0[keep_mask_s1]

        batching_lvl_s1 = batching_lvl_s1[keep_mask_s1]
        batch_win_inds_s1 = batch_win_inds_s1[keep_mask_s1]

        voxel_info['voxel_keep_inds'] = voxel_keep_inds
        voxel_info['voxel_batching_level_shift0'] = batching_lvl_s0
        voxel_info['batch_win_inds_shift0'] = batch_win_inds_s0
        voxel_info['voxel_batching_level_shift1'] = batching_lvl_s1
        voxel_info['batch_win_inds_shift1'] = batch_win_inds_s1
        voxel_keep_inds = voxel_info['voxel_keep_inds']

        voxel_num_before_batching = len(voxel_info['voxel_coords'])
        voxel_info['voxel_features'] = voxel_info['voxel_features'][voxel_keep_inds]
        voxel_info['voxel_coords'] = voxel_info['voxel_coords'][voxel_keep_inds]

        # Some other variables need to be batched.
        for k, v in voxel_info.items():
            if isinstance(v, torch.Tensor) and len(v) == voxel_num_before_batching:
                voxel_info[k] = v[voxel_keep_inds]

        return voxel_info

    @torch.no_grad()
    def window_partition(self, coors):
        voxel_info = {}
        for i in range(2):
            batch_win_inds, coors_in_win = get_window_coors(coors, self.sparse_shape, self.window_shape, i == 1)
            voxel_info[f'batch_win_inds_shift{i}'] = batch_win_inds
            voxel_info[f'coors_in_win_shift{i}'] = coors_in_win

        return voxel_info

    @torch.no_grad()
    def get_pos_embed(self, inds_dict, coors_in_win, feat_dim, dtype):
        """
        Args:
            coors_in_win: shape=[N, 3], order: z, y, x
        """
        # [N,]
        window_shape = self.window_shape
        if len(window_shape) == 2:
            ndim = 2
            win_x, win_y = window_shape
            win_z = 0
        elif window_shape[-1] == 1:
            ndim = 2
            win_x, win_y = window_shape[:2]
            win_z = 0
        else:
            win_x, win_y, win_z = window_shape
            ndim = 3

        assert coors_in_win.size(1) == 3
        z, y, x = coors_in_win[:, 0] - win_z / 2, coors_in_win[:, 1] - win_y / 2, coors_in_win[:, 2] - win_x / 2
        assert (x >= -win_x / 2 - 1e-4).all()
        assert (x <= win_x / 2 - 1 + 1e-4).all()

        if self.normalize_pos:
            x = x / win_x * 2 * 3.1415  # [-pi, pi]
            y = y / win_y * 2 * 3.1415  # [-pi, pi]
            z = z / win_z * 2 * 3.1415  # [-pi, pi]

        pos_length = feat_dim // ndim
        # [pos_length]
        inv_freq = torch.arange(
            pos_length, dtype=torch.float32, device=coors_in_win.device)
        inv_freq = self.pos_temperature ** (2 * torch.div(inv_freq, 2, rounding_mode='floor') / pos_length)

        # [num_tokens, pos_length]
        embed_x = x[:, None] / inv_freq[None, :]
        embed_y = y[:, None] / inv_freq[None, :]
        if ndim == 3:
            embed_z = z[:, None] / inv_freq[None, :]

        # [num_tokens, pos_length]
        embed_x = torch.stack([embed_x[:, ::2].sin(), embed_x[:, 1::2].cos()], dim=-1).flatten(1)
        embed_y = torch.stack([embed_y[:, ::2].sin(), embed_y[:, 1::2].cos()], dim=-1).flatten(1)
        if ndim == 3:
            embed_z = torch.stack([embed_z[:, ::2].sin(), embed_z[:, 1::2].cos()], dim=-1).flatten(1)

        # [num_tokens, c]
        if ndim == 3:
            pos_embed_2d = torch.cat([embed_x, embed_y, embed_z], dim=-1).to(dtype)
        else:
            pos_embed_2d = torch.cat([embed_x, embed_y], dim=-1).to(dtype)

        pos_embed_dict = flat2window_v2(pos_embed_2d, inds_dict)

        return pos_embed_dict

    @torch.no_grad()
    def get_key_padding_mask(self, ind_dict):
        num_all_voxel = len(ind_dict['voxel_batching_level'])
        key_padding = torch.ones((num_all_voxel, 1)).to(ind_dict['voxel_batching_level'].device).bool()

        window_key_padding_dict = flat2window_v2(key_padding, ind_dict)

        # logical not. True means masked
        for key, value in window_key_padding_dict.items():
            window_key_padding_dict[key] = value.logical_not().squeeze(2)

        return window_key_padding_dict

class WindowAttention(nn.Module):
    def __init__(self, d_model, nhead, attn_drop, cosine=True, tau_min=0.01):
        super(WindowAttention, self).__init__()

        if cosine:
            self.self_attn = CosineMultiheadAttention(
                d_model, nhead, dropout=attn_drop, batch_first=False, tau_min=tau_min,
                cosine=True, non_shared_tau=False)
        else:
            self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=attn_drop)

    def forward(self, feat_2d, pos_dict, ind_dict, key_padding_dict):
        out_feat_dict = {}
        feat_3d_dict = flat2window_v2(feat_2d, ind_dict)
        for name in feat_3d_dict:
            #  [n, num_token, embed_dim]
            pos = pos_dict[name]

            feat_3d = feat_3d_dict[name]
            feat_3d = feat_3d.permute(1, 0, 2)

            v = feat_3d

            if pos is not None:
                pos = pos.permute(1, 0, 2)
                assert pos.shape == feat_3d.shape, f'pos_shape: {pos.shape}, feat_shape:{feat_3d.shape}'
                q = k = feat_3d + pos
            else:
                q = k = feat_3d

            key_padding_mask = key_padding_dict[name]
            out_feat_3d, attn_map = self.self_attn(q, k, value=v, key_padding_mask=key_padding_mask)
            out_feat_dict[name] = out_feat_3d.permute(1, 0, 2)

        results = window2flat_v2(out_feat_dict, ind_dict)

        return results

class MLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0.):
        super(MLP, self).__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class EncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, mlp_hidden_dim=256, drop=0., attn_drop=0.1, drop_path=0.):
        super(EncoderLayer, self).__init__()
        self.win_attn = WindowAttention(d_model, nhead, attn_drop)
        # Implementation of Feedforward model
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.mlp = MLP(in_features=d_model, hidden_features=mlp_hidden_dim, drop=drop)

    def forward(self, x, pos_dict, ind_dict, key_padding_mask_dict):
        shortcut = x

        x = self.win_attn(x, pos_dict, ind_dict, key_padding_mask_dict) #[N, d_model]

        x = shortcut + self.drop_path(self.norm1(x))

        # FFN
        x = x + self.drop_path(self.norm2(self.mlp(x)))

        return x

class SWFormerBlock(nn.Module):
    def __init__(self, d_model, nhead, depth=4, mlp_ratio=2., attn_drop=0.1, drop=0., drop_path=0.):
        super(SWFormerBlock, self).__init__()

        mlp_hidden_dim = int(d_model * mlp_ratio)

        self.depth = depth
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, nhead, mlp_hidden_dim,
                         attn_drop=attn_drop,
                         drop=drop,
                         drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path)
            for i in range(depth)])

    def forward(self, batch_dict, using_checkpoint=True):
        voxel_features = batch_dict['voxel_features']

        # window shift 0
        ind_dict = batch_dict['flat2win_inds_shift0']
        pos_dict = batch_dict['pos_dict_shift0']
        key_mask_dict = batch_dict['key_mask_shift0']
        if using_checkpoint and self.training:
            for layer in self.layers[:int(self.depth/2)]:
                voxel_features = checkpoint(layer, voxel_features, pos_dict, ind_dict, key_mask_dict)
        else:
            for layer in self.layers[:int(self.depth/2)]:
                voxel_features = layer(voxel_features, pos_dict, ind_dict, key_mask_dict)

        # window shift 1
        ind_dict = batch_dict['flat2win_inds_shift1']
        pos_dict = batch_dict['pos_dict_shift1']
        key_mask_dict = batch_dict['key_mask_shift1']
        if using_checkpoint and self.training:
            for layer in self.layers[int(self.depth/2):]:
                voxel_features = checkpoint(layer, voxel_features, pos_dict, ind_dict, key_mask_dict)
        else:
            for layer in self.layers[int(self.depth/2):]:
                voxel_features = layer(voxel_features, pos_dict, ind_dict, key_mask_dict)

        return voxel_features
