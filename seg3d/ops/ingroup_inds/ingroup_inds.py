import torch
from torch.autograd import Function

from . import ingroup_inds_ext


class IngroupIndicesFunction(Function):
    @staticmethod
    def forward(ctx, group_inds):
        out_inds = torch.zeros_like(group_inds) - 1
        ingroup_inds_ext.forward(group_inds, out_inds)
        ctx.mark_non_differentiable(out_inds)
        return out_inds

    @staticmethod
    def backward(ctx, g):
        return None


get_inner_win_inds = IngroupIndicesFunction.apply
