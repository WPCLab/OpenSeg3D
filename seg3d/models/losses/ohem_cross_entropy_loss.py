import torch
import torch.nn as nn


class OHEMCrossEntropyLoss(nn.Module):
    def __init__(self,
                 keep_ratio=None,
                 ignore_index=255,
                 class_weight=None,
                 loss_name='loss_ohem_cross_entropy'):
        super(OHEMCrossEntropyLoss, self).__init__()

        self.keep_ratio = keep_ratio
        self.ignore_index = ignore_index
        self.class_weight = class_weight
        self.cross_entropy = nn.CrossEntropyLoss(reduction='none',
                                                 weight=self.class_weight,
                                                 ignore_index=self.ignore_index)
        self._loss_name = loss_name

    def forward(self,
                inputs,
                targets):
        losses = self.cross_entropy(inputs, targets)
        if self.keep_ratio:
            _, sort_indices = losses.sort(descending=True)
            kept_count = int(losses.shape[0] * self.keep_ratio)
            losses = losses[sort_indices[:kept_count]]
        loss = losses.mean()
        return loss

    @property
    def loss_name(self):
        """Loss Name.
        This function must be implemented and will return the name of this
        loss function. This name will be used to combine different loss items
        by simple sum operation. In addition, if you want this loss item to be
        included into the backward graph, `loss_` must be the prefix of the
        name.
        Returns:
            str: The name of this loss item.
        """
        return self._loss_name


if __name__ == '__main__':
    ohem_ce_loss = OHEMCrossEntropyLoss(keep_ratio=0.3)
    loss = ohem_ce_loss(torch.rand(15, 2), torch.ones(15).long())
    print(loss)
