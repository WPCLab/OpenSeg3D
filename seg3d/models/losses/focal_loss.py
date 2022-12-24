import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    def __init__(self,
                 gamma=2.0,
                 alpha=0.5,
                 num_classes=-1,
                 ignore_index=255,
                 class_weight=None,
                 reduction='mean',
                 loss_name='loss_focal'):
        """`Focal Loss <https://arxiv.org/abs/1708.02002>`_
        Args:
            gamma (float, optional): The gamma for calculating the modulating
                factor. Defaults to 2.0.
            alpha (float | list[float], optional): A balanced form for Focal
                Loss. Defaults to 0.5.
            num_classes (int, optional): Number of classes
            ignore_index (int, optional): The label index to be ignored.
                Default: 255
            class_weight (list[float], optional): Weight of each class.
                Defaults to None.
            reduction (str, optional): The method used to reduce the loss into
                a scalar. Defaults to 'mean'. Options are "none", "mean" and
                "sum".
            loss_name (str, optional): Name of the loss item. If you want this
                loss item to be included into the backward graph, `loss_` must
                be the prefix of the name. Defaults to 'loss_focal'.
        """
        super(FocalLoss, self).__init__()
        assert reduction in ('none', 'mean', 'sum'), \
            "AssertionError: reduction should be 'none', 'mean' or " \
            "'sum'"
        assert isinstance(alpha, (float, list)), \
            'AssertionError: alpha should be of type float'
        assert isinstance(gamma, float), \
            'AssertionError: gamma should be of type float'
        assert isinstance(loss_name, str), \
            'AssertionError: loss_name should be of type str'
        self.gamma = gamma
        self.alpha = alpha
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.class_weight = class_weight
        self.reduction = reduction
        self._loss_name = loss_name

    def forward(self,
                inputs,
                targets):
        """Forward function.
        Args:
            inputs (torch.Tensor): The prediction with shape
                (N, C) where C = number of classes, or
                (N, C, d_1, d_2, ..., d_K) with K≥1 in the
                case of K-dimensional loss.
            targets (torch.Tensor): The ground truth. If containing class
                indices, shape (N) where each value is 0≤targets[i]≤C−1,
                or (N, d_1, d_2, ..., d_K) with K≥1 in the case of
                K-dimensional loss. If containing class probabilities,
                same shape as the input.
        Returns:
            torch.Tensor: The calculated loss
        """
        final_weight = torch.ones(1, inputs.size(1)).type_as(inputs)
        if self.class_weight is not None:
            final_weight = final_weight * inputs.new_tensor(self.class_weight)

        valid_mask = (targets != self.ignore_index)
        inputs = inputs[valid_mask]
        targets = targets[valid_mask]

        p = torch.sigmoid(inputs)
        targets = F.one_hot(targets, self.num_classes).float()
        ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
        p_t = p * targets + (1 - p) * (1 - targets)
        loss = ce_loss * ((1 - p_t) ** self.gamma)

        if self.alpha >= 0:
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            loss = alpha_t * loss

        loss = loss * final_weight
        if self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()

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
