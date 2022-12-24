import numpy as np
import torch
import torch.distributed as dist


class IOUMetric(object):
    """IOU Metric.

    Evaluate the result of the Semantic Segmentation.

    Args:
        class_names (List[str]): class names
    """

    def __init__(self, class_names):
        self.class_names = class_names

        self.hist_list = []

    @staticmethod
    def fast_hist(preds, labels, num_classes):
        """Compute the confusion matrix for every batch.
        Args:
            preds (np.ndarray):  Prediction labels of points with shape of
            (num_points, ).
            labels (np.ndarray): Ground truth labels of points with shape of
            (num_points, ).
            num_classes (int): number of classes
        Returns:
            np.ndarray: Calculated confusion matrix.
        """

        k = (labels >= 0) & (labels < num_classes)
        bin_count = np.bincount(
            num_classes * labels[k].astype(int) + preds[k],
            minlength=num_classes ** 2)
        return bin_count[:num_classes ** 2].reshape(num_classes, num_classes)

    @staticmethod
    def per_class_iou(hist):
        """Compute the per class iou.
        Args:
            hist(np.ndarray):  Overall confusion martix
            (num_classes, num_classes ).
        Returns:
            np.ndarray: Calculated per class iou
        """

        return np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))

    def add(self, pred_labels, gt_labels):
        preds = pred_labels.clone().numpy().astype(np.int)
        labels = gt_labels.clone().numpy().astype(np.int)

        # calculate one instance result
        hist = self.fast_hist(preds, labels, len(self.class_names))
        self.hist_list.append(hist)

    @staticmethod
    def reduce_tensor(tensor):
        rt = tensor.clone()
        dist.all_reduce(rt, op=dist.ReduceOp.SUM)
        return rt

    def get_metric(self):
        hist = sum(self.hist_list)
        try:
            dist.barrier()
            hist = torch.from_numpy(hist).cuda()
            hist = self.reduce_tensor(hist).cpu().numpy()
            iou = self.per_class_iou(hist)
        except:
            iou = self.per_class_iou(hist)

        # mean iou
        metric = dict()
        miou = np.nanmean(iou)
        metric['mIOU'] = miou

        # iou per class
        iou_dict = dict()
        for i in range(len(self.class_names)):
            iou_dict[self.class_names[i]] = float(iou[i])
        metric['IOU'] = iou_dict
        return metric


if __name__ == '__main__':
    class_names = ['c0', 'c1', 'c2', 'c3']
    iou_metric = IOUMetric(class_names)

    pred_labels = torch.Tensor([1, 2, 3])
    gt_labels = torch.Tensor([1, 1, 3])
    iou_metric.add(pred_labels, gt_labels)

    pred_labels = torch.Tensor([0, 2, 3])
    gt_labels = torch.Tensor([1, 3, 3])
    iou_metric.add(pred_labels, gt_labels)

    print(iou_metric.get_metric())
