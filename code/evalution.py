import numpy as np
import torch
import torch.nn.functional as F
from tensorflow.keras.metrics import MeanIoU

def calc_semantic_segmantation_confusion(pred_labels, gt_labels):
    pred_labels = iter(pred_labels)
    gt_labels = iter(gt_labels)

    n_class = 8

    confusion = np.zeros((n_class, n_class), dtype=np.int64)
    for pred_label, gt_label in zip(pred_labels, gt_labels):
        if pred_label.ndim != 2 or gt_label.ndim != 2:
            raise ValueError('ndim of labels should be two')
        if pred_label.shape != gt_label.shape:
            raise ValueError('Shape of ground truth and prediction should be same')
        pred_label = pred_label.flatten()
        gt_label = gt_label.flatten()

        lb_max = np.max((pred_label, gt_label))
        if lb_max >= n_class:
            expanded_confusion = np.zeros(
                (lb_max + 1, lb_max + 1), dtype=np.int64
            )
            expanded_confusion[0:n_class, 0:n_class] = confusion

            n_class = lb_max + 1
            confusion = expanded_confusion

        mask = gt_label >= 0
        confusion += np.bincount(
            n_class * gt_label[mask].astype(int) + pred_label[mask],
            minlength=n_class**2
        ).reshape((n_class, n_class))

    for iter_ in (pred_labels, gt_labels):
        if next(iter_, None) is not None:
            raise ValueError('Length of input iterables need to be same')
    return confusion


def calc_semantic_segmantation_iou(confusion):
    iou_denominator = (confusion.sum(axis=1) + confusion.sum(axis=0) - np.diag(confusion))
    iou = np.diag(confusion) / iou_denominator
    return iou[:-1]


def eval_semantic_segmentation(pred_labels, gt_labels):
    confusion = calc_semantic_segmantation_confusion(pred_labels, gt_labels)
    iou = calc_semantic_segmantation_iou(confusion)
    pixel_accuracy = np.diag(confusion).sum() / confusion.sum()
    class_accuracy = np.diag(confusion) / (np.sum(confusion, axis=1) + 1e-10)

    return {'iou': iou,
            'confusion': confusion,
            'miou': np.nanmean(iou),
            'pixel_accuracy': pixel_accuracy,
            'class_accuracy': class_accuracy,
            'mean_class_accuracy': np.nanmean(class_accuracy[:-1])
            }