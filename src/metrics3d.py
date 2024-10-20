import torch
from torch import Tensor
from utils import our_dice_batch
import pandas as pd
from medpy.metric.binary import hd


def update_metrics_3D(
    metrics, pred: Tensor, gt: Tensor, patient_id: str, classes, metric_types
) -> dict:
    metric_functions = {
        "dice": ("Dice", lambda p, g: our_dice_batch(p, g)),
        "sensitivity": (
            "Sensitivity",
            lambda p, g: Sensitivity_Specifity_metrics(p, g)[0],
        ),
        "specificity": (
            "Specificity",
            lambda p, g: Sensitivity_Specifity_metrics(p, g)[1],
        ),
        "hausdorff": ("Hausdorff", hausdorf3d),
        "iou": ("IoU", jaccard_index),
        "precision": ("Precision", precision_metric),
        "volumetric": ("Volumetric", volumetric_similarity),
        "VOE": ("VOE", lambda p, g: 1 - jaccard_index(p, g)),
    }

    B, K, W, H = pred.shape  # B is the number of files for this patient

    # Now all metrics can just reduct last 3 dimensions
    pred = pred.permute(1, 0, 2, 3)
    gt = gt.permute(1, 0, 2, 3)

    for metric_type in metric_types:
        if metric_type in metric_functions:
            metric_name, func = metric_functions[metric_type]
            results = func(pred, gt)
            assert results.shape == (K,), f"shape is {results.shape}"
            metrics = append_3d_metrics(
                metrics, patient_id, classes, metric_name, results
            )

    return metrics


def hausdorf3d(pred: Tensor, gt: Tensor):
    res = torch.zeros(pred.size(0))
    pred = pred.numpy()
    gt = gt.numpy()
    for k in range(len(pred)):
        i = pred[k]
        g = gt[k]
        res[k] = hd(i, g)
    return res


def append_3d_metrics(old_metrics, patient_id, classes, metric_name, results):
    # First transform the results tensor to a numpy array
    results = results.cpu().numpy()

    data = []
    for i, class_name in enumerate(classes):
        data.append(
            {
                "patient_id": patient_id,
                "class": class_name,
                "metric_type": metric_name,
                "metric_value": results[i],
            }
        )

    new_metrics = pd.concat([old_metrics, pd.DataFrame(data)], ignore_index=True)

    return new_metrics


def Sensitivity_Specifity_metrics(pred_seg: Tensor, gt_seg: Tensor):
    # True positives
    true_positives = (pred_seg * gt_seg).sum(
        dim=(1, 2, 3)
    )  # Only sum over the height and width (pixel dimensions)

    # True Negatives
    true_negatives = ((1 - pred_seg) * (1 - gt_seg)).sum(dim=(1, 2, 3))

    # False Positives
    false_positives = (pred_seg * (1 - gt_seg)).sum(dim=(1, 2, 3))

    # False Negatives
    false_negatives = ((1 - pred_seg) * gt_seg).sum(dim=(1, 2, 3))

    # Sensitivity = TP / (TP + FN)
    sensitivity = true_positives / (
        true_positives + false_negatives + 1e-10
    )  # Add a small value to avoid division by zero

    # Specificity = TN / (TN + FP)
    specificity = true_negatives / (
        true_negatives + false_positives + 1e-10
    )  # Add a small value to avoid division by zero

    return sensitivity, specificity


def jaccard_index(pred, gt):
    # Compute intersection and union over the last two dimensions
    intersection = torch.logical_and(gt == 1, pred == 1).sum(dim=(-3, -2, -1)).float()
    union = torch.logical_or(gt == 1, pred == 1).sum(dim=(-3, -2, -1)).float()

    # Handle division by zero (no positives in ground truth and prediction)
    iou = (intersection + 1e-8) / (
        union + 1e-8
    )  # Adding small epsilon to prevent division by zero
    return iou


def precision_metric(pred, gt):
    # Compute true positives and predicted positives over the last two dimensions
    true_positive = torch.logical_and(gt == 1, pred == 1).sum(dim=(-3, -2, -1)).float()
    predicted_positive = (pred == 1).sum(dim=(-3, -2, -1)).float()

    # Handle division by zero (no positives predicted)
    precision = (true_positive + 1e-8) / (
        predicted_positive + 1e-8
    )  # Adding small epsilon to prevent division by zero
    return precision


def volumetric_similarity(pred, gt):
    # Calculate the volumes (number of foreground pixels)
    vol_gt = (gt == 1).sum(dim=(-3, -2, -1)).float()
    vol_pred = (pred == 1).sum(dim=(-3, -2, -1)).float()

    # Compute the absolute difference and the total volume
    abs_diff = torch.abs(vol_gt - vol_pred)
    total_vol = vol_gt + vol_pred

    # Return Volumetric Similarity (1 - relative volume difference)
    vs = 1 - abs_diff / (total_vol + 1e-6)  # Adding epsilon to avoid division by zero
    return vs
