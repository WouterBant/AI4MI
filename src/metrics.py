
from typing import Callable, Iterable, List, Set, Tuple, TypeVar, cast
import torch
from torch import Tensor
from utils import dice_coef

def update_metrics(K: int, total_pred_seg: Tensor, total_gt_seg: Tensor):
    
    dice_metric = dice_coef(total_pred_seg, total_gt_seg) # the dice metric
    
    # Hausdorff metric #TODO fix hausdorff metric nu nog fout
    #hausdorff_metric = Hausdorff_metric_fast(total_pred_seg, total_gt_seg)
    
    # Sensitivity metric and also specificity
    sensitivity, specificity = Sensitivity_Specifity_metrics(total_pred_seg, total_gt_seg)
    
    
    
#TODO: Check for the correct implementation of the Hausdorff metric maybe also use the scikit-image implementation. 

def Hausdorff_metric_fast(pred_seg: Tensor, gt_seg: Tensor):
    n_samples, n_classes, height, width = pred_seg.shape
    
    # Reshape the predictions and ground truts
    pred_flat = pred_seg.view(n_samples, n_classes, -1)
    gt_flat = gt_seg.view(n_samples, n_classes, -1)
    
    # Compute the distance between al points in each class
    dist_matrix = torch.cdist(pred_flat, gt_flat, p=2)
    
    # Calculate the minimum distances
    min_pred_to_gt = torch.min(dist_matrix, dim=3)[0]
    min_gt_to_pred = torch.min(dist_matrix, dim=2)[0]
    
    # Take the maximum of the minimum distances for each sample and class
    max_min_pred_to_gt = torch.max(min_pred_to_gt, dim=2)[0]
    max_min_gt_to_pred = torch.max(min_gt_to_pred, dim=2)[0]
    
    # Hausdorff distance is the maximum of these two values
    hausdorff_distance = torch.max(max_min_pred_to_gt, max_min_gt_to_pred)
    
    return hausdorff_distance

def Sensitivity_Specifity_metrics(pred_seg: Tensor, gt_seg: Tensor):
    n_samples, n_classes, height, width = pred_seg.shape
    
    # True positives
    true_positives = (pred_seg * gt_seg).sum(dim=(2, 3)) # Only sum over the height and width (pixel dimensions)
    
    # True Negatives
    true_negatives = ((1 - pred_seg) * (1 - gt_seg)).sum(dim=(2, 3))
    
    # False Positives
    false_positives = (pred_seg * (1 - gt_seg)).sum(dim=(2, 3))
    
    # False Negatives
    false_negatives = ((1 - pred_seg) * gt_seg).sum(dim=(2, 3))
    
    # Sensitivity = TP / (TP + FN)
    sensitivity = true_positives / (true_positives + false_negatives + 1e-10) # Add a small value to avoid division by zero
    
    # Specificity = TN / (TN + FP)
    specificity = true_negatives / (true_negatives + false_positives + 1e-10) # Add a small value to avoid division by zero
    
    return sensitivity, specificity
    