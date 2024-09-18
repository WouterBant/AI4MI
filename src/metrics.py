
from typing import Callable, Iterable, List, Set, Tuple, TypeVar, cast
import torch
from torch import Tensor
from utils import dice_coef
from scipy.spatial.distance import directed_hausdorff
import numpy as np

def update_metrics(K: int, total_pred_seg: Tensor, total_gt_seg: Tensor):
    
    dice_metric = dice_coef(total_pred_seg, total_gt_seg) # the dice metric
    
    # Hausdorff metric
    #hausdorff_metric = total_hausdorff_distance(total_pred_seg, total_gt_seg)
    
    # Sensitivity metric and also specificity
    sensitivity, specificity = Sensitivity_Specifity_metrics(total_pred_seg, total_gt_seg)
    
    
    
#TODO: Check for the correct implementation of the Hausdorff metric maybe also use the scikit-image implementation. 
def total_hausdorff_distance(ground_truth_tensor, prediction_tensor):
    hausdorf_metrics = torch.zeros((ground_truth_tensor.shape[0], ground_truth_tensor.shape[1]))
    for class_idx in range(ground_truth_tensor.shape[1]):
        for sample_idx in range(ground_truth_tensor.shape[0]):
            ground_truth_img = ground_truth_tensor[sample_idx, class_idx]
            prediction_img = prediction_tensor[sample_idx, class_idx]
            hausdorf_metrics[sample_idx, class_idx] = calculate_hausdorff_distance(ground_truth_img, prediction_img)

def calculate_hausdorff_distance(ground_truth_tensor, prediction_tensor):
    # Convert PyTorch tensors to NumPy arrays and move them to the CPU
    ground_truth_np = ground_truth_tensor.cpu().numpy()
    prediction_np = prediction_tensor.cpu().numpy()
    
    # Extract coordinates of the segmented regions (points with value 1)
    ground_truth_points = np.argwhere(ground_truth_np == 1)
    prediction_points = np.argwhere(prediction_np == 1)
    
    # Calculate directed Hausdorff distance in both directions
    forward_hausdorff = directed_hausdorff(ground_truth_points, prediction_points)[0]
    backward_hausdorff = directed_hausdorff(prediction_points, ground_truth_points)[0]
    
    # Return the maximum of the two directed distances (Hausdorff distance)
    hausdorff_distance = max(forward_hausdorff, backward_hausdorff)
    
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
    