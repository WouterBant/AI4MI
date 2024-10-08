
from typing import Callable, Iterable, List, Set, Tuple, TypeVar, cast
import torch
from torch import Tensor
from utils import dice_coef
from scipy.spatial.distance import directed_hausdorff
from scipy.ndimage import distance_transform_edt
import numpy as np
import pandas as pd
import os
import time

def print_store_metrics(metrics, destination):
    classes = ["Background", "Esophagus", "Heart", "Trachea", "Aorta"]

    # Create an empty dictionary to store the results
    results = {}

    # Loop through each metric and calculate the average and std
    for metric in metrics.keys():
        avg_metric = torch.mean(metrics[metric], dim=0)
        std_metric = torch.std(metrics[metric], dim=0)

        # Format the values as "average (std)" with 4 decimal places
        formatted_metrics = [f"{avg:.4f} ({std:.4f})" for avg, std in zip(avg_metric, std_metric)]
        
        # Store the formatted metrics in the results dictionary
        results[metric] = formatted_metrics

    # Create a DataFrame with the classes as columns and metrics as the index
    df = pd.DataFrame(results, index=classes).T
    
    print("\n Metric results:\n")
    print(df)

    print("\n Latex code for the table:\n")
    print(df.to_latex())
    
    # Save the DataFrame to a CSV file
    df.to_csv(str(destination) + "/results.csv")

def update_metrics_2D(pred: Tensor, gt: Tensor, metric_type: str) -> dict:
    
    if metric_type not in ["dice", "sensitivity", "specificity", "hausdorff", "iou", "precision", "volumetric", "VOE"]:
        raise ValueError(f"Unsupported metric type: {metric_type}")

    if metric_type == "dice":
        return dice_coef(pred, gt)
    
    if metric_type == "sensitivity":
        return Sensitivity_Specifity_metrics(pred, gt)[0]
    
    if metric_type == "specificity":
        return Sensitivity_Specifity_metrics(pred, gt)[1]
    
    # Would not recommend using this metric as it is very slow
    if metric_type == "hausdorff":
        return total_hausdorff_distance(pred, gt)
    
    if metric_type == "iou":
        return jaccard_index(pred, gt)
    
    if metric_type == "precision":
        return precision_metric(pred, gt)
    
    if metric_type == "volumetric":
        return volumetric_similarity(pred, gt)
    
    if metric_type == "VOE": # Volume Overlap Error
        return 1 - jaccard_index(pred, gt)
    
    #TODO maybe add average Averagey Symmetric Surface Distance (ASSD) however also slow and already have Hausdorff distance
    
    
#TODO: Check for the correct implementation of the Hausdorff metric maybe also use the scikit-image implementation. 
#TODO: Make more efficient using: https://cs.stackexchange.com/questions/117989/hausdorff-distance-between-two-binary-images-according-to-distance-maps
def total_hausdorff_distance(pred_tensor, gt_tensor):
    hausdorf_metrics = torch.zeros((gt_tensor.shape[0], gt_tensor.shape[1]))
    for class_idx in range(gt_tensor.shape[1]):
        for sample_idx in range(gt_tensor.shape[0]):
            ground_truth_img = gt_tensor[sample_idx, class_idx]
            prediction_img = pred_tensor[sample_idx, class_idx]
            hausdorf_metrics[sample_idx, class_idx] = calculate_hausdorff_distance(ground_truth_img, prediction_img)
    return hausdorf_metrics

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


def hausdorff_distance_fast(img1, img2):
    """
    Calculate the Hausdorff distance between two binary images using distance maps.
    
    Arguments:
    img1, img2 -- torch tensors of shape (H, W) with 0 for background and 1 for the segmented object.
    
    Returns:
    Hausdorff distance between the two images.
    """
    # Convert torch tensors to numpy arrays
    img1_np = img1.cpu().numpy().astype(np.uint8)
    img2_np = img2.cpu().numpy().astype(np.uint8)
    
    # Compute the complement (background becomes object and vice versa)
    complement_img1 = 1 - img1_np
    complement_img2 = 1 - img2_np
    
    # Compute the distance transform for the complements
    dist_map_img1 = distance_transform_edt(complement_img1)
    dist_map_img2 = distance_transform_edt(complement_img2)
    
    # Compute h(Y, X): the maximum distance in dist_map_img1 for points where img2 is 1
    h_yx = np.max(dist_map_img1[img2_np == 1])
    
    # Compute h(X, Y): the maximum distance in dist_map_img2 for points where img1 is 1
    h_xy = np.max(dist_map_img2[img1_np == 1])
    
    # Hausdorff distance is the maximum of the two distances
    hausdorff_dist = max(h_yx, h_xy)
    
    return hausdorff_dist



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
    
def jaccard_index(pred, gt):
    # Compute intersection and union over the last two dimensions
    intersection = torch.logical_and(gt == 1, pred == 1).sum(dim=(-2, -1)).float()
    union = torch.logical_or(gt == 1, pred == 1).sum(dim=(-2, -1)).float()
    
    # Handle division by zero (no positives in ground truth and prediction)
    iou = intersection / (union + 1e-6)  # Adding small epsilon to prevent division by zero
    return iou

def precision_metric(pred, gt):
    # Compute true positives and predicted positives over the last two dimensions
    true_positive = torch.logical_and(gt == 1, pred == 1).sum(dim=(-2, -1)).float()
    predicted_positive = (pred == 1).sum(dim=(-2, -1)).float()
    
    # Handle division by zero (no positives predicted)
    precision = true_positive / (predicted_positive + 1e-6)  # Adding small epsilon to prevent division by zero
    return precision

def volumetric_similarity(pred, gt):
    # Calculate the volumes (number of foreground pixels)
    vol_gt = (gt == 1).sum(dim=(-2, -1)).float()
    vol_pred = (pred == 1).sum(dim=(-2, -1)).float()

    # Compute the absolute difference and the total volume
    abs_diff = torch.abs(vol_gt - vol_pred)
    total_vol = vol_gt + vol_pred

    # Return Volumetric Similarity (1 - relative volume difference)
    vs = 1 - abs_diff / (total_vol + 1e-6)  # Adding epsilon to avoid division by zero
    return vs


