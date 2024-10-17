import heapq
from operator import itemgetter
from typing import List

import sys

sys.path.append("../src")
from dataset import SliceDataset
from utils import class2one_hot

from tqdm import tqdm
from pathlib import Path
from dataclasses import dataclass
from operator import attrgetter

import torch
from torch import nn
from torch import einsum
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt
from skimage.segmentation import find_boundaries
from matplotlib.colors import LinearSegmentedColormap


class CrossEntropy:
    # returns loss for each element in the batch

    def __init__(self, **kwargs):
        self.idk = kwargs["idk"]

    def __call__(self, pred_softmax: torch.Tensor, weak_target: torch.Tensor):
        log_p = (pred_softmax[:, self.idk, ...] + 1e-10).log()
        mask = weak_target[:, self.idk, ...].float()

        loss = -einsum("bkwh,bkwh->b", mask, log_p)
        loss /= mask.sum() + 1e-10

        return loss


class _Tensor:
    """
    Wrapper to make ordering and hashing of tensors possible
    """

    def __init__(self, tensor: torch.Tensor):
        self.tensor = tensor

    def __hash__(self):
        # let god decide the order
        return hash(self.tensor.data_ptr())

    def __eq__(self, other):
        return self.__hash__() == other.__hash__()

    def __lt__(self, other):
        return self.__hash__() < other.__hash__()


@dataclass
class WorstPrediction:
    loss: int
    image: torch.Tensor
    gt: torch.Tensor
    prediction: torch.Tensor


@torch.no_grad()
def get_worst_predictions(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    loss_fn: str,
    num_worst: int = 10,
    filter_empty_preds: bool = False,
) -> List[WorstPrediction]:
    """
    Get the worst predictions from a model based on the highest loss values.

    Args:
    model (nn.Module): The PyTorch model to evaluate.
    dataloader (DataLoader): DataLoader for the dataset. Assumed to have shuffle=False.
    device (torch.device): The device to run the model on.
    loss_fn (Callable): Loss function that takes (y_pred, y) and returns a loss tensor.
    num_worst (int): Number of worst predictions to return.
    filter_empty_preds (bool): Whether to filter out predictions with no positive pixels.

    Returns:
    List[Tuple[float, torch.Tensor, torch.Tensor, torch.Tensor]]:
        List of tuples containing (loss, input, true_label, predicted_output)
        for the worst predictions, sorted from worst to best.
    """
    model.eval()
    worst_predictions = []

    for batch in tqdm(dataloader):
        x, y = batch["images"], batch["gts"]
        x, y = x.to(device), y.to(device)

        match loss_fn:
            case "cross_entropy":
                l = CrossEntropy(idk=[1, 2, 3, 4])
                if hasattr(model, "sam"):
                    pred_logits = model(x, multimask_output=True, image_size=512)["masks"]
                else:
                    pred_logits = model(x)
                y_pred = F.softmax(
                    1 * pred_logits, dim=1
                )  # 1 is the temperature parameter
                losses = l(y_pred, y)  # (batch_size,)
            case _:
                raise ValueError(f"Unknown loss function: {loss_fn}")

        for i in range(len(x)):
            if filter_empty_preds and y_pred[i].argmax(dim=0).sum() == 0:
                continue
            loss_value = losses[i].item() if torch.is_tensor(losses[i]) else losses[i]
            item = (loss_value, _Tensor(x[i]), _Tensor(y[i]), _Tensor(y_pred[i]))

            if len(worst_predictions) < num_worst:
                heapq.heappush(worst_predictions, item)
            else:
                heapq.heappushpop(worst_predictions, item)

    return [
        WorstPrediction(a, b.tensor, c.tensor, d.tensor)
        for (a, b, c, d) in worst_predictions
    ]

def visualize_worst_predictions(
    worst_predictions: List[WorstPrediction],
    batch_size: int = 3
):
    # Define the colors for the classes (Esophagus, Heart, Trachea, Aorta)
    colors = ['red', 'green', 'blue', 'yellow']
    color_labels = [0, 1, 2, 3, 4]  # Class 0 (background) is excluded

    def apply_color_overlay(img, mask, class_colors, labels):
        colored_img = np.stack([img, img, img], axis=-1)  # Convert grayscale to RGB
        for class_idx, color in enumerate(class_colors, start=1):
            mask_class = mask == labels[class_idx]  # Mask for the class
            rgba_color = plt.cm.colors.to_rgba(color, alpha=0.5)  # Apply alpha for color
            colored_img[mask_class] = rgba_color[:3]  # Ignore alpha channel here
        return colored_img

    # Process predictions in batches
    for batch_start in range(0, len(worst_predictions), batch_size):
        batch_end = min(batch_start + batch_size, len(worst_predictions))
        batch = worst_predictions[batch_start:batch_end]
        
        fig, axes = plt.subplots(len(batch), 3, figsize=(20, 5 * len(batch)))
        plt.subplots_adjust(wspace=0.05, hspace=0.2)

        if len(batch) == 1:
            axes = np.expand_dims(axes, axis=0)

        for idx, bad_prediction in enumerate(batch):
            loss, img, gt, pred = bad_prediction.loss, bad_prediction.image, bad_prediction.gt, bad_prediction.prediction
            
            img = img.cpu().numpy().squeeze()
            gt = gt.cpu().argmax(dim=0).numpy()
            pred = pred.cpu().argmax(dim=0).numpy()

            # Plot the original input image
            axes[idx, 0].imshow(img, cmap='gray')
            axes[idx, 0].axis('off')

            # Apply ground truth color overlay
            gt_colored = apply_color_overlay(img, gt, colors, color_labels)
            axes[idx, 1].imshow(img, cmap='gray')
            axes[idx, 1].imshow(gt_colored, alpha=0.5)
            axes[idx, 1].axis('off')

            # Apply prediction color overlay
            pred_colored = apply_color_overlay(img, pred, colors, color_labels)
            axes[idx, 2].imshow(img, cmap='gray')
            axes[idx, 2].imshow(pred_colored, alpha=0.5)
            axes[idx, 2].axis('off')

            # Add loss information as text above the first image
            axes[idx, 0].text(0.5, 1.05, f"Loss: {loss:.4f}", transform=axes[idx, 0].transAxes,
                              fontsize=20, ha='center', va='bottom')

        # Add a legend with color representation for each class
        legend_elements = [
            plt.Rectangle((0, 0), 1, 1, facecolor=color, alpha=0.5, label=label)
            for color, label in zip(colors, ['Esophagus', 'Heart', 'Trachea', 'Aorta'])
        ]

        fig.legend(handles=legend_elements, loc="lower center", ncol=4, 
                   fancybox=True, shadow=True, title="Predicted Segmentation Classes:", 
                   fontsize=12, title_fontsize=12)

        plt.tight_layout()
        plt.show()

def get_dataloader(split: str = "train", batch_size: int = 1) -> DataLoader:
    root_dir = "../" / Path("data") / "SEGTHOR_MANUAL_SPLIT"

    K = 5  # Number of classes

    img_transform = transforms.Compose(
        [
            lambda img: img.convert("L"),  # convert to grayscale
            lambda img: np.array(img)[np.newaxis, ...],
            lambda nd: nd / 255,  # max <= 1 (range [0, 1])
            lambda nd: torch.tensor(nd, dtype=torch.float32),
        ]
    )

    gt_transform = transforms.Compose(
        [
            lambda img: np.array(img)[...],
            # The idea is that the classes are mapped to {0, 255} for binary cases
            # {0, 85, 170, 255} for 4 classes
            # {0, 51, 102, 153, 204, 255} for 6 classes
            # Very sketchy but that works here and that simplifies visualization
            lambda nd: nd / (255 / (K - 1)) if K != 5 else nd / 63,  # max <= 1
            lambda nd: torch.tensor(nd, dtype=torch.int64)[
                None, ...
            ],  # Add one dimension to simulate batch
            lambda t: class2one_hot(t, K=K),
            itemgetter(0),
        ]
    )

    _set = SliceDataset(
        split,
        root_dir,
        img_transform=img_transform,
        gt_transform=gt_transform,
        debug=False,
    )
    return DataLoader(_set, batch_size=batch_size, num_workers=0, shuffle=False)
