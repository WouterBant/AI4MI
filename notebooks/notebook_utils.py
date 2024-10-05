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
) -> List[WorstPrediction]:
    """
    Get the worst predictions from a model based on the highest loss values.

    Args:
    model (nn.Module): The PyTorch model to evaluate.
    dataloader (DataLoader): DataLoader for the dataset. Assumed to have shuffle=False.
    device (torch.device): The device to run the model on.
    loss_fn (Callable): Loss function that takes (y_pred, y) and returns a loss tensor.
    num_worst (int): Number of worst predictions to return.

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
                pred_logits = model(x)
                y_pred = F.softmax(
                    1 * pred_logits, dim=1
                )  # 1 is the temperature parameter
                losses = l(y_pred, y)  # (batch_size,)
            case _:
                raise ValueError(f"Unknown loss function: {loss_fn}")

        for i in range(len(x)):
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
    show_segmentation_area: bool = True,
    show_segmentation_boundary: bool = True,
):
    N = len(worst_predictions)
    fig, ax = plt.subplots(N, 3, figsize=(9, N * 3))
    plt.subplots_adjust(wspace=0.1, hspace=0.3)

    for idx, bad_prediction in enumerate(worst_predictions):
        loss, img, gt, pred = attrgetter("loss", "image", "gt", "prediction")(
            bad_prediction
        )
        color_map = plt.get_cmap("viridis", 5)
        gt_colored = color_map(gt.argmax(dim=0).cpu().numpy())
        pred_colored = color_map(pred.argmax(dim=0).cpu().numpy())
        gt_boundaries = find_boundaries(gt.argmax(dim=0).cpu().numpy())
        pred_boundaries = find_boundaries(pred.argmax(dim=0).cpu().numpy())

        ax[idx, 0].annotate(
            f"Loss: {loss:.4f}",
            xy=(0, 1.1),
            xycoords="axes fraction",
            fontsize=12,
            ha="left",
            va="bottom",
            color="black",
        )
        ax[idx, 0].imshow(img[0], cmap="gray")

        # Overlay the colored ground truth on the original image
        ax[idx, 1].imshow(img[0], cmap="gray")
        if show_segmentation_area:
            ax[idx, 1].imshow(gt_colored, alpha=0.5)
        if show_segmentation_boundary:
            ax[idx, 1].imshow(gt_boundaries, cmap="inferno", alpha=0.5)

        ax[idx, 2].imshow(img[0], cmap="gray")
        if show_segmentation_area:
            ax[idx, 2].imshow(pred_colored, alpha=0.5)
        if show_segmentation_boundary:
            ax[idx, 2].imshow(pred_boundaries, cmap="inferno", alpha=0.5)

        for i in range(3):
            ax[idx, i].axis("off")

    fig.tight_layout(rect=[0, 0, 1, 0.95])

    if show_segmentation_area:
        legend_elements = [
            plt.Rectangle((0, 0), 1, 1, color=color_map(i)) for i in range(5)
        ]
        fig.legend(
            legend_elements,
            ["Background", "Esophagus", "Heart", "Trachea", "Aorta"],
            loc="lower center",
            ncol=5,
        )
    plt.show()


def get_dataloader(split: str = "train", batch_size: int = 1) -> DataLoader:
    root_dir = "../" / Path("data") / "SEGTHOR"

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
    return DataLoader(_set, batch_size=batch_size, num_workers=4, shuffle=False)
