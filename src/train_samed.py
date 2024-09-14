from samed.sam_lora import LoRA_Sam
from samed.segment_anything import sam_model_registry

from dataset import SliceDataset
from utils import class2one_hot
from pathlib import Path
from operator import itemgetter
from torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np
import torch


def dataloaders():
    K = 5

    root_dir = Path("data") / "SEGTHOR"

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

    train_set = SliceDataset(
        "train",
        root_dir,
        img_transform=img_transform,
        gt_transform=gt_transform,
        debug=False,
    )
    train_loader = DataLoader(train_set, batch_size=1, num_workers=4, shuffle=True)

    val_set = SliceDataset(
        "val",
        root_dir,
        img_transform=img_transform,
        gt_transform=gt_transform,
        debug=False,
    )
    val_loader = DataLoader(val_set, batch_size=1, num_workers=4, shuffle=False)

    return (train_loader, val_loader)


def main():
    train_loader, val_loader = dataloaders()
    sam = sam_model_registry["vit_b"](  # TODO check these arguments 
        num_classes=5,
        pixel_mean=[0, 0, 0],
        pixel_std=[1, 1, 1],
    )
    model = LoRA_Sam(sam, r=4)
    
    for i, data in enumerate(train_loader):

        if i == 1:
            break
    
        img = data["images"]
        gt = data["gts"]
        print(img.shape)
        print(gt.shape)

        preds = model(img)
        for key in preds:
            print(key, preds[key].shape)


if __name__ == "__main__":
    main()