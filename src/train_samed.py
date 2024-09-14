from samed.sam_lora import LoRA_Sam
from samed.segment_anything import sam_model_registry
from losses import DiceLoss

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


def calc_loss(outputs, low_res_label_batch, ce_loss, dice_loss, dice_weight:float=0.8):
    low_res_logits = outputs['low_res_logits']
    loss_ce = ce_loss(low_res_logits, low_res_label_batch[:].float())
    loss_dice = dice_loss(low_res_logits, low_res_label_batch, softmax=True)
    loss = (1 - dice_weight) * loss_ce + dice_weight * loss_dice
    return loss, loss_ce, loss_dice

def main():
    ce_loss = torch.nn.CrossEntropyLoss()
    dice_loss = DiceLoss(5)
    train_loader, val_loader = dataloaders()
    sam, _ = sam_model_registry["vit_b"](  # TODO check these arguments 
        checkpoint="src/samed/checkpoints/sam_vit_b_01ec64.pth",
        num_classes=5,
        pixel_mean=[0, 0, 0],
        pixel_std=[1, 1, 1],
    )
    # print num parameters
    print(sum(p.numel() for p in sam.parameters() if p.requires_grad))

    model = LoRA_Sam(sam, r=4)
    print(sum(p.numel() for p in model.parameters() if p.requires_grad))

    # model.load_lora_parameters("") ones you have the file
    
    for i, data in enumerate(train_loader):

        if i == 5:
            break
    
        img = data["images"]
        gt = data["gts"]
        print(img.shape)
        print(gt.shape)

        preds = model(img)

        # note preds['masks'] is true and false

        import code; code.interact(local=locals())
        for key in preds:
            print(key, preds[key].shape)

        loss, loss_ce, loss_dice = calc_loss(preds, gt, ce_loss, dice_loss, dice_weight=0.8)
        


if __name__ == "__main__":
    main()