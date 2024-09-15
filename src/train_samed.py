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
import torch.optim as optim
import torch.nn as nn
from torch.amp import GradScaler, autocast
from tqdm import tqdm


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
    # m = nn.MaxPool2d(kernel_size=2, stride=2)

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
            # m,
            itemgetter(0),
        ]
    )

    # 94016114
    # 4492658

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

    model = LoRA_Sam(sam, r=4)    

    # model.load_lora_parameters("") ones you have the file
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    # scaler = GradScaler()

    meansum = 0
    stdsum = 0
    print(len(train_loader))
    for i, data in tqdm(enumerate(train_loader)):
        optimizer.zero_grad()
        # if i == 5:
        #     break
    
        img = data["images"].to(device)
        gt = data["gts"].to(device)
        # print(img.shape)
        # print("mean", img.mean())
        # print("std", img.std())
        meansum += img.mean()
        stdsum += img.std()

        # preds = model(img)

        # note preds['masks'] is true and false

        # import code; code.interact(local=locals())

        # with autocast("cuda" if torch.cuda.is_available() else "cpu"):
        # preds = model(img)
        # print(preds['low_res_logits'].shape)
        # # import code; code.interact(local=locals())
        # loss, loss_ce, loss_dice = calc_loss(preds, gt, ce_loss, dice_loss, dice_weight=0.8)
        # loss.backward()
        # print(loss.item())
        # optimizer.step()
        # for key in preds:
        #     print(key, preds[key].shape)

        # loss, loss_ce, loss_dice = calc_loss(preds, gt, ce_loss, dice_loss, dice_weight=0.8)
        # loss.backward()
        # optimizer.step()
        # scaler.scale(loss).backward()
        # scaler.step(optimizer)
        # scaler.update()

        # print(f"[{i + 1}, {i + 1}] loss: {loss.item() / 10:.3f}")
    print(meansum / len(train_loader))
    print(stdsum / len(train_loader))


if __name__ == "__main__":
    main()