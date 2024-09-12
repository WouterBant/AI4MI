#!/usr/bin/env python3

# MIT License

# Copyright (c) 2024 Hoel Kervadec

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import argparse
import warnings
from typing import Any
from pathlib import Path
from pprint import pprint
from operator import itemgetter
from shutil import copytree, rmtree
from datetime import datetime
import wandb

import torch
import numpy as np
import torch.nn.functional as F
from torch import nn, Tensor
from torchvision import transforms
from torch.utils.data import DataLoader

from dataset import SliceDataset
from ShallowNet import shallowCNN
from ENet import ENet
from utils import (
    Dcm,
    class2one_hot,
    get_optimizer,
    init_wandb,
    log_sample_images_wandb,
    probs2one_hot,
    probs2class,
    tqdm_,
    dice_coef,
    save_images,
    set_seed,
)
from losses import CrossEntropy


datasets_params: dict[str, dict[str, Any]] = {}
# K for the number of classes
# Avoids the clases with C (often used for the number of Channel)
datasets_params["TOY2"] = {"K": 2, "net": shallowCNN, "B": 2, "names": ["background", "foreground"]}
datasets_params["SEGTHOR"] = {"K": 5, "net": ENet, "B": 8, "names": ["Background", "Esophagus", "Heart", "Trachea", "Aorta"]}


def setup(args) -> tuple[nn.Module, Any, Any, DataLoader, DataLoader, int]:
    # Networks and scheduler
    gpu: bool = args.gpu and torch.cuda.is_available()
    device = torch.device("cuda") if gpu else torch.device("cpu")

    if args.gpu and not torch.cuda.is_available():
        print(
            ">> NOTE GPU is picked but is not available, defaulting to CPU if in debug mode"
        )
        if not args.debug:
            raise RuntimeError("GPU is picked but is not available")

    print(f">> Picked {device} to run experiments")

    K: int = datasets_params[args.dataset]["K"]
    net = datasets_params[args.dataset]["net"](1, K)
    net.init_weights()  # TODO probably remove it and use the default one from pytorch
    net.to(device)

    # Use torch.compile for kernel fusion to be faster on the gpu
    if gpu and args.epochs > 3:  # jit compilation takes too much time for few epochs
        print(">> Compiling the network for faster execution")
        net = torch.compile(net)

    # TODO consider using a learning rate scheduler
    # TODO consider adding weight decay and gradient clipping
    # TODO consider using SGD with momentum as it works well for cnns
    # TODO otherwise use AdamW as it is a bug fix of Adam that is more stable
    # TODO consider training with mixed precision to speed up training
    # Initialize optimizer based on args
    optimizer = get_optimizer(args, net)

    # Dataset part
    B: int = args.batch_size
    root_dir = Path("data") / args.dataset

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
        debug=args.debug,
    )
    train_loader = DataLoader(train_set, batch_size=B, num_workers=args.num_workers, shuffle=True)

    val_set = SliceDataset(
        "val",
        root_dir,
        img_transform=img_transform,
        gt_transform=gt_transform,
        debug=args.debug,
    )
    val_loader = DataLoader(val_set, batch_size=B, num_workers=args.num_workers, shuffle=False)

    args.dest.mkdir(parents=True, exist_ok=True)

    return (net, optimizer, device, train_loader, val_loader, K)


def runTraining(args):
    print(f">>> Setting up to train on {args.dataset} with {args.mode}")
    net, optimizer, device, train_loader, val_loader, K = setup(args)

    if args.mode == "full":
        loss_fn = CrossEntropy(
            idk=list(range(K))
        )  # Supervise both background and foreground
    elif args.mode in ["partial"] and args.dataset in ["SEGTHOR", "SEGTHOR_STUDENTS"]:
        print(
            ">> NOTE Partial loss will not supervise the heart (class 2) so don't use it"
        )
        loss_fn = CrossEntropy(idk=[0, 1, 3, 4])  # Do not supervise the heart (class 2)
    else:
        raise ValueError(args.mode, args.dataset)

    # Notice one has the length of the _loader_, and the other one of the _dataset_
    log_loss_tra: Tensor = torch.zeros((args.epochs, len(train_loader)))
    log_dice_tra: Tensor = torch.zeros((args.epochs, len(train_loader.dataset), K))
    log_loss_val: Tensor = torch.zeros((args.epochs, len(val_loader)))
    log_dice_val: Tensor = torch.zeros((args.epochs, len(val_loader.dataset), K))

    best_dice = train_steps_done = val_steps_done = 0

    for e in range(args.epochs):
        for m in ["train", "val"]:
            match m:
                case "train":
                    net.train()
                    opt = optimizer
                    cm = Dcm
                    desc = f">> Training   ({e: 4d})"
                    loader = train_loader
                    log_loss = log_loss_tra
                    log_dice = log_dice_tra
                case "val":
                    net.eval()
                    opt = None
                    cm = torch.no_grad
                    desc = f">> Validation ({e: 4d})"
                    loader = val_loader
                    log_loss = log_loss_val
                    log_dice = log_dice_val
            with cm():  # Either dummy context manager, or the torch.no_grad for validation
                j = 0
                tq_iter = tqdm_(enumerate(loader), total=len(loader), desc=desc)
                for i, data in tq_iter:
                    img = data["images"].to(device)
                    gt = data["gts"].to(device)

                    if opt:  # So only for training
                        opt.zero_grad()

                    # Sanity tests to see we loaded and encoded the data correctly
                    assert 0 <= img.min() and img.max() <= 1
                    B, _, W, H = img.shape

                    pred_logits = net(img)
                    pred_probs = F.softmax(
                        1 * pred_logits, dim=1
                    )  # 1 is the temperature parameter

                    # Metrics computation, not used for training
                    pred_seg = probs2one_hot(pred_probs)
                    log_dice[e, j : j + B, :] = dice_coef(
                        pred_seg, gt
                    )  # One DSC value per sample and per class

                    # TODO: add additional metrics

                    loss = loss_fn(pred_probs, gt)
                    log_loss[e, i] = (
                        loss.item()
                    )  # One loss value per batch (averaged in the loss)

                    if args.use_wandb:
                        if m == "train":
                            train_steps_done += 1
                            steps_done = train_steps_done
                        elif m == "val":
                            val_steps_done += 1
                            steps_done = val_steps_done

                        # Log metrics every 50 steps
                        if m == "train" and steps_done % 50 == 0:
                            metrics = {
                                f"{m}_dice_{k}": log_dice[e, j : j + B, k].mean().item()
                                for k in range(K)
                            }
                            metrics[f"{m}_loss"] = loss.item()
                            wandb.log(metrics)

                        # Log sample images every 1000 steps
                        if steps_done % 1000 == 0:
                            log_sample_images_wandb(
                                img, gt, pred_probs, K, steps_done, m, datasets_params[args.dataset]["names"]
                            )

                    if opt:  # Only for training
                        loss.backward()
                        if args.clip_grad:
                            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                        opt.step()

                    if m == "val":
                        with warnings.catch_warnings():
                            warnings.filterwarnings("ignore", category=UserWarning)
                            predicted_class: Tensor = probs2class(pred_probs)
                            mult: int = 63 if K == 5 else (255 / (K - 1))
                            save_images(
                                predicted_class * mult,
                                data["stems"],
                                args.dest / f"iter{e:03d}" / m,
                            )

                    j += B  # Keep in mind that _in theory_, each batch might have a different size
                    # For the DSC average: do not take the background class (0) into account:
                    postfix_dict: dict[str, str] = {
                        "Dice": f"{log_dice[e, :j, 1:].mean():05.3f}",
                        "Loss": f"{log_loss[e, :i + 1].mean():5.2e}",
                    }
                    if K > 2:
                        postfix_dict |= {
                            f"Dice-{k}": f"{log_dice[e, :j, k].mean():05.3f}"
                            for k in range(1, K)
                        }
                    tq_iter.set_postfix(postfix_dict)

        # I save it at each epochs, in case the code crashes or I decide to stop it early
        np.save(args.dest / "loss_tra.npy", log_loss_tra)
        np.save(args.dest / "dice_tra.npy", log_dice_tra)
        np.save(args.dest / "loss_val.npy", log_loss_val)
        np.save(args.dest / "dice_val.npy", log_dice_val)

        # Log the averaged validation metrics only at the end of each epoch
        if args.use_wandb:
            metrics = {
                f"val_dice_{k}": log_dice_val[e, :, k].mean().item() for k in range(K)
            }
            metrics[f"val_loss"] = log_loss_val[e, :].mean().item()
            wandb.log(metrics)

        current_dice: float = log_dice_val[e, :, 1:].mean().item()
        if current_dice > best_dice:
            print(
                f">>> Improved dice at epoch {e}: {best_dice:05.3f}->{current_dice:05.3f} DSC"
            )
            best_dice = current_dice
            with open(args.dest / "best_epoch.txt", "w") as f:
                f.write(str(e))

            best_folder = args.dest / "best_epoch"
            if best_folder.exists():
                rmtree(best_folder)
            copytree(args.dest / f"iter{e:03d}", Path(best_folder))

            torch.save(net, args.dest / "bestmodel.pkl")
            torch.save(net.state_dict(), args.dest / "bestweights.pt")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--epochs", default=200, type=int)
    parser.add_argument("--lr", default=0.0005, type=float, help="Learning rate")
    parser.add_argument(
        "--batch_size",
        type=int,
        help="Batch size",
        default=None,
    )
    parser.add_argument(
        "--optimizer",
        default="adam",
        choices=["adam", "sgd", "adamw"],
        help="Optimizer to use",
    )
    parser.add_argument("--dataset", default="SEGTHOR", choices=datasets_params.keys())
    parser.add_argument("--mode", default="full", choices=["partial", "full"])
    parser.add_argument(
        "--dest",
        type=Path,
        default=None,
        help="Destination directory to save the results (predictions and weights).",
    )
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Keep only a fraction (10 samples) of the datasets, "
        "to test the logic around epochs and logging easily.",
    )
    parser.add_argument(
        "--deterministic", action="store_true", help="Make the training deterministic"
    )
    parser.add_argument(
        "--use_wandb", action="store_true", help="Use wandb for logging"
    )
    parser.add_argument(
        "--clip_grad", action="store_true", help="Enable gradient clipping"
    )

    args = parser.parse_args()

    pprint(args)

    if args.deterministic:
        set_seed(2024)

    # Disable wandb if it failed to initialize
    args.use_wandb = args.use_wandb and init_wandb(args)

    if args.batch_size is None:
        args.batch_size = datasets_params[args.dataset]["B"]

    if args.dest is None:
        args.dest = Path(f"results/{args.dataset}/{datetime.now().strftime("%Y-%m-%d")}")

    runTraining(args)


if __name__ == "__main__":
    main()