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
import os
import wandb
from pathlib import Path
from functools import partial
from multiprocessing import Pool
from contextlib import AbstractContextManager
from typing import Callable, Iterable, List, Set, Tuple, TypeVar, cast

import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from torch import Tensor, einsum
import random


tqdm_ = partial(
    tqdm,
    dynamic_ncols=True,
    leave=True,
    bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{rate_fmt}{postfix}]",
)


class Dcm(AbstractContextManager):
    # Dummy Context manager
    def __exit__(self, *args, **kwargs):
        pass


# Functools
A = TypeVar("A")
B = TypeVar("B")


def map_(fn: Callable[[A], B], iter: Iterable[A]) -> List[B]:
    return list(map(fn, iter))


def mmap_(fn: Callable[[A], B], iter: Iterable[A]) -> List[B]:
    return Pool().map(fn, iter)


def starmmap_(fn: Callable[[Tuple[A]], B], iter: Iterable[Tuple[A]]) -> List[B]:
    return Pool().starmap(fn, iter)


# Assert utils
def uniq(a: Tensor) -> Set:
    return set(torch.unique(a.cpu()).numpy())


def sset(a: Tensor, sub: Iterable) -> bool:
    return uniq(a).issubset(sub)


def eq(a: Tensor, b) -> bool:
    return torch.eq(a, b).all()


def simplex(t: Tensor, axis=1) -> bool:
    _sum = cast(Tensor, t.sum(axis).type(torch.float32))
    _ones = torch.ones_like(_sum, dtype=torch.float32)
    return torch.allclose(_sum, _ones)


def one_hot(t: Tensor, axis=1) -> bool:
    return simplex(t, axis) and sset(t, [0, 1])


def class2one_hot(seg: Tensor, K: int) -> Tensor:
    # Breaking change but otherwise can't deal with both 2d and 3d
    # if len(seg.shape) == 3:  # Only w, h, d, used by the dataloader
    #     return class2one_hot(seg.unsqueeze(dim=0), K)[0]

    assert sset(seg, list(range(K))), (uniq(seg), K)

    b, *img_shape = seg.shape

    device = seg.device
    res = torch.zeros((b, K, *img_shape), dtype=torch.int32, device=device).scatter_(
        1, seg[:, None, ...], 1
    )

    assert res.shape == (b, K, *img_shape)
    assert one_hot(res)

    return res


def probs2class(probs: Tensor) -> Tensor:
    b, _, *img_shape = probs.shape
    assert simplex(probs)

    res = probs.argmax(dim=1)
    assert res.shape == (b, *img_shape)

    return res


def probs2one_hot(probs: Tensor) -> Tensor:
    _, K, *_ = probs.shape
    assert simplex(probs)

    res = class2one_hot(probs2class(probs), K)
    assert res.shape == probs.shape
    assert one_hot(res)

    return res


# Save the raw predictions
def save_images(segs: Tensor, names: Iterable[str], root: Path) -> None:
    for seg, name in zip(segs, names):
        save_path = (root / name).with_suffix(".png")
        save_path.parent.mkdir(parents=True, exist_ok=True)

        if len(seg.shape) == 2:
            Image.fromarray(seg.detach().cpu().numpy().astype(np.uint8)).save(save_path)
        elif len(seg.shape) == 3:
            np.save(str(save_path), seg.detach().cpu().numpy())
        else:
            raise ValueError(seg.shape)


# Metrics
def meta_dice(
    sum_str: str, label: Tensor, pred: Tensor, smooth: float = 1e-8
) -> Tensor:
    assert label.shape == pred.shape
    assert one_hot(label)
    assert one_hot(pred)

    inter_size: Tensor = einsum(sum_str, [intersection(label, pred)]).type(
        torch.float32
    )
    sum_sizes: Tensor = (einsum(sum_str, [label]) + einsum(sum_str, [pred])).type(
        torch.float32
    )

    dices: Tensor = (2 * inter_size + smooth) / (sum_sizes + smooth)

    return dices


dice_coef = partial(meta_dice, "bk...->bk")
dice_batch = partial(meta_dice, "bk...->k")  # used for 3d dice


def intersection(a: Tensor, b: Tensor) -> Tensor:
    assert a.shape == b.shape
    assert sset(a, [0, 1])
    assert sset(b, [0, 1])

    res = a & b
    assert sset(res, [0, 1])

    return res


def union(a: Tensor, b: Tensor) -> Tensor:
    assert a.shape == b.shape
    assert sset(a, [0, 1])
    assert sset(b, [0, 1])

    res = a | b
    assert sset(res, [0, 1])

    return res


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


def init_wandb(args: argparse.Namespace) -> bool:
    try:
        os.environ["WANDB_API_KEY"] = os.getenv("WANDB_API_KEY")
    except TypeError:
        print("Please set your WANBD_API_KEY as an environment variable")
        print("You can find your API key at: https://wandb.ai/authorize")
        print("You can set it as an environment variable using:")
        print("export WANDB_API_KEY='your_api_key'")
        print("Defaulting to no wandb logging")
        return False

    try:
        wandb.init(project="segmentation", entity="AIfourMI", config=args)
    except Exception as e:
        print(f"Error initializing wandb: {e}")
        return False

    return True


def log_sample_images_wandb(
    img: Tensor, gt: Tensor, predicted_class: Tensor, mult: int, steps_done: int, m: str
):
    # Select a random image from the batch
    idx = random.randint(0, B - 1)

    # Convert tensors to numpy arrays and scale to 0-255
    img_np = (img[idx, 0].cpu().numpy() * 255).astype(np.uint8)
    gt_np = (gt[idx, 1:].argmax(dim=0).cpu().numpy() * mult).astype(np.uint8)
    pred_np = (predicted_class[idx].cpu().numpy() * mult).astype(np.uint8)

    # Create a side-by-side comparison
    comparison = np.hstack((img_np, gt_np, pred_np))

    # Log the image to wandb
    wandb.log(
        {
            f"{m}_sample_images_{steps_done}": wandb.Image(
                comparison,
                caption=f"Left: Input | Middle: Ground Truth | Right: Prediction",
            )
        }
    )


def get_optimizer(
    args: argparse.Namespace, net: torch.nn.Module
) -> torch.optim.Optimizer:
    if args.optimizer == "adam":
        optimizer = torch.optim.Adam(net.parameters(), lr=args.lr, betas=(0.9, 0.999))
    elif args.optimizer == "sgd":
        optimizer = torch.optim.SGD(net.parameters(), lr=args.lr, momentum=0.9)
    elif args.optimizer == "adamw":
        optimizer = torch.optim.AdamW(
            net.parameters(),
            lr=args.lr,
            betas=(0.9, 0.999),
            weight_decay=args.weight_decay,
        )
    else:
        raise ValueError(f"Unsupported optimizer: {args.optimizer}")

    return optimizer
