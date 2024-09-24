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

from pathlib import Path
from typing import Callable, Union
import random

import torch
from torch import Tensor
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


def make_dataset(root, subset) -> list[tuple[Path, Path]]:
    assert subset in ["train", "val", "test"]

    root = Path(root)

    img_path = root / subset / "img"
    full_path = root / subset / "gt"

    images = sorted(img_path.glob("*.png"))
    full_labels = sorted(full_path.glob("*.png"))

    return list(zip(images, full_labels))


class SliceDataset(Dataset):
    def __init__(
        self,
        subset,
        root_dir,
        img_transform=None,
        gt_transform=None,
        augment=False,
        equalize=False,
        debug=False,
    ):
        self.root_dir: str = root_dir
        self.img_transform: Callable = img_transform
        self.gt_transform: Callable = gt_transform
        self.augmentation: bool = augment  # TODO: implement
        self.equalize: bool = equalize  # TODO: know if we need it

        # TODO make our own test set, now 5453 train and 1967 val
        self.files = make_dataset(root_dir, subset)
        if debug:
            self.files = self.files[:10]

        print(f">> Created {subset} dataset with {len(self)} images...")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index) -> dict[str, Union[Tensor, int, str]]:
        img_path, gt_path = self.files[index]

        img: Tensor = self.img_transform(Image.open(img_path))
        gt: Tensor = self.gt_transform(Image.open(gt_path))

        if self.augmentation:

            # Apply augmentation if random is above trheshold
            if random.random() > 2/3:

                # Only apply one augmentation at a time
                random_val = random.random()

                if random_val < 0.2:
                    img = transforms.functional.vflip(img)
                    gt = transforms.functional.vflip(gt)

                elif random_val < 0.4:
                    angle = random.uniform(-5, 5)
                    img = transforms.functional.rotate(img, angle)
                    gt = transforms.functional.rotate(gt, angle)

                elif random_val < 0.7:
                    # Custom cropping implementation
                    crop_size = (int(img.size(1) / 1.1), int(img.size(2) / 1.1))
                    i = random.randint(0, img.size(1) - crop_size[0])
                    j = random.randint(0, img.size(2) - crop_size[1])
                    img = img[:, i:i+crop_size[0], j:j+crop_size[1]]
                    gt = gt[:, i:i+crop_size[0], j:j+crop_size[1]]
                    
                else:
                    noise = torch.randn(img.size()) * 0.01
                    img = img + noise

        _, W, H = img.shape
        K, _, _ = gt.shape
        assert gt.shape == (K, W, H)

        return {"images": img, "gts": gt, "stems": img_path.stem}