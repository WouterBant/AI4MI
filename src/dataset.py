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
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import numpy as np
from data_augmenter import CTImageDataset, load_ct_images_and_gts, AugmentationPipeline


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
        normalize=False,
        debug=False,
    ):
        self.root_dir: str = root_dir
        self.img_transform: Callable = img_transform
        self.gt_transform: Callable = gt_transform
        self.augmentation: bool = augment
        self.normalize: bool = normalize

        self.files = make_dataset(root_dir, subset)
        if debug:
            self.files = self.files[:10]

        print(f">> Created {subset} dataset with {len(self)} images...")

    def __len__(self):
        return len(self.files)

    def histogram_equalization(self, img: Tensor) -> Tensor:
        # Convert to numpy for easier histogram manipulation
        img_np = img.cpu().numpy()
        
        # Calculate histogram
        hist, bin_edges = np.histogram(img_np, bins=256, range=(img_np.min(), img_np.max()))
        
        # Calculate cumulative distribution function (CDF)
        cdf = hist.cumsum()
        cdf_normalized = cdf * hist.max() / cdf.max()
        
        # Linear interpolation of CDF to map intensity values
        img_equalized = np.interp(img_np.flatten(), bin_edges[:-1], cdf_normalized)
        
        # Reshape back to original shape and convert to tensor
        img_equalized = torch.from_numpy(img_equalized.reshape(img_np.shape)).to(img.device)
        
        return img_equalized

    def normalize_img(self, img: Tensor) -> Tensor:
        # Apply histogram equalization
        equalized_img = self.histogram_equalization(img)

        # Normalize to [-1, 1] range
        normalized_img = 2 * (equalized_img - equalized_img.min()) / (equalized_img.max() - equalized_img.min()) - 1

        return normalized_img.float()


    def __getitem__(self, index) -> dict[str, Union[Tensor, int, str]]:
        img_path, gt_path = self.files[index]

        img: Tensor = self.img_transform(Image.open(img_path))
        gt: Tensor = self.gt_transform(Image.open(gt_path))

        if self.normalize:
            img = self.normalize_img(img)

        if self.augmentation:
            print("Augmenting...")
            target_patch_size = img.shape  # Assuming HxW image
            augmentation_pipeline = AugmentationPipeline(target_patch_size)
            batchgen = CTImageDataset(img, gt, 1, True)
            augmented_data_generator = augmentation_pipeline.create_data_generator(batchgen)
            augmented_batch = next(augmented_data_generator)
            img = augmented_batch["data"][0]
            gt = augmented_batch["seg"][0]
            
        _, W, H = img.shape
        K, _, _ = gt.shape
        assert gt.shape == (K, W, H)

        return {"images": img, "gts": gt, "stems": img_path.stem}
