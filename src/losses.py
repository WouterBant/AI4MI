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


import torch
from torch import nn
from torch import einsum
import argparse
from typing import Callable
from monai.losses import DiceLoss, GeneralizedDiceLoss, GeneralizedWassersteinDiceLoss

from utils import simplex, sset

def get_loss_fn(args: argparse.Namespace, K) -> Callable:
    """ Return the loss function class

    Args:
        args (argparse.Namespace): _description_
        K (_type_): Number of classes

    Returns:
        Callable: loss
    """    
    if args.loss == "ce":
        print(f"Using CrossEntropy loss with {K} classes")
        return CrossEntropy(
            idk=list(range(K))
        )  # Supervise both background and foreground
    elif args.loss == "dice":
        print(f"Using Dice loss")
        return DiceLoss(
            include_background=True
        )
    elif args.loss == "gdl":
        print(f"Using Generalized Dice loss")
        return GeneralizedDiceLoss(
            include_background=True
        )
        
    elif args.loss == "dce":
        print(f"Using Dice CrossEntropy loss")
        return DiceCrossEntropy(
            idk=list(range(K)),
            ce_lambda=args.ce_lambda
        )
        
    elif args.loss == "gwdl":
        #TODO: Define the distance matrix
        print(f"Using Generalized Wasserstein Dice loss")
        return GeneralizedWassersteinDiceLoss(
            dist_matrix=None
        )
        
    else:
        raise ValueError(f"Unsupported loss function: {args.loss}")
        


class CrossEntropy:
    def __init__(self, **kwargs):
        # Self.idk is used to filter out some classes of the target mask. Use fancy indexing
        self.idk = kwargs["idk"]
        print(f"Initialized {self.__class__.__name__} with {kwargs}")

    def __call__(self, pred_softmax, weak_target):
        assert pred_softmax.shape == weak_target.shape
        assert simplex(pred_softmax)
        assert sset(weak_target, [0, 1])

        log_p = (pred_softmax[:, self.idk, ...] + 1e-10).log()
        mask = weak_target[:, self.idk, ...].float()

        loss = -einsum("bkwh,bkwh->", mask, log_p)
        loss /= mask.sum() + 1e-10

        return loss


class PartialCrossEntropy(CrossEntropy):
    def __init__(self, **kwargs):
        super().__init__(idk=[1], **kwargs)


class DiceLoss(nn.Module):
    # TODO not confident about this implementation
    def __init__(self, n_classes):
        super(DiceLoss, self).__init__()
        self.n_classes = n_classes

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i  # * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob.unsqueeze(1))
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _dice_loss(self, score, target):
        target = target.float()
        smooth = 1e-5
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        return loss

    def forward(self, inputs, target, weight=None, softmax=False):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        # target = self._one_hot_encoder(target)
        if weight is None:
            weight = [1] * self.n_classes
        assert inputs.size() == target.size(), 'predict {} & target {} shape do not match'.format(inputs.size(),
                                                                                                  target.size())
        class_wise_dice = []
        loss = 0.0
        for i in range(0, self.n_classes):
            dice = self._dice_loss(inputs[:, i], target[:, i])
            class_wise_dice.append(1.0 - dice.item())
            loss += dice * weight[i]
        return loss / self.n_classes


# TODO add (generalized) dice loss https://github.com/wolny/pytorch-3dunet/blob/master/pytorch3dunet/unet3d/losses.py#L8
class DiceCrossEntropy(CrossEntropy):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.ce_lambda = kwargs["ce_lambda"]
        self.dice_loss = DiceLoss(include_background=True)

    def __call__(self, pred_softmax, weak_target):
        ce_loss = super().__call__(pred_softmax, weak_target)
        dice_loss = self.dice_loss(pred_softmax, weak_target)
        total_loss = self.ce_lambda * ce_loss + dice_loss
        return total_loss
