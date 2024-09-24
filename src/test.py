# Script used to test the model and calculate the metrics

import argparse
import warnings
from typing import Any, Optional
from pathlib import Path
from pprint import pprint
from operator import itemgetter
from shutil import copytree, rmtree
from datetime import datetime
import wandb
import json
import tqdm
import pickle

import torch
import numpy as np
import torch.nn.functional as F
from torch import nn, Tensor
from torchvision import transforms
from torch.utils.data import DataLoader

from dataset import SliceDataset
from ShallowNet import shallowCNN
from scheduler import CosineWarmupScheduler
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
from losses import get_loss_fn, CrossEntropy, DiceLoss
from metrics import update_metrics, print_store_metrics
from adaptive_sampler import AdaptiveSampler

from samed.sam_lora import LoRA_Sam
from samed.segment_anything import sam_model_registry


torch.set_float32_matmul_precision("high")

datasets_params: dict[str, dict[str, Any]] = {}
# K for the number of classes
# Avoids the clases with C (often used for the number of Channel)
datasets_params["TOY2"] = {
    "K": 2,
    "net": shallowCNN,
    "B": 2,
    "names": ["background", "foreground"],
}
datasets_params["SEGTHOR"] = {
    "K": 5,
    "net": ENet,
    "B": 8,
    "names": ["Background", "Esophagus", "Heart", "Trachea", "Aorta"],
}
datasets_params["SEGTHOR_MANUAL_SPLIT"] = {
    "K": 5,
    "net": ENet,
    "B": 8,
    "names": ["Background", "Esophagus", "Heart", "Trachea", "Aorta"],
}


def setup(
    args,
) -> tuple[
    nn.Module, Any, Any, DataLoader, DataLoader, int, Optional[CosineWarmupScheduler], Optional[AdaptiveSampler]
]:
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
    if args.model == "samed":
        sam, _ = sam_model_registry["vit_b"](  # TODO check these arguments
            checkpoint="src/samed/checkpoints/sam_vit_b_01ec64.pth",
            num_classes=K,
            pixel_mean=[0.0457, 0.0457, 0.0457],
            pixel_std=[0.0723, 0.0723, 0.0723],
        )
        net = LoRA_Sam(sam, r=4)
    else:
        net = datasets_params[args.dataset]["net"](1, K)
        net.init_weights()  # TODO probably remove it and use the default one from pytorch
    
    if args.from_checkpoint:
        print(args.from_checkpoint)
        net = torch.compile(net)   # When the model was compiled when saved, it needs to be compiled again

        # Load the checkpoint
        checkpoint = torch.load(args.from_checkpoint, map_location=device)

        # If the checkpoint contains 'state_dict', use it, otherwise use the checkpoint directly
        state_dict = checkpoint.get('state_dict', checkpoint)

        # Get the model's current state_dict
        model_dict = net.state_dict()

        # Filter out keys with mismatched shapes
        filtered_dict = {k: v for k, v in state_dict.items() if k in model_dict and v.shape == model_dict[k].shape}

        # Display the parameters that are skipped
        skipped_params = [k for k in state_dict if k not in filtered_dict]
        if skipped_params:
            print(f"Skipped loading parameters with mismatched shapes: {skipped_params}")

        # Update the model's state_dict with the filtered parameters
        model_dict.update(filtered_dict)

        # Load the updated state_dict into the model
        net.load_state_dict(model_dict, strict=True)

    net.to(device)

    # Use torch.compile for kernel fusion to be faster on the gpu
    if gpu and args.epochs > 3:  # jit compilation takes too much time for few epochs
        print(">> Compiling the network for faster execution")
        net = torch.compile(net)

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

    sampler = None
    if args.use_sampler:
        sampler = AdaptiveSampler(train_set, B, args.epochs)
        train_loader = DataLoader(
            train_set, batch_size=B, num_workers=args.num_workers, sampler=sampler
        )
    else:
        train_loader = DataLoader(
            train_set, batch_size=B, num_workers=args.num_workers, shuffle=True
        )

    val_set = SliceDataset(
        "val",
        root_dir,
        img_transform=img_transform,
        gt_transform=gt_transform,
        debug=args.debug,
    )
    val_loader = DataLoader(
        val_set, batch_size=B, num_workers=args.num_workers, shuffle=False
    )
    if args.dataset == "SEGTHOR_MANUAL_SPLIT":
        test_set = SliceDataset(
            "test",
            root_dir,
            img_transform=img_transform,
            gt_transform=gt_transform,
            debug=args.debug,
        )
        test_loader = DataLoader(
            test_set, batch_size=B, num_workers=args.num_workers, shuffle=False
        )
    else: 
        test_loader = None

    scheduler = None
    if args.use_scheduler:
        total_steps = args.epochs * len(train_loader) / args.gradient_accumulation_steps
        warmup_steps = int(0.25 * total_steps)  # 25% of total steps for warmup
        scheduler = CosineWarmupScheduler(optimizer, args.lr, warmup_steps, total_steps)

    args.dest.mkdir(parents=True, exist_ok=True)

    return (net, optimizer, device, train_loader, val_loader, test_loader, K, scheduler, sampler)


def run_test(args):
    print(f">>> Setting up to train on {args.dataset} with {args.mode}")
    net, optimizer, device, train_loader, val_loader, test_loader, K, scheduler, sampler = setup(args)
    
    if args.mode=='partial':
        raise NotImplementedError("args.mode partial training should not be used")
    
    for mode in ['train', 'val', 'test']:
        if mode in ['train', 'val']:
            # currently only do metrics for test
            continue
        else:
            net.eval()
            loader = test_loader
            print(f">>> Running {mode} mode")
        
        
        context_manager = torch.no_grad
        
        metrics = {}
        metric_types = ["dice", "sensitivity", "specificity"]
        for metric_type in metric_types:
            metrics[metric_type] = torch.zeros((len(loader.dataset), K))
        
        data_count = 0
        
        # Loop over the dataset
        with context_manager():
            for i, data in enumerate(tqdm.tqdm(loader)):
                img = data["images"].to(device)
                gt = data["gts"].to(device)
                
                Batch_size, _, _, _ = img.shape
                
                # Samed model
                if args.model == "samed":
                        preds = net(img)
                        pred_logits = preds["low_res_logits"]
                        pred_probs = F.softmax(
                            1 * pred_logits, dim=1
                        )  # 1 is the temperature parameter

                # Other models
                else:
                    pred_logits = net(img)
                    pred_probs = F.softmax(
                        1 * pred_logits, dim=1
                    )  # 1 is the temperature parameter
                    
                # Metrics
                segmentation_prediction = probs2one_hot(pred_probs)
                for metric_type in metric_types:
                    metrics[metric_type][data_count:data_count+Batch_size, :] = update_metrics(segmentation_prediction, gt, metric_type) 

            # Save the metrics in pickle format
            with open(args.dest / f"{mode}_metrics.pkl", "wb") as f:
                pickle.dump(metrics, f)
                
            # Print and store the metrics
            print_store_metrics(metrics, args.dest / f"{mode}_metrics")
                
            
                
                
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--epochs", default=200, type=int)
    parser.add_argument("--lr", default=0.0005, type=float, help="Learning rate")
    parser.add_argument("--weight_decay", default=0.1, type=float, help="Weight decay")
    parser.add_argument(
        "--batch_size",
        type=int,
        help="Batch size",
        default=None,
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of steps to accumulate gradients over",
    )
    parser.add_argument(
        "--use_scheduler", action="store_true", help="Use CosineWarmupScheduler"
    )
    parser.add_argument(
        "--use_sampler", action="store_true", help="Use AdaptiveSampler"
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Model to use",
    )
    parser.add_argument(
        "--optimizer",
        default="adam",
        choices=["adam", "sgd", "adamw", "sgd-wd"],
        help="Optimizer to use",
    )
    parser.add_argument("--dataset", default="SEGTHOR", choices=datasets_params.keys())
    parser.add_argument("--mode", default="full", choices=["partial", "full"])
    parser.add_argument("--loss", default="ce", choices=["ce", "dice_monai", "gdl", "dce"])
    parser.add_argument("--ce_lambda", default=1.0, type=float)
    parser.add_argument(
        "--dest",
        type=Path,
        default=None,
        help="Destination directory to save the results (predictions and weights).",
    )
    parser.add_argument("--from_checkpoint", type=Path, default=None)
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument("--num_workers", type=int, default=4)
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
        args.dest = Path(
            f"results/{args.dataset}/{datetime.now().strftime("%Y-%m-%d")}"
        )

    run_test(args)


if __name__ == "__main__":
    main()

               
                
    




