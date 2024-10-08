# Some command to generate the scripts for some models watch out, if trained with CRF, enable CRF flat 
# ALWAYS CHANGE THE CHECKPOINT PATH AND THE MODEL NAME
# src/test.py --dataset SEGTHOR --mode full --dest results/SEGTHOR --batch_size 5 --model ENet --num_workers 4 --crf --finetune_crf --from_checkpoint src/samed/checkpoints/sam_vit_b_01ec64.pth


# Script used to test the model and calculate the metrics

import argparse
from typing import Any
from pathlib import Path
from pprint import pprint
from operator import itemgetter
from datetime import datetime
import tqdm
import pickle

import torch
import numpy as np
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.nn as nn

from dataset import SliceDataset
from ShallowNet import shallowCNN
from ENet import ENet
from utils import (
    class2one_hot,
    probs2one_hot,
)
from metrics import update_metrics_2D, print_store_metrics
from crf_model import apply_crf

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

def setup(args):
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
    if args.model == "samed" or args.model == "samed_fast":
        if args.model == "samed_fast":
            from samed_fast.sam_lora import LoRA_Sam
            from samed_fast.segment_anything import sam_model_registry
        elif args.model == "samed":
            from samed.sam_lora import LoRA_Sam
            from samed.segment_anything import sam_model_registry

        sam, _ = sam_model_registry["vit_b"](
            checkpoint="src/samed/checkpoints/sam_vit_b_01ec64.pth",
            num_classes=K,
            pixel_mean=[0.0457, 0.0457, 0.0457],
            pixel_std=[1.0, 1.0, 1.0],
            image_size=512
        )
        net = LoRA_Sam(sam, r=args.r)
    elif args.model == "ENet":
        net = datasets_params[args.dataset]["net"](1, K)
        net.init_weights()
    
    if args.from_checkpoint:
        print(args.from_checkpoint)
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

    if args.crf:
        net = apply_crf(net, args)
    net.to(device)

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

    test_set = SliceDataset(
        "test",
        root_dir,
        img_transform=img_transform,
        gt_transform=gt_transform,
        debug=args.debug,
        normalize=args.normalize,
    )
    test_loader = DataLoader(
        test_set, batch_size=B, num_workers=args.num_workers, shuffle=False
    )

    n_patients = len(set([str(f).split("_")[1] for f in test_set.files]))

    args.dest.mkdir(parents=True, exist_ok=True)

    return (net, device, test_loader, n_patients, K)


def run_test(args):
    print(f">>> Setting up to testing on {args.dataset} with {args.mode}")
    net, device, loader, n_patients, K = setup(args)
    
    if args.mode=='partial':
        raise NotImplementedError("args.mode partial training should not be used")

    mode = "test"
    net.eval()
    metrics = {}
    metric_types = ["dice", "sensitivity", "specificity", "hausdorff", "iou", "precision", "volumetric", "VOE"]
    for metric_type in metric_types:
        metrics[metric_type] = torch.zeros((len(loader.dataset), K))
    
    data_count = 0
    
    # Loop over the dataset
    with torch.no_grad():
        for i, data in enumerate(tqdm.tqdm(loader)):
            img = data["images"].to(device)
            gt = data["gts"].to(device)
            stems = data["stems"]
            
            Batch_size, _, _, _ = img.shape
            
            # Samed model
            if args.model == "samed" or args.model == "samed_fast":
                    preds = net(img, multimask_output=True, image_size=512)
                    pred_logits = preds["masks"]
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
                metrics[metric_type][data_count:data_count+Batch_size, :] = update_metrics_2D(segmentation_prediction, gt, metric_type) 
            data_count += Batch_size
            
        # Save the metrics in pickle format
        save_directory = args.dest / args.model
        save_directory.mkdir(parents=True, exist_ok=True)
        with open(str(save_directory) + f"/{mode}_metrics.pkl", "wb") as f:
            pickle.dump(metrics, f)
            
        # Print and store the metrics
        print_store_metrics(metrics, str(save_directory))
                
def apply_crf(net, args):
    from crfseg.model import CRF
    
    if args.finetune_crf:
        # Freeze the provided model except the last layer
        for param in net.parameters():
            param.requires_grad = False
        layers = list(net.children())
        
        # Now unfreeze the last layer
        for param in layers[-1].parameters():
            param.requires_grad = True
                
    # Add the CRF layer to the model
    model = nn.Sequential(
        net,
        CRF(n_spatial_dims=2)
    )
    return model        
                
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--batch_size",
        type=int,
        help="Batch size",
        default=None,
    )
    parser.add_argument(
        "--model",
        required=True,
        choices=["samed", "samed_fast", "ENet", "nnUnet"],
        help="Model to use",
    )
    parser.add_argument(
        "--dest",
        type=Path,
        default=None,
        help="Destination directory to save the results (predictions and weights).",
    )
    parser.add_argument(
        "--r",
        type=int,
        default=6,
        help="The rank of the LoRa matrices.",
    )
    parser.add_argument("--dataset", default="SEGTHOR_MANUAL_SPLIT", choices=datasets_params.keys())
    parser.add_argument("--mode", default="full", choices=["partial", "full"])
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
        "--normalize",
        action="store_true",
        help="Normalize the input images",
    )
    parser.add_argument(
        "--crf", action="store_true", help="Apply CRF on the output"
    )
    parser.add_argument(
        "--finetune_crf", action="store_true", help="Freeze the model and only train CRF and the last layer"
    )
    args = parser.parse_args()

    pprint(args)

    if args.batch_size is None:
        args.batch_size = datasets_params[args.dataset]["B"]

    if args.dest is None:
        args.dest = Path(
            f"results/{args.dataset}/{datetime.now().strftime("%Y-%m-%d")}"
        )

    run_test(args)


if __name__ == "__main__":
    main()