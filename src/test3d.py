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
from metrics3d import update_metrics_3D  # , print_store_metrics
from crf_model import apply_crf
from collections import defaultdict
import pandas as pd

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
            image_size=512,
        )
        net = LoRA_Sam(sam, r=args.r)
    elif args.model == "ENet":
        net = datasets_params[args.dataset]["net"](1, K)
        net.init_weights()
    elif args.model == "SAM2UNet":
        from sam2unet_model import SAM2UNet

        datasets_params[args.dataset]["net"] = SAM2UNet
        net = datasets_params[args.dataset]["net"](args.hiera_path)

    if args.crf:
        net = apply_crf(net, args)

    if args.from_checkpoint:
        print(args.from_checkpoint)
        # Load the checkpoint
        checkpoint = torch.load(args.from_checkpoint, map_location=device)

        # If the checkpoint contains 'state_dict', use it, otherwise use the checkpoint directly
        state_dict = checkpoint.get("state_dict", checkpoint)

        # Get the model's current state_dict
        model_dict = net.state_dict()

        # Filter out keys with mismatched shapes
        filtered_dict = {
            k: v
            for k, v in state_dict.items()
            if k in model_dict and v.shape == model_dict[k].shape
        }

        # Display the parameters that are skipped
        skipped_params = [k for k in state_dict if k not in filtered_dict]
        if skipped_params:
            print(
                f"Skipped loading parameters with mismatched shapes: {skipped_params}"
            )

        # Update the model's state_dict with the filtered parameters
        model_dict.update(filtered_dict)

        # Load the updated state_dict into the model
        net.load_state_dict(model_dict, strict=True)

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

    id2nfiles = dict()
    for f in test_set.files:
        patient_id = str(f).split("_")[-2]
        id2nfiles[patient_id] = id2nfiles.get(patient_id, 0) + 1

    args.dest.mkdir(parents=True, exist_ok=True)

    return (net, device, test_loader, id2nfiles, K)


def run_test(args):
    """
    Save the one-hot encoded predictions and the ground truth both in (num_files, num_classes, H, W) format for each patient
    After these are filled the metrics are calculated and saved in a csv file, the metrics are calculated for each class.
    Tensors are not stored.
    """
    print(f">>> Setting up to testing on {args.dataset} with {args.mode}")
    net, device, loader, id2nfiles, K = setup(args)
    print("ids", id2nfiles)

    start_idx = 0 if args.include_background else 1

    # Preallocate the tensors in a dictionary
    predictions = {
        patient_id: torch.zeros(
            (id2nfiles[patient_id], K - start_idx, 256, 256), dtype=torch.uint8
        )  # K - 1 to skip the background class
        for patient_id in id2nfiles
    }
    ground_truths = {
        patient_id: torch.zeros(
            (id2nfiles[patient_id], K - start_idx, 256, 256), dtype=torch.uint8
        )
        for patient_id in id2nfiles
    }
    cur_idx = {patient_id: 0 for patient_id in id2nfiles}

    if args.mode == "partial":
        raise NotImplementedError("args.mode partial training should not be used")

    net.eval()
    # Loop over the dataset to fill the tensors
    with torch.no_grad():
        for i, data in enumerate(tqdm.tqdm(loader)):
            img = data["images"].to(device)
            gt = data["gts"].to(device)
            stems = data["stems"]
            B = img.shape[0]

            # Samed model
            if args.model == "samed" or args.model == "samed_fast":
                preds = net(img, multimask_output=True, image_size=512)
                pred_logits = preds["masks"]
                pred_probs = F.softmax(
                    1 * pred_logits, dim=1
                )  # 1 is the temperature parameter
            elif args.model == "SAM2UNet":
                pred_logits, _, _ = net(img)  # Get the primary output from the model
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

            for b in range(B):
                patient_id = stems[b].split("_")[-2]
                assert segmentation_prediction[b].shape == gt[b].shape
                predictions[patient_id][cur_idx[patient_id]] = segmentation_prediction[
                    b
                ][start_idx:]
                ground_truths[patient_id][cur_idx[patient_id]] = gt[b][start_idx:]
                cur_idx[patient_id] += 1

    # Initialize dataframe for storing metric results
    columns = ["patient_id", "slice_name", "class", "metric_type", "metric_value"]
    metrics = pd.DataFrame(columns=columns)
    metric_types = [
        "dice",
        "sensitivity",
        "specificity",
        "hausdorff",
        "iou",
        "precision",
        "volumetric",
        "VOE",
    ]

    # now compute the metrics for each patient
    for patient_id in predictions:
        pred = predictions[patient_id]
        gt = ground_truths[patient_id]
        metrics = update_metrics_3D(
            metrics,
            pred,
            gt,
            patient_id,
            datasets_params[args.dataset]["names"][start_idx:],
            metric_types,
        )

    # Save the metrics in pickle format
    save_directory = Path(
        f"results_metrics/{args.model}/metrics3d/{str(args.from_checkpoint)[:-3]}"
    )

    save_directory.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(str(save_directory) + f"/{args.model}_metrics.csv")


def convert_to_dict(d):
    if isinstance(d, defaultdict):
        # Recursively convert any nested defaultdicts
        return {k: convert_to_dict(v) for k, v in d.items()}
    else:
        return d


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
        choices=["samed", "samed_fast", "ENet", "SAM2UNet"],
        help="Model to use",
    )
    parser.add_argument(
        "--hiera_path",
        type=str,
        required=False,
        help="path to the sam2 pretrained hiera",
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
    parser.add_argument(
        "--include_background",
        action="store_true",
        help="Include the background class in the metrics",
    )
    parser.add_argument(
        "--dataset", default="SEGTHOR_MANUAL_SPLIT", choices=datasets_params.keys()
    )
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
    parser.add_argument("--crf", action="store_true", help="Apply CRF on the output")
    parser.add_argument(
        "--finetune_crf",
        action="store_true",
        help="Freeze the model and only train CRF and the last layer",
    )
    args = parser.parse_args()

    pprint(args)

    if args.batch_size is None:
        args.batch_size = datasets_params[args.dataset]["B"]

    if args.dest is None:
        try:
            args.dest = Path(
                f"results/{str(args.from_checkpoint).split("checkpoints/")[1].strip(".pt")}_{datetime.now().strftime("%Y-%m-%d")}"
            )
        except:
            args.dest = Path(
                f"results/{args.dataset}/{datetime.now().strftime("%Y-%m-%d")}"
            )

    run_test(args)


if __name__ == "__main__":
    main()
