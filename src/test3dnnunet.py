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
from metrics3d import update_metrics_3D #, print_store_metrics
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
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    K: int = datasets_params[args.dataset]["K"]

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
        "test",  # TODO change this
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

    return (device, test_loader, id2nfiles, K)


def run_test(args):
    """
    Save the one-hot encoded predictions and the ground truth both in (num_files, num_classes, H, W) format for each patient
    After these are filled the metrics are calculated and saved in a csv file, the metrics are calculated for each class.
    Tensors are not stored.
    """

    print(f">>> Setting up to testing on {args.dataset} with {args.mode}")
    device, loader, id2nfiles, K = setup(args)
    print("ids", id2nfiles)

    # Preallocate the tensors in a dictionary
    predictions = {
        patient_id: torch.zeros((id2nfiles[patient_id], K, 256, 256), dtype=torch.uint8)  # K - 1 to skip the background class
        for patient_id in id2nfiles
    }
    ground_truths = {
        patient_id: torch.zeros((id2nfiles[patient_id], K, 256, 256), dtype=torch.uint8)
        for patient_id in id2nfiles
    }
    cur_idx = {patient_id: 0 for patient_id in id2nfiles}
    
    if args.mode=='partial':
        raise NotImplementedError("args.mode partial training should not be used")

    # Loop over the dataset to fill the tensors
    with torch.no_grad():
        for i, data in enumerate(tqdm.tqdm(loader)):
            img = data["images"].to(device)
            gt = data["gts"].to(device)
            stems = data["stems"]
            B = img.shape[0]
            # Metrics

            for b in range(B):
                patient_id = stems[b].split("_")[-2]
                ground_truths[patient_id][cur_idx[patient_id]] = gt[b]
                cur_idx[patient_id] += 1
    
    for file_path in args.folder.iterdir():
        if file_path.is_file():
            print("loading", file_path)
            data = np.load(file_path)
            array = data["probabilities"]
            print(array.shape)
            file_name = str(file_path).split("/")[-1]
            id_ = file_name.split("_")[-1].split(".")[0]
            predictions[id_] = probs2one_hot(F.interpolate(torch.from_numpy(array), size=(256, 256), mode='nearest').permute(1,0,3,2))
            print(predictions[id_].shape)

    # Initialize dataframe for storing metric results
    columns = ['patient_id', 'slice_name', 'class', 'metric_type', 'metric_value']
    metrics = pd.DataFrame(columns=columns)
    metric_types = ["dice", "sensitivity", "specificity", "iou", "hausdorff", "precision", "volumetric", "VOE"]

    # now compute the metrics for each patient
    for patient_id in predictions:
        pred = predictions[patient_id]
        gt = ground_truths[patient_id]
        print(pred.shape, gt.shape, "here")
        metrics = update_metrics_3D(metrics, pred, gt, patient_id, datasets_params[args.dataset]["names"], metric_types)  # TODO implement this
    
    # Save the metrics in pickle format
    save_directory = Path(f"results_metrics/{str(args.folder)}")
    print(save_directory)
    save_directory.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(str(save_directory) + f"/test_metrics.csv")

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
        "--dest",
        type=Path,
        default=None,
        help="Destination directory to save the results (predictions and weights).",
    )
    parser.add_argument(
        "--folder", 
        type=Path,
        default="predictions", 
        help="Path to folder with predictions"
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