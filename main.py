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

import os
# MPS issue: aten::max_unpool2d' not available for MPS devices
# Solution: set fallback to 1 before importing torch
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import argparse
import warnings
from typing import Any
from pathlib import Path
from pprint import pprint
from operator import itemgetter
from shutil import copytree, rmtree

import torch
import numpy as np
import torch.nn.functional as F
from torch import nn, Tensor
from torchvision import transforms
from torch.utils.data import DataLoader

from dataset import SliceDataset
from models.Network_wrapper import get_model
from utils.losses import CrossEntropy
from utils.metrics import dice_coef
from utils.tensor_utils import *


def setup(args) -> tuple[nn.Module, Any, Any, DataLoader, DataLoader, int]:
    
    # Seed & Logging part
    if args.seed is not None: set_seed(args.seed)
    if not args.disable_wandb: setup_wandb(args)

    # Model part
    device = get_device(args.gpu)
    K: int = args.datasets_params[args.dataset]["K"]
    net = args.model(1, K)
    net.init_weights()
    net.to(device)
    
    # Optimizer part
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)

    # Dataset part
    B: int = args.datasets_params[args.dataset]["B"] # Batch size
    root_dir = Path("data") / args.dataset

    # Transforms
    img_transform = transforms.Compose(
        [
            lambda img: img.convert("L"),
            lambda img: np.array(img)[np.newaxis, ...],
            lambda nd: nd / 255,  # max <= 1
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

    # Datasets and loaders
    train_set = SliceDataset(
        "train",
        root_dir,
        img_transform=img_transform,
        gt_transform=gt_transform,
        debug=args.debug,
    )
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
        loss_fn = CrossEntropy(idk=[0, 1, 3, 4])  # Do not supervise the heart (class 2)
    else:
        raise ValueError(args.mode, args.dataset)

    # Notice one has the length of the _loader_, and the other one of the _dataset_
    log_loss_tra: Tensor = torch.zeros((args.epochs, len(train_loader)))
    log_dice_tra: Tensor = torch.zeros((args.epochs, len(train_loader.dataset), K))
    log_loss_val: Tensor = torch.zeros((args.epochs, len(val_loader)))
    log_dice_val: Tensor = torch.zeros((args.epochs, len(val_loader.dataset), K))

    best_dice: float = 0

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

                    loss = loss_fn(pred_probs, gt)
                    log_loss[e, i] = (
                        loss.item()
                    )  # One loss value per batch (averaged in the loss)

                    if opt:  # Only for training
                        loss.backward()
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
                
                # Log the metrics after each 'e' epoch 
                if not args.disable_wandb:
                    wandb.log({
                        f"{m}_loss": log_loss[e].mean().item(),
                        f"{m}_dice": log_dice[e, :, 1:].mean().item(),
                        f"{m}_dice_per_class": {f"dice_{k}": log_dice[e, :, k].mean().item() for k in range(1, K)}
                    })

        # I save it at each epochs, in case the code crashes or I decide to stop it early
        np.save(args.dest / "loss_tra.npy", log_loss_tra)
        np.save(args.dest / "dice_tra.npy", log_dice_tra)
        np.save(args.dest / "loss_val.npy", log_loss_val)
        np.save(args.dest / "dice_val.npy", log_dice_val)

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

            # Log model checkpoint
            wandb.save(str(args.dest / "bestmodel.pkl"))
            wandb.save(str(args.dest / "bestweights.pt"))

def get_args():

    # Group 1: Dataset & Model configuration
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset", 
        default="TOY2", 
        choices=["TOY2", "SEGTHOR"],
        help="Which dataset to use for the training."
    )
    parser.add_argument(
        '--model_name',
        type=str,
        default='shallowCNN',
        choices=['shallowCNN', 'ENet', 'UDBRNet'],
        help='Model to use for training'
    )
    parser.add_argument(
        "--mode", 
        default="full", 
        choices=["partial", "full"],
        help="Whether to supervise all the classes ('full') or, "
        "only a subset of them ('partial')."
    )

    # Group 2: Training parameters
    parser.add_argument(
        "--epochs", 
        default=25, 
        type=int,
        help="Number of epochs to train."
    )
    parser.add_argument(
        "--lr", 
        type=float, 
        default=0.0005,
        help="Learning rate for the optimizer."
    )
    parser.add_argument(
        "--num_workers", 
        type=int, 
        default=0, 
        help="Number of subprocesses to use for data loading. "
        "Default 0 to avoid pickle lambda error."
    )
    parser.add_argument(
        "--gpu", 
        action="store_true",
        help="Use the GPU if available, otherwise fall back to the CPU."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed for reproducibility."
    )

    # Group 3: Output directory
    parser.add_argument(
        "--dest",
        type=Path,
        required=True,
        help="Destination directory to save the results (predictions and weights)."
    )

    # Group 4: Debugging and logging settings
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Keep only a fraction (10 samples) of the datasets, "
        "to test the logic around epochs and logging easily."
    )
    parser.add_argument(
        '--disable_wandb',
        action='store_true',
        help='Use flag to disable wandb logging, i.e. for debugging.'
    )
    parser.add_argument(
        '--wandb_project_name',
        type=str,
        help='Project wandb will be logging run to.'
    )

    args = parser.parse_args()
    pprint(args)

    # Model selection
    args.model = get_model(args.model_name)

    # Dataset-specific parameters
    datasets_params = {
        # K = number of classes, B = batch size
        "TOY2":    {"K": 2, "B": 2},   
        "SEGTHOR": {"K": 5, "B": 8},
    }
    args.datasets_params = datasets_params

    return args


def main():
    args = get_args()
    runTraining(args)

if __name__ == "__main__":
    main()
