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

# MPS issue: aten::max_unpool2d' not available for MPS devices
# Solution: set fallback to 1 before importing torch
import os

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

from lightning import seed_everything

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from lightning.pytorch import LightningModule, Trainer
from lightning.pytorch.loggers import WandbLogger
from PIL import Image
from torch.utils.data import DataLoader
from torchvision.transforms import v2

import wandb
from dataset import SliceDataset
from models import get_model, SegVolLightning
from utils.losses import get_loss
from utils.metrics import dice_batch, dice_coef
from utils.tensor_utils import (
    Class2OneHot,
    ReScale,
    print_args,
    probs2class,
    probs2one_hot,
    resize_and_save_slice,
    save_images,
    tqdm_,
)

torch.set_float32_matmul_precision("medium")


def setup_wandb(args):
    wandb.init(
        project=args.wandb_project_name,
        config={
            "epochs": args.epochs,
            "dataset": args.dataset,
            "learning_rate": args.lr,
            "batch_size": args.batch_size,
            "mode": args.mode,
            "seed": args.seed,
            "model": args.model_name,
            "loss": args.loss,
            "precision": args.precision,
            "include_background": args.include_background,
        },
    )


class MyModel(LightningModule):
    def __init__(self, args, batch_size, K):
        super().__init__()

        # Model part
        self.args = args
        self.batch_size = batch_size
        self.K = K
        self.net = args.model(1, self.K)
        self.net.init_weights()
        self.loss_fn = get_loss(
            args.loss, self.K, include_background=args.include_background
        )

        self.net = args.model(1, self.K)
        self.net.init_weights()

        # if True: # meant to be a flag
        #     self.net = torch.compile(self.net)

        self.loss_fn = get_loss(
            args.loss, self.K, include_background=args.include_background
        )

        # Dataset part
        self.batch_size: int = args.datasets_params[args.dataset]["B"]  # Batch size
        self.root_dir: Path = Path(args.data_dir) / str(args.dataset)

        args.dest.mkdir(parents=True, exist_ok=True)

    def configure_optimizers(self):
        return torch.optim.AdamW(
            filter(lambda x: x.requires_grad, self.net.parameters()), lr=self.args.lr
        )

    def on_fit_start(self) -> None:
        super().on_fit_start()

        self.log_loss_tra = torch.zeros(
            (self.args.epochs, len(self.train_dataloader()))
        )
        self.log_dice_tra = torch.zeros((self.args.epochs, len(self.train_set), self.K))
        self.log_loss_val = torch.zeros((self.args.epochs, len(self.val_dataloader())))
        self.log_dice_val = torch.zeros((self.args.epochs, len(self.val_set), self.K))

        self.best_dice = 0
        self.gt_shape = self._get_gt_shape()
        self.log_dice_3d_tra = torch.zeros(
            (self.args.epochs, len(self.gt_shape["train"].keys()), self.K)
        )
        self.log_dice_3d_val = torch.zeros(
            (self.args.epochs, len(self.gt_shape["val"].keys()), self.K)
        )

    def train_dataloader(self):
        self.train_set = SliceDataset(
            "train",
            self.root_dir,
            img_transform=v2.Compose([v2.ToDtype(torch.float32, scale=True)]),
            gt_transform=v2.Compose(
                [ReScale(self.K), v2.ToDtype(torch.int64), Class2OneHot(self.K)]
            ),
            debug=self.args.debug,
        )
        return DataLoader(
            self.train_set,
            batch_size=self.batch_size,
            num_workers=self.args.num_workers,
            shuffle=True,
        )

    def val_dataloader(self):
        self.val_set = SliceDataset(
            "val",
            self.root_dir,
            img_transform=v2.Compose([v2.ToDtype(torch.float32, scale=True)]),
            gt_transform=v2.Compose(
                [ReScale(self.K), v2.ToDtype(torch.int64), Class2OneHot(self.K)]
            ),
            debug=self.args.debug,
        )
        return DataLoader(
            self.val_set,
            batch_size=self.batch_size,
            num_workers=self.args.num_workers,
            shuffle=False,
        )

    def _get_gt_shape(self):
        # For each patient in dataset, get the ground truth volume shape
        self.gt_shape = {"train": {}, "val": {}}
        for split in self.gt_shape:
            directory = self.root_dir / split / "gt"
            split_patient_ids = set(x.stem.split("_")[1] for x in directory.iterdir())

            for patient_number in split_patient_ids:
                patient_id = f"Patient_{patient_number}"
                patients = list(directory.glob(patient_id + "*"))

                H, W = Image.open(patients[0]).size
                D = len(patients)
                self.gt_shape[split][patient_id] = (H, W, D)
        return self.gt_shape

    def on_validation_epoch_start(self) -> None:
        super().on_validation_epoch_start()

        self.gt_volumes = {
            p: np.zeros((Z, self.K, X, Y), dtype=np.uint8)
            for p, (X, Y, Z) in self.gt_shape["val"].items()
        }

        self.pred_volumes = {
            p: np.zeros((Z, self.K, X, Y), dtype=np.uint8)
            for p, (X, Y, Z) in self.gt_shape["val"].items()
        }

    def _prepare_3d_dice(self, batch_stems, gt, pred_seg):
        for i, seg_class in enumerate(pred_seg):
            stem = batch_stems[i]
            _, patient_n, z = stem.split("_")
            patient_id = f"Patient_{patient_n}"

            X, Y, _ = self.gt_shape["val"][patient_id]

            self.pred_volumes[patient_id] = resize_and_save_slice(
                seg_class, self.K, X, Y, z, self.pred_volumes[patient_id]
            )
            self.gt_volumes[patient_id] = resize_and_save_slice(
                gt[i], self.K, X, Y, z, self.gt_volumes[patient_id]
            )

    def forward(self, x):
        # Sanity tests to see we loaded and encoded the data correctly
        return self.net(x)

    def training_step(self, batch, batch_idx):
        img, gt = batch["images"], batch["gts"]
        B = img.size(0)
        pred_logits = self(img)
        pred_probs = F.softmax(self.args.temperature * pred_logits, dim=1)
        pred_seg = probs2one_hot(pred_probs)
        loss = self.loss_fn(pred_probs, gt)
        self.log_loss_tra[self.current_epoch, batch_idx] = loss.detach()

        dice_scores = dice_coef(pred_seg, gt)

        start_idx = batch_idx * self.batch_size
        end_idx = start_idx + B
        self.log_dice_tra[self.current_epoch, start_idx:end_idx, :] = dice_scores

        for k in range(1, self.K):
            class_dice = dice_scores[:, k].mean().item()
            self.log(f"train/dice_class_{k}", class_dice, on_step=True, prog_bar=True)

        self.log("train/loss", loss, on_step=True, prog_bar=True, logger=True)
        self.log_dict(
            {
                f"train/dice/{k}": self.log_dice_tra[
                    self.current_epoch, :end_idx, k
                ].mean()
                for k in range(1, self.K)
            },
            prog_bar=True,
            logger=True,
        )
        return loss

    def validation_step(self, batch, batch_idx):
        img, gt = batch["images"], batch["gts"]
        B = img.size(0)
        pred_logits = self(img)
        pred_probs = F.softmax(self.args.temperature * pred_logits, dim=1)
        pred_seg = probs2one_hot(pred_probs)
        loss = self.loss_fn(pred_probs, gt)
        dice_scores = dice_coef(pred_seg, gt)

        # Correct indexing using cumulative sample index
        start_idx = batch_idx * self.batch_size
        end_idx = start_idx + B

        self.log_dice_val[self.current_epoch, start_idx:end_idx, :] = dice_scores
        self.log_loss_val[self.current_epoch, batch_idx] = loss.detach()
        self._prepare_3d_dice(batch["stems"], gt, pred_seg)

    def on_validation_epoch_end(self):
        val_dice_scores = self.log_dice_val[self.current_epoch, :, :]
        val_loss = self.log_loss_val[self.current_epoch].mean().item()
        val_dice_excl_bg = val_dice_scores[:, 1:].mean().item()

        print(
            f"Epoch {self.current_epoch} validation Dice (excluding background): {val_dice_excl_bg}"
        )
        for k in range(1, self.K):
            class_dice = val_dice_scores[:, k].mean().item()
            print(f"Class {k} Dice: {class_dice}")

        # log_dict = {
        #     "val/loss": val_loss,
        #     "val/dice/total": val_dice_excl_bg,
        # }
        # for k in range(1, self.K):
        #     log_dict[f"val/dice/class_{k}"] = val_dice_scores[:, k].mean().item()

        log_dict = {
            "val/loss": val_loss,
            "val/dice/total": val_dice_excl_bg,
        }
        for k, v in self.get_dice_per_class(
            self.log_dice_val, self.K, self.current_epoch
        ).items():
            log_dict[f"val/dice/{k}"] = v
        if self.args.dataset == "SEGTHOR" and self.current_epoch != 0:
            print(len(self.pred_volumes), self.current_epoch)
            for i, (patient_id, pred_vol) in tqdm_(
                enumerate(self.pred_volumes.items()), total=len(self.pred_volumes)
            ):
                gt_vol = torch.from_numpy(self.gt_volumes[patient_id]).to(self.device)
                pred_vol = torch.from_numpy(pred_vol).to(self.device)
                dice_3d = dice_batch(gt_vol, pred_vol)
                self.log_dice_3d_val[self.current_epoch, i, :] = dice_3d

            log_dict["val/dice_3d/total"] = (
                self.log_dice_3d_val[self.current_epoch, :, 1:].mean().item()
            )
            for k, v in self.get_dice_per_class(
                self.log_dice_3d_val, self.K, self.current_epoch
            ).items():
                log_dict[f"val/dice_3d/{k}"] = v

        self.log_dict(log_dict)

        current_dice = self.log_dice_val[self.current_epoch, :, 1:].mean().detach()
        if current_dice > self.best_dice:
            self.best_dice = current_dice
            self.save_model()

        super().on_validation_epoch_end()

    def get_dice_per_class(self, log, K, e):
        if self.args.dataset == "SEGTHOR":
            class_names = [
                (1, "background"),
                (2, "esophagus"),
                (3, "heart"),
                (4, "trachea"),
                (5, "aorta"),
            ]
            dice_per_class = {
                f"dice_{k}_{n}": log[e, :, k - 1].mean().item() for k, n in class_names
            }
        else:
            dice_per_class = {
                f"dice_{k}": log[e, :, k].mean().item() for k in range(1, K)
            }
        return dice_per_class

    def save_model(self):
        torch.save(self.net, self.args.dest / "bestmodel.pkl")
        torch.save(self.net.state_dict(), self.args.dest / "bestweights.pt")
        # if self.args.wandb_project_name:
        #     self.logger.save(str(self.args.dest / "bestweights.pt"))
        # Save model and weights in the specified results directory


def runTraining(args):
    print(f">>> Setting up to train on {args.dataset} with {args.mode}")

    K = args.datasets_params[args.dataset]["K"]
    batch_size = args.datasets_params[args.dataset]["B"]

    # Datasets and loaders
    if args.dataset == "segthor_train":
        model = SegVolLightning(args, batch_size, K)

    else:
        model = MyModel(args, batch_size, K)

    wandb_logger = (
        WandbLogger(project=args.wandb_project_name)
        if args.wandb_project_name
        else None
    )
    trainer = Trainer(
        accelerator="cpu" if args.cpu else "auto",
        max_epochs=args.epochs,
        precision=args.precision,
        num_sanity_val_steps=1,  # Sanity check fails due to the 3D dice computation
        logger=wandb_logger,
        log_every_n_steps=5,
        # limit_train_batches=2
    )

    trainer.fit(model)


def get_args():
    # Group 1: Dataset & Model configuration
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name",
        type=str.lower,
        default="shallowcnn",
        choices=["shallowcnn", "enet", "udbrnet", "segvol"],
        help="Model to use for training",
    )
    parser.add_argument(
        "--mode",
        default="full",
        choices=["partial", "full"],
        help="Whether to supervise all the classes ('full') or, "
        "only a subset of them ('partial').",
    )
    parser.add_argument(
        "--dataset",
        default="SEGTHOR",
        choices=["SEGTHOR", "TOY2", "segthor_train"],
        help="Which dataset to use for the training.",
    )
    parser.add_argument(
        "--data_dir",
        type=Path,
        default="data",
        help="Path to get the GT scan, in order to get the correct number of slices",
    )

    # Group 2: Training parameters
    parser.add_argument("--epochs", default=25, type=int)
    parser.add_argument("--batch_size", default=8, type=int)
    parser.add_argument("--temperature", default=1, type=float)
    parser.add_argument(
        "--lr", type=float, default=0.0005, help="Learning rate for the optimizer."
    )
    parser.add_argument(
        "--loss",
        choices=["ce", "dice", "dicece", "dicefocal", "ce_torch"],
        default="dicefocal",
        help="Loss function to use for training.",
    )
    parser.add_argument(
        "--include_background",
        action="store_true",
        help="Whether to include the background class in the loss computation.",
    )
    parser.add_argument(
        "--seed", default=42, type=int, help="Seed to use for reproducibility."
    )
    parser.add_argument(
        "--precision",
        default=32,
        type=str,
        choices=[
            "bf16",
            "bf16-mixed",
            "bf16-true",
            "16",
            "16-mixed",
            "16-true",
            "32",
            "64",
        ],
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Force the code to run on CPU, even if a GPU is available.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=0,
        help="Number of subprocesses to use for data loading. "
        "Default 0 to avoid pickle lambda error.",
    )

    # Group 3: Output directory
    parser.add_argument(
        "--dest",
        type=Path,
        default=None,
        help="Destination directory to save the results (predictions and weights).",
    )

    # Group 4: Debugging and logging settings
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Keep only a fraction (10 samples) of the datasets, "
        "to test the logic around epochs and logging easily.",
    )
    parser.add_argument(
        "--wandb_project_name",
        type=str,
        help="Project wandb will be logging run to.",
    )

    args = parser.parse_args()

    # If dest not provided, create one
    if args.dest is None:
        # CE: 'args.mode = full'
        # Other: 'args.mode = partial'
        args.dest = Path(f"results/{args.dataset}/{args.mode}/{args.model_name}")

    # Model selection
    args.model = get_model(args.model_name)
    print_args(args)

    args.datasets_params = {
        "TOY2": {"K": 2, "B": args.batch_size},
        "SEGTHOR": {"K": 5, "B": args.batch_size},
        "segthor_train": {"K": 5, "B": 1},
    }
    return args


def main():
    args = get_args()
    print(args)

    seed_everything(args.seed, verbose=False)
    if not args.wandb_project_name:
        setup_wandb(args)
    runTraining(args)


if __name__ == "__main__":
    main()
