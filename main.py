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
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from skimage.transform import resize
from torch.utils.data import DataLoader
from torchvision.transforms import v2

import wandb
from dataset import SliceDataset, VolumeDataset
from models import get_model
from utils.losses import get_loss
from utils.metrics import dice_batch, dice_coef
import pytorch_lightning as pl
from lightning.pytorch.loggers import WandbLogger
from utils.tensor_utils import (
    probs2class,
    probs2one_hot,
    save_images,
    tqdm_,
    print_args,
    set_seed,
)

"""
from nnunetv2.training.nnUNetTrainer import get_training_transforms

from typing import Tuple, Union, List
# nnunet augmenter, we only allow multithreaded augmentations for efficiency
from batchgenerators.dataloading.multi_threaded_augmenter import MultiThreadedAugmenter
from batchgenerators.dataloading.nondet_multi_threaded_augmenter import NonDetMultiThreadedAugmenter
from batchgenerators.dataloading.single_threaded_augmenter import SingleThreadedAugmenter
from batchgenerators.utilities.file_and_folder_operations import join, load_json, isfile, save_json, maybe_mkdir_p
from batchgeneratorsv2.helpers.scalar_type import RandomScalar
from batchgeneratorsv2.transforms.base.basic_transform import BasicTransform
from batchgeneratorsv2.transforms.intensity.brightness import MultiplicativeBrightnessTransform
from batchgeneratorsv2.transforms.intensity.contrast import ContrastTransform, BGContrast
from batchgeneratorsv2.transforms.intensity.gamma import GammaTransform
from batchgeneratorsv2.transforms.intensity.gaussian_noise import GaussianNoiseTransform
from batchgeneratorsv2.transforms.nnunet.random_binary_operator import ApplyRandomBinaryOperatorTransform
from batchgeneratorsv2.transforms.nnunet.remove_connected_components import \
    RemoveRandomConnectedComponentFromOneHotEncodingTransform
from batchgeneratorsv2.transforms.nnunet.seg_to_onehot import MoveSegAsOneHotToDataTransform
from batchgeneratorsv2.transforms.noise.gaussian_blur import GaussianBlurTransform
from batchgeneratorsv2.transforms.spatial.low_resolution import SimulateLowResolutionTransform
from batchgeneratorsv2.transforms.spatial.mirroring import MirrorTransform
from batchgeneratorsv2.transforms.spatial.spatial import SpatialTransform
from batchgeneratorsv2.transforms.utils.compose import ComposeTransforms
from batchgeneratorsv2.transforms.utils.deep_supervision_downsampling import DownsampleSegForDSTransform
from batchgeneratorsv2.transforms.utils.nnunet_masking import MaskImageTransform
from batchgeneratorsv2.transforms.utils.pseudo2d import Convert3DTo2DTransform, Convert2DTo3DTransform
from batchgeneratorsv2.transforms.utils.random import RandomTransform
from batchgeneratorsv2.transforms.utils.remove_label import RemoveLabelTansform
from batchgeneratorsv2.transforms.utils.seg_to_regions import ConvertSegmentationToRegionsTransform
"""

torch.set_float32_matmul_precision("medium")


class ReScale(v2.Transform):
    def __init__(self, K):
        self.scale = 1 / (255 / (K - 1)) if K != 5 else 1 / 63

    def __call__(self, img):
        return img * self.scale


class Class2OneHot(v2.Transform):
    def __init__(self, K):
        self.K = K

    def __call__(self, seg):
        b, *img_shape = seg.shape

        device = seg.device
        res = torch.zeros(
            (b, self.K, *img_shape), dtype=torch.int32, device=device
        ).scatter_(1, seg[:, None, ...], 1)
        return res[0]


def resize_and_save_slice(arr, K, X, Y, z, target_arr):
    resized_arr = resize(
        arr.cpu().numpy(),
        (K, X, Y),
        mode="constant",
        preserve_range=True,
        anti_aliasing=False,
        order=0,
    )
    target_arr[int(z), :, :, :] = resized_arr[...]
    return target_arr


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


class MyModel(pl.LightningModule):
    def __init__(self, args, batch_size, K, train_loader, val_loader):
        super().__init__()

        # Model part
        self.args = args
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.batch_size = batch_size
        self.K = K
        self.net = args.model(1, self.K)
        self.net.init_weights()
        self.loss_fn = get_loss(
            args.loss, self.K, include_background=args.include_background
        )

        # Dataset part
        self.root_dir: Path = Path(args.data_dir) / str(args.dataset)
        self.gt_shape = self._get_gt_shape()

        # Logging part
        self.log_loss_tra = torch.zeros((args.epochs, len(self.train_loader)))
        self.log_dice_tra = torch.zeros(
            (args.epochs, len(self.train_loader.dataset), self.K)
        )
        self.log_loss_val = torch.zeros((args.epochs, len(self.val_loader)))
        self.log_dice_val = torch.zeros(
            (args.epochs, len(self.val_loader.dataset), self.K)
        )
        self.log_dice_3d_tra = torch.zeros(
            (args.epochs, len(self.gt_shape["train"].keys()), self.K)
        )
        self.log_dice_3d_val = torch.zeros(
            (args.epochs, len(self.gt_shape["val"].keys()), self.K)
        )

        self.best_dice = 0

    def configure_optimizers(self):
        return torch.optim.Adam(
            filter(lambda x: x.requires_grad, self.net.parameters()), lr=self.args.lr
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
        pred_logits = self(img)
        pred_probs = F.softmax(1 * pred_logits, dim=1)
        pred_seg = probs2one_hot(pred_probs)
        loss = self.loss_fn(pred_probs, gt)
        self.log_loss_tra[self.current_epoch, batch_idx] = loss.detach()
        self.log_dice_tra[
            self.current_epoch, batch_idx : batch_idx + img.size(0), :
        ] = dice_coef(pred_seg, gt)

        self.log("train/loss", loss, on_step=True, prog_bar=True, logger=True)
        self.log_dict(
            {
                f"train/dice/{k}": self.log_dice_tra[
                    self.current_epoch, : batch_idx + img.size(0), k
                ].mean()
                for k in range(1, self.K)
            },
            prog_bar=True,
            logger=False,
            on_step=True,
        )
        return loss

    def validation_step(self, batch, batch_idx):
        img, gt = batch["images"], batch["gts"]
        pred_logits = self(img)
        pred_probs = F.softmax(1 * pred_logits, dim=1)
        pred_seg = probs2one_hot(pred_probs)
        loss = self.loss_fn(pred_probs, gt)
        self.log_loss_val[self.current_epoch, batch_idx] = loss.detach()
        self.log_dice_val[
            self.current_epoch, batch_idx : batch_idx + img.size(0), :
        ] = dice_coef(pred_seg, gt)

        self._prepare_3d_dice(batch["stems"], gt, pred_seg)

    def on_validation_epoch_end(self):
        log_dict = {
            "val/loss": self.log_loss_val[self.current_epoch].mean().detach(),
            "val/dice/total": self.log_dice_val[self.current_epoch, :, 1:]
            .mean()
            .detach(),
        }
        for k, v in self.get_dice_per_class(
            self.log_dice_val, self.K, self.current_epoch
        ).items():
            log_dict[f"val/dice/{k}"] = v
        if self.args.dataset == "SEGTHOR":
            for i, (patient_id, pred_vol) in tqdm_(
                enumerate(self.pred_volumes.items()), total=len(self.pred_volumes)
            ):
                gt_vol = torch.from_numpy(self.gt_volumes[patient_id]).to(self.device)
                pred_vol = torch.from_numpy(pred_vol).to(self.device)

                dice_3d = dice_batch(gt_vol, pred_vol)
                self.log_dice_3d_val[self.current_epoch, i, :] = dice_3d

            log_dict["val/dice_3d/total"] = (
                self.log_dice_3d_val[self.current_epoch, :, 1:].mean().detach()
            )
            # log_dict["val/dice_3d_class"] = self.get_dice_per_class(self.log_dice_3d_val, self.K, self.current_epoch)
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

    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return self.val_loader

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
        # Save model and weights in the specified results directory
        model_path = self.args.dest / "bestmodel.pkl"
        weights_path = self.args.dest / "bestweights.pt"
        
        torch.save(self.net, model_path)
        torch.save(self.net.state_dict(), weights_path)
        
        # TODO: this breaks the code, need to fix it
        # Save the artifact in the results directory
        # if self.args.wandb_project_name:
            # self.logger.save(str(self.args.dest / "bestweights.pt"))

## nnUnet functions - broken code below
"""        
def get_dataloaders(self):
        patch_size = self.configuration_manager.patch_size
        dim = len(patch_size)

        # needed for deep supervision: how much do we need to downscale the segmentation targets for the different
        # outputs?

        deep_supervision_scales = self._get_deep_supervision_scales()

        (
            rotation_for_DA,
            do_dummy_2d_data_aug,
            initial_patch_size,
            mirror_axes,
        ) = self.configure_rotation_dummyDA_mirroring_and_inital_patch_size()

        # training pipeline
        tr_transforms = self.get_training_transforms(
            patch_size, rotation_for_DA, deep_supervision_scales, mirror_axes, do_dummy_2d_data_aug,
            use_mask_for_norm=self.configuration_manager.use_mask_for_norm,
            is_cascaded=self.is_cascaded, foreground_labels=self.label_manager.foreground_labels,
            regions=self.label_manager.foreground_regions if self.label_manager.has_regions else None,
            ignore_label=self.label_manager.ignore_label)

        # validation pipeline
        val_transforms = self.get_validation_transforms(deep_supervision_scales,
                                                        is_cascaded=self.is_cascaded,
                                                        foreground_labels=self.label_manager.foreground_labels,
                                                        regions=self.label_manager.foreground_regions if
                                                        self.label_manager.has_regions else None,
                                                        ignore_label=self.label_manager.ignore_label)

        dataset_tr, dataset_val = self.get_tr_and_val_datasets()

        if dim == 2:
            dl_tr = nnUNetDataLoader2D(dataset_tr, self.batch_size,
                                       initial_patch_size,
                                       self.configuration_manager.patch_size,
                                       self.label_manager,
                                       oversample_foreground_percent=self.oversample_foreground_percent,
                                       sampling_probabilities=None, pad_sides=None, transforms=tr_transforms)
            dl_val = nnUNetDataLoader2D(dataset_val, self.batch_size,
                                        self.configuration_manager.patch_size,
                                        self.configuration_manager.patch_size,
                                        self.label_manager,
                                        oversample_foreground_percent=self.oversample_foreground_percent,
                                        sampling_probabilities=None, pad_sides=None, transforms=val_transforms)
        else:
            dl_tr = nnUNetDataLoader3D(dataset_tr, self.batch_size,
                                       initial_patch_size,
                                       self.configuration_manager.patch_size,
                                       self.label_manager,
                                       oversample_foreground_percent=self.oversample_foreground_percent,
                                       sampling_probabilities=None, pad_sides=None, transforms=tr_transforms)
            dl_val = nnUNetDataLoader3D(dataset_val, self.batch_size,
                                        self.configuration_manager.patch_size,
                                        self.configuration_manager.patch_size,
                                        self.label_manager,
                                        oversample_foreground_percent=self.oversample_foreground_percent,
                                        sampling_probabilities=None, pad_sides=None, transforms=val_transforms)

        allowed_num_processes = get_allowed_n_proc_DA()
        if allowed_num_processes == 0:
            mt_gen_train = SingleThreadedAugmenter(dl_tr, None)
            mt_gen_val = SingleThreadedAugmenter(dl_val, None)
        else:
            mt_gen_train = NonDetMultiThreadedAugmenter(data_loader=dl_tr, transform=None,
                                                        num_processes=allowed_num_processes,
                                                        num_cached=max(6, allowed_num_processes // 2), seeds=None,
                                                        pin_memory=self.device.type == 'cuda', wait_time=0.002)
            mt_gen_val = NonDetMultiThreadedAugmenter(data_loader=dl_val,
                                                      transform=None, num_processes=max(1, allowed_num_processes // 2),
                                                      num_cached=max(3, allowed_num_processes // 4), seeds=None,
                                                      pin_memory=self.device.type == 'cuda',
                                                      wait_time=0.002)
        # # let's get this party started
        _ = next(mt_gen_train)
        _ = next(mt_gen_val)
        return mt_gen_train, mt_gen_val

def get_training_transforms(
            patch_size: Union[np.ndarray, Tuple[int]],
            rotation_for_DA: RandomScalar,
            deep_supervision_scales: Union[List, Tuple, None],
            mirror_axes: Tuple[int, ...],
            do_dummy_2d_data_aug: bool,
            use_mask_for_norm: List[bool] = None,
            is_cascaded: bool = False,
            foreground_labels: Union[Tuple[int, ...], List[int]] = None,
            regions: List[Union[List[int], Tuple[int, ...], int]] = None,
            ignore_label: int = None,
    ) -> BasicTransform:
        transforms = []
        if do_dummy_2d_data_aug:
            ignore_axes = (0,)
            transforms.append(Convert3DTo2DTransform())
            patch_size_spatial = patch_size[1:]
        else:
            patch_size_spatial = patch_size
            ignore_axes = None
        transforms.append(
            SpatialTransform(
                patch_size_spatial, patch_center_dist_from_border=0, random_crop=False, p_elastic_deform=0,
                p_rotation=0.2,
                rotation=rotation_for_DA, p_scaling=0.2, scaling=(0.7, 1.4), p_synchronize_scaling_across_axes=1,
                bg_style_seg_sampling=False  # , mode_seg='nearest'
            )
        )

        if do_dummy_2d_data_aug:
            transforms.append(Convert2DTo3DTransform())

        transforms.append(RandomTransform(
            GaussianNoiseTransform(
                noise_variance=(0, 0.1),
                p_per_channel=1,
                synchronize_channels=True
            ), apply_probability=0.1
        ))
        transforms.append(RandomTransform(
            GaussianBlurTransform(
                blur_sigma=(0.5, 1.),
                synchronize_channels=False,
                synchronize_axes=False,
                p_per_channel=0.5, benchmark=True
            ), apply_probability=0.2
        ))
        transforms.append(RandomTransform(
            MultiplicativeBrightnessTransform(
                multiplier_range=BGContrast((0.75, 1.25)),
                synchronize_channels=False,
                p_per_channel=1
            ), apply_probability=0.15
        ))
        transforms.append(RandomTransform(
            ContrastTransform(
                contrast_range=BGContrast((0.75, 1.25)),
                preserve_range=True,
                synchronize_channels=False,
                p_per_channel=1
            ), apply_probability=0.15
        ))
        transforms.append(RandomTransform(
            SimulateLowResolutionTransform(
                scale=(0.5, 1),
                synchronize_channels=False,
                synchronize_axes=True,
                ignore_axes=ignore_axes,
                allowed_channels=None,
                p_per_channel=0.5
            ), apply_probability=0.25
        ))
        transforms.append(RandomTransform(
            GammaTransform(
                gamma=BGContrast((0.7, 1.5)),
                p_invert_image=1,
                synchronize_channels=False,
                p_per_channel=1,
                p_retain_stats=1
            ), apply_probability=0.1
        ))
        transforms.append(RandomTransform(
            GammaTransform(
                gamma=BGContrast((0.7, 1.5)),
                p_invert_image=0,
                synchronize_channels=False,
                p_per_channel=1,
                p_retain_stats=1
            ), apply_probability=0.3
        ))
        if mirror_axes is not None and len(mirror_axes) > 0:
            transforms.append(
                MirrorTransform(
                    allowed_axes=mirror_axes
                )
            )

        if use_mask_for_norm is not None and any(use_mask_for_norm):
            transforms.append(MaskImageTransform(
                apply_to_channels=[i for i in range(len(use_mask_for_norm)) if use_mask_for_norm[i]],
                channel_idx_in_seg=0,
                set_outside_to=0,
            ))

        transforms.append(
            RemoveLabelTansform(-1, 0)
        )
        if is_cascaded:
            assert foreground_labels is not None, 'We need foreground_labels for cascade augmentations'
            transforms.append(
                MoveSegAsOneHotToDataTransform(
                    source_channel_idx=1,
                    all_labels=foreground_labels,
                    remove_channel_from_source=True
                )
            )
            transforms.append(
                RandomTransform(
                    ApplyRandomBinaryOperatorTransform(
                        channel_idx=list(range(-len(foreground_labels), 0)),
                        strel_size=(1, 8),
                        p_per_label=1
                    ), apply_probability=0.4
                )
            )
            transforms.append(
                RandomTransform(
                    RemoveRandomConnectedComponentFromOneHotEncodingTransform(
                        channel_idx=list(range(-len(foreground_labels), 0)),
                        fill_with_other_class_p=0,
                        dont_do_if_covers_more_than_x_percent=0.15,
                        p_per_label=1
                    ), apply_probability=0.2
                )
            )

        if regions is not None:
            # the ignore label must also be converted
            transforms.append(
                ConvertSegmentationToRegionsTransform(
                    regions=list(regions) + [ignore_label] if ignore_label is not None else regions,
                    channel_in_seg=0
                )
            )

        if deep_supervision_scales is not None:
            transforms.append(DownsampleSegForDSTransform(ds_scales=deep_supervision_scales))

        return ComposeTransforms(transforms)
"""

import nibabel as nib
import numpy as np
import torch
from torch.utils.data import DataLoader
from pathlib import Path

def export_sample_as_nifti(image_tensor: torch.Tensor, gt_tensor: torch.Tensor, out_dir: Path, sample_name: str):
    """
    Function to save a sample image and ground truth tensor as NIfTI files.

    Args:
        image_tensor (torch.Tensor): The image tensor to save (shape: [D, H, W]).
        gt_tensor (torch.Tensor): The ground truth tensor to save (shape: [K, H, W]).
        out_dir (Path): Directory to save the NIfTI files.
        sample_name (str): Base name for the sample files.
    """
    # Ensure the output directory exists
    out_dir.mkdir(parents=True, exist_ok=True)

    # Convert tensors to numpy arrays for compatibility with nibabel
    image_np = image_tensor.numpy()
    gt_np = gt_tensor.numpy()

    # Create nibabel NIfTI images
    img_nii = nib.Nifti1Image(image_np, affine=np.eye(4))
    gt_nii = nib.Nifti1Image(gt_np, affine=np.eye(4))

    # Save the images
    nib.save(img_nii, out_dir / f"{sample_name}_image.nii.gz")
    nib.save(gt_nii, out_dir / f"{sample_name}_gt.nii.gz")
    print(f"Saved sample to {out_dir}")

def runTraining(args):
    print(f">>> Setting up to train on {args.dataset} with {args.mode}")

    K = args.datasets_params[args.dataset]["K"]
    root_dir = Path(args.data_dir) / args.dataset
    batch_size = args.datasets_params[args.dataset]["B"]
    args.dest.mkdir(parents=True, exist_ok=True)

    # Transforms
    img_transform = v2.Compose([v2.ToDtype(torch.float32, scale=True)])
    gt_transform = v2.Compose([ReScale(K), v2.ToDtype(torch.int64), Class2OneHot(K)])

    # Datasets and loaders
    if args.train_3d:
        train_set = VolumeDataset(
            "train",
            root_dir,
            img_transform=img_transform,
            gt_transform=gt_transform,
            debug=args.debug,
        )
        val_set = VolumeDataset(
            "val",
            root_dir,
            img_transform=img_transform,
            gt_transform=gt_transform,
            debug=args.debug,
        )
    else:
        train_set = SliceDataset(
            "train",
            root_dir,
            img_transform=img_transform,
            gt_transform=gt_transform,
            debug=args.debug,
        )
        val_set = SliceDataset(
            "val",
            root_dir,
            img_transform=img_transform,
            gt_transform=gt_transform,
            debug=args.debug,
        )
    train_loader = DataLoader(
        train_set, batch_size=batch_size, num_workers=args.num_workers, shuffle=True
    )

    val_loader = DataLoader(
        val_set, batch_size=batch_size, num_workers=args.num_workers, shuffle=False
    )
    
    # Export one sample from each loader to verify
    print(">> Exporting one sample from train loader...")
    train_sample = next(iter(train_loader))
    print("<< after sample", print(train_sample["volume"].shape))
    train_sample = train_sample[0]
    export_sample_as_nifti(
        train_sample["volume"] if args.train_3d else train_sample["images"],
        train_sample["gt"] if args.train_3d else train_sample["gts"],
        args.dest,
        sample_name=f"train_sample_{args.dataset}"
    )

    print(">> Exporting one sample from val loader...")
    val_sample = next(iter(val_loader))
    print("<< after sample", print(val_sample["volume"].shape))
    val_sample = val_sample[0]
    export_sample_as_nifti(
        val_sample["volume"] if args.train_3d else val_sample["images"],
        val_sample["gt"] if args.train_3d else val_sample["gts"],
        args.dest,
        sample_name=f"val_sample_{args.dataset}"
    )

    print(f">> Export completed. Check {args.dest} for NIfTI files.")
    
    assert False # Stop here to avoid training
    
    """
    assert args.num_workers < 4, "Num_workers must be greater than 4 because I haven't implemented single threaded augmenter yet\
    and I don't want to get too deep into the nnunet codebase to modify parameters."
    nnunet_train_augmenter = NonDetMultiThreadedAugmenter(data_loader=dl_tr, transform=None,
                                                        num_processes=args.num_workers,
                                                        num_cached=max(6, args.num_workers // 2), seeds=None,
                                                        pin_memory=args.cpu == False, wait_time=0.002)
     = NonDetMultiThreadedAugmenter(data_loader=dl_val,
                                                      transform=None, num_processes=max(1, args.num_workers // 2),
                                                      num_cached=max(3, args.num_workers // 4), seeds=None,
                                                      pin_memory=args.cpu == False,
                                                      wait_time=0.002)
    """
    print
    
    model = MyModel(args, batch_size, K, train_loader, val_loader)

    wandb_logger = (
        WandbLogger(project=args.wandb_project_name)
        if args.wandb_project_name
        else None
    )

    trainer = pl.Trainer(
        accelerator="cpu" if args.cpu else "auto",
        max_epochs=args.epochs,
        precision=args.precision,
        num_sanity_val_steps=0,  # Sanity check fails due to the 3D dice computation
        logger=wandb_logger,
    )
    trainer.fit(model, train_loader, val_loader)


def get_args():

    # Group 1: Dataset & Model configuration
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name",
        type=str,
        default="shallowCNN",
        choices=["shallowCNN", "ENet"],
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
        choices=["SEGTHOR", "TOY2"],
        help="Which dataset to use for the training.",
    )
    parser.add_argument(
        "--data_dir",
        type=Path,
        default="data",
        help="Path to get the GT scan, in order to get the correct number of slices",
    )
    parser.add_argument(
        "--train_3d",
        action="store_true",
        help="Whether to train a 3D model or a 2D model, default 2D."
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
        "--wandb_project_name",  # clean code dictates I leave this as "--wandb" but I'm not breaking people's flows yet
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
    }
    return args


def main():
    args = get_args()

    seed_everything(args.seed)
    if not args.wandb_project_name:
        setup_wandb(args)
    runTraining(args)


if __name__ == "__main__":
    main()
