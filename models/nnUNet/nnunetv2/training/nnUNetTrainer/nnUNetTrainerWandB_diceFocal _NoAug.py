import json
import torch
import wandb
import numpy as np
from typing import Union, Tuple, List
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from batchgenerators.utilities.file_and_folder_operations import join
from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper
from nnunetv2.training.loss.dice_loss import DC_and_Focal_loss

from batchgeneratorsv2.helpers.scalar_type import RandomScalar
from batchgeneratorsv2.transforms.base.basic_transform import BasicTransform


class nnUNetTrainerWandB_diceFocal_NoAug(nnUNetTrainer):
    def __init__(
        self,
        plans: dict,
        configuration: str,
        fold: int,
        dataset_json: dict,
        unpack_dataset: bool = True,
        device: torch.device = torch.device("cuda"),
    ):
        super().__init__(
            plans, configuration, fold, dataset_json, unpack_dataset, device
        )
        self.num_epochs: int = 70

        wandb.init(
            project="nnUNet",
            config={
                "epochs": self.num_epochs,
                "dataset": self.plans_manager.dataset_name,
                "learning_rate": self.initial_lr,
                "batch_size": self.batch_size,
                "model": "nnUNet",
                "loss": "DiceFocal",
                "precision": "fp16" if self.grad_scaler is not None else "fp32",
                "configuration": self.configuration_name,
                "fold": self.fold,
            },
        )
    
    def initialize(self):
        # Call to load the dataset
        self.dataset_tr, self.dataset_val = self.load_dataset()

        # Print training split
        print("Training split:")
        for patient in self.dataset_tr:
            print(patient)

        # Print validation split
        print("Validation split:")
        for patient in self.dataset_val:
            print(patient)

    def _build_loss(self):

        loss = DC_and_Focal_loss(
            {
                "batch_dice": self.configuration_manager.batch_dice,
                "smooth": 1e-5,
                # "do_bg": False,
                "do_bg": True,
            },
            {"alpha": 0.5, "gamma": 2, "smooth": 1e-5},
        )

        if self.enable_deep_supervision:
            deep_supervision_scales = self._get_deep_supervision_scales()

            # we give each output a weight which decreases exponentially (division by 2) as the resolution decreases
            # this gives higher resolution outputs more weight in the loss
            weights = np.array(
                [1 / (2**i) for i in range(len(deep_supervision_scales))]
            )
            weights[-1] = 0

            # we don't use the lowest 2 outputs. Normalize weights so that they sum to 1
            weights = weights / weights.sum()
            # now wrap the loss
            loss = DeepSupervisionWrapper(loss, weights)
        return loss

    @staticmethod
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
        return nnUNetTrainer.get_validation_transforms(
            deep_supervision_scales,
            is_cascaded,
            foreground_labels,
            regions,
            ignore_label,
        )

    def configure_rotation_dummyDA_mirroring_and_inital_patch_size(self):
        # we need to disable mirroring here so that no mirroring will be applied in inference!
        rotation_for_DA, do_dummy_2d_data_aug, _, _ = (
            super().configure_rotation_dummyDA_mirroring_and_inital_patch_size()
        )
        mirror_axes = None
        self.inference_allowed_mirroring_axes = None
        initial_patch_size = self.configuration_manager.patch_size
        return rotation_for_DA, do_dummy_2d_data_aug, initial_patch_size, mirror_axes

    def on_train_epoch_start(self):
        super().on_train_epoch_start()
        wandb.log(
            {"learning_rate": self.optimizer.param_groups[0]["lr"]},
            step=self.current_epoch,
        )

    def on_train_epoch_end(self, train_outputs: List[dict]):
        super().on_train_epoch_end(train_outputs)
        wandb.log(
            {"train_loss": self.logger.my_fantastic_logging["train_losses"][-1]},
            step=self.current_epoch,
        )

    def on_validation_epoch_end(self, val_outputs: List[dict]):
        super().on_validation_epoch_end(val_outputs)
        wandb.log(
            {
                "val_loss": self.logger.my_fantastic_logging["val_losses"][-1],
                "mean_dice": self.logger.my_fantastic_logging["ema_fg_dice"][-1],
            },
            step=self.current_epoch,
        )

        # Log dice per class/region
        dice_per_class = self.logger.my_fantastic_logging["dice_per_class_or_region"][
            -1
        ]
        for i, dice in enumerate(dice_per_class):
            wandb.log({f"dice_class_{i}": dice}, step=self.current_epoch)

    def on_epoch_end(self):
        super().on_epoch_end()
        wandb.log(
            {
                "epoch": self.current_epoch,
                "epoch_time": self.logger.my_fantastic_logging["epoch_end_timestamps"][
                    -1
                ]
                - self.logger.my_fantastic_logging["epoch_start_timestamps"][-1],
            },
            step=self.current_epoch,
        )

    def on_train_end(self):
        super().on_train_end()

    def perform_actual_validation(self, save_probabilities: bool = False):
        super().perform_actual_validation(save_probabilities)
        if self.local_rank == 0:

            with open(join(self.output_folder, "validation", "summary.json"), "r") as f:
                metrics = json.load(f)
            wandb.save(join(self.output_folder, "validation", "summary.json"))
            wandb.log(
                {
                    "final_mean_dice": metrics["foreground_mean"]["Dice"],
                    "final_mean_iou": metrics["foreground_mean"]["IoU"],
                }
            )

    def save_checkpoint(self, filename: str) -> None:
        super().save_checkpoint(filename)
        if self.local_rank == 0:
            wandb.save(filename)
