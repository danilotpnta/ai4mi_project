import json
import torch
import wandb
import numpy as np
from typing import List
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from batchgenerators.utilities.file_and_folder_operations import join

class nnUNetTrainerWandB_diceCE_doBg(nnUNetTrainer):
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
        self.num_epochs = 70
        
        wandb.init(
            project="nnUNet",
            config={
                "epochs": self.num_epochs,
                "dataset": self.plans_manager.dataset_name,
                "learning_rate": self.initial_lr,
                "batch_size": self.batch_size,
                "model": "nnUNet",
                "loss": "DiceCE",
                "precision": "fp16" if self.grad_scaler is not None else "fp32",
                "configuration": self.configuration_name,
                "fold": self.fold,
            },
        )

        
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
        dice_per_class = self.logger.my_fantastic_logging["dice_per_class_or_region"][-1]
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


