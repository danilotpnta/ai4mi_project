import csv
import os
from collections import defaultdict
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from lightning.pytorch import LightningModule
from torch.utils.data import DataLoader
import wandb

from models.segvol.base import SegVolConfig
from models.segvol.lora_model import SegVolLoRA
from utils.dataset import VolumetricDataset
from utils.metrics import hd95_batch

# from utils.losses import get_loss


class SegVolLightning(LightningModule):
    def __init__(self, args, batch_size, K, **kwargs):
        super().__init__()
        self.args = args
        self.batch_size = batch_size
        self.K = K

        config = SegVolConfig(test_mode=False)
        self.model = SegVolLoRA(config)
        self.categories = ["background", "esophagus", "heart", "trachea", "aorta"]

        # NOTE: We must patch the source code of SegVol to change the loss function
        if args.loss == "dicefocal":
            print(">>Changed BCELoss to FocalLoss")
            from monai.losses.focal_loss import FocalLoss

            self.model.model.bce_loss = FocalLoss(include_background=False)

        # if True: # meant to be a flag to compile but it crashes on a lot of backends
        #     self.net = torch.compile(self.net)

        # Dataset part
        self.batch_size: int = args.datasets_params[args.dataset]["B"]  # Batch size
        self.root_dir: Path = Path(args.data_dir) / str(args.dataset)

        args.dest.mkdir(parents=True, exist_ok=True)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            filter(lambda x: x.requires_grad, self.parameters()),
            lr=self.args.lr,
            weight_decay=0.005,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, self.args.epochs
        )
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    def on_fit_start(self) -> None:
        super().on_fit_start()
        self.log_loss_tra = torch.zeros(self.args.epochs, len(self.train_dataloader()))
        self.log_dice_tra = torch.zeros(self.args.epochs, len(self.train_set), self.K)
        self.log_loss_val = torch.zeros(self.args.epochs, len(self.val_dataloader()))
        self.log_dice_val = torch.zeros(self.args.epochs, len(self.val_set), self.K)

        self.log_hd95_tra = torch.zeros(self.args.epochs, len(self.train_set), self.K)
        self.log_hd95_val = torch.zeros(self.args.epochs, len(self.val_set), self.K)
        self.best_dice = 0

    def train_dataloader(self):
        self.train_set = VolumetricDataset(
            self.root_dir / "train",
            ratio=0.8,
            processor=self.model.processor,
            num_classes=self.K,
            train=True,
            cache_size=0,
        )
        return DataLoader(
            self.train_set,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.args.num_workers,
            # pin_memory=True,
        )

    def val_dataloader(self):
        self.val_set = VolumetricDataset(
            self.root_dir / "train",
            ratio=0.8,
            processor=self.model.processor,
            num_classes=self.K,
            train=False,
            cache_size=40,
        )
        return DataLoader(
            self.val_set,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.args.num_workers,
            # pin_memory=True,
        )

    def on_validation_epoch_start(self) -> None:
        super().on_validation_epoch_start()
        self.eval()

    def forward(self, image, gt):
        raise NotImplementedError

    def on_train_epoch_start(self):
        super().on_train_epoch_start()
        self.train()

    def training_step(self, batch, batch_idx):
        img, gt = batch["image"], batch["label"]

        compound_loss = 0.0
        for k in range(1, self.K):
            text_label = self.categories[k]
            mask_label = gt[:, k]

            loss = self.model.forward_train(
                img, train_organs=text_label, train_labels=mask_label, modality="CT"
            )
            compound_loss += loss

        self.log_loss_tra[self.current_epoch, batch_idx] = compound_loss.detach()
        # self.log_dice_tra[
        #     self.current_epoch, batch_idx : batch_idx + img.size(0), :
        # ] = dice_coef(pred_seg, gt)

        self.log("train/loss", loss, prog_bar=True, logger=True)
        return compound_loss

    def validation_step(self, batch, batch_idx):
        img, gt = batch["image"], batch["label"]

        gt_bg = torch.zeros_like(gt[0,0])
        masks = []
        for k in range(1, self.K):
            # text prompt
            text_prompt = [self.categories[k]]

            # # point prompt
            # point_prompt, point_prompt_map = self.model.processor.point_prompt_b(batch['zoom_out_label'][0, k], device=self.device)   # inputs w/o batch dim, outputs w batch dim

            # # bbox prompt
            # bbox_prompt, bbox_prompt_map = self.model.processor.bbox_prompt_b(batch['zoom_out_label'][0, k], device=self.device)   # inputs w/o batch dim, outputs w batch dim

            logits_mask = self.model.forward_test(
                image=img,
                zoomed_image=batch["zoom_out_image"],
                # point_prompt_group=[point_prompt, point_prompt_map],
                # bbox_prompt_group=[bbox_prompt, bbox_prompt_map],
                text_prompt=text_prompt,
                use_zoom=True,
            )

            predict = torch.where(torch.sigmoid(logits_mask[0][0]) > 0.5, 1.0, 0.0).to(gt.dtype)

            self.log_dice_val[self.current_epoch, batch_idx, k] = self.model.processor.dice_score(logits_mask[0][0], gt[0, k]).detach()

            gt_bg |= predict
            masks.append(predict)

        gt_bg = (1-gt_bg)
        self.log_dice_val[self.current_epoch, batch_idx, 0] = self.model.processor.dice_score(gt_bg, gt[0,0]).detach()

        hd95_score = hd95_batch(
            gt, torch.stack([gt_bg]+masks).unsqueeze(0), include_background=True
        )

        self.log_hd95_val[self.current_epoch, batch_idx] = hd95_score.detach()


    def get_metric_per_class(self, log, K, e, metric="dice", pre="val/"):
        if "segthor" in self.args.dataset.lower():
            class_names = [
                (1, "background"),
                (2, "esophagus"),
                (3, "heart"),
                (4, "trachea"),
                (5, "aorta"),
            ]
            dice_per_class = {
                f"{pre}{metric}_{k}_{n}": log[e, :, k - 1].mean().item() for k, n in class_names
            }
        else:
            dice_per_class = {
                f"{pre}{metric}_{k}": log[e, :, k].mean().item() for k in range(1, K)
            }
        return dice_per_class


    def on_predict_epoch_start(self):
        self.pred_dice = defaultdict(list)
        self.pred_hd95 = defaultdict(list)

    def predict_step(self, batch):
        img, gt = batch["image"], batch["label"]
        self.eval()

        logits = []
        masks = []
        gt_bg = torch.zeros_like(gt[0,0])
        for k in range(1, self.K):
            # text prompt
            text_prompt = [self.categories[k]]

            # # point prompt
            # point_prompt, point_prompt_map = self.model.processor.point_prompt_b(batch['zoom_out_label'][0, k], device=self.device)   # inputs w/o batch dim, outputs w batch dim

            # # bbox prompt
            # bbox_prompt, bbox_prompt_map = self.model.processor.bbox_prompt_b(batch['zoom_out_label'][0, k], device=self.device)   # inputs w/o batch dim, outputs w batch dim

            logits_mask = self.model.forward_test(
                image=img,
                zoomed_image=batch["zoom_out_image"],
                # point_prompt_group=[point_prompt, point_prompt_map],
                # bbox_prompt_group=[bbox_prompt, bbox_prompt_map],
                text_prompt=text_prompt,
                use_zoom=True,
            )
            # Remove batch dim and mask dim
            logits.append(logits_mask[0][0])
    
            predict = torch.where(torch.sigmoid(logits_mask[0][0]) > 0.5, 1.0, 0.0).to(gt.dtype)
            self.pred_dice[f"{k}{self.categories[k]}"].append(
                self.model.processor.dice_score(logits_mask[0][0], gt[0, k])
                .detach()
                .cpu()
                .item()
            )

            gt_bg |= predict
            logits.append(logits_mask[0][0])
            masks.append(predict)

        gt_bg = (1-gt_bg)

        self.pred_dice[f"0/{self.categories[0]}"].append(self.model.processor.dice_score(gt_bg, gt[0, 0])
                .detach()
                .cpu()
                .item())

        hd95_score = hd95_batch(
            gt, torch.stack([gt_bg]+masks).unsqueeze(0), include_background=True
        ).detach().cpu().numpy()
        for i, score in enumerate(hd95_score):    
            self.pred_hd95[f"{i}/{self.categories[i]}"].append(score)

        ct_path = self.val_set.path / batch["stem"][0] / f"{batch['stem'][0]}.nii.gz"
        save_path = os.path.join(self.args.dest, f"{batch['stem'][0]}_pred.nii.gz")
        return self.save_preds(
            ct_path,
            save_path,
            logits,
            start_coord=batch["foreground_start_coord"][0],
            end_coord=batch["foreground_end_coord"][0],
        )

    def on_predict_epoch_end(self):
        save_path_dice = os.path.join(self.args.dest, "dice_pred.npy")
        save_path_hd95 = os.path.join(self.args.dest, "hd95_pred.npy")
        # Convert predict_dict to numpy array
        dice_scores = np.array([self.pred_dice[k] for k in sorted(self.pred_dice.keys())]).T
        hd95_scores = np.array([self.pred_hd95[k] for k in sorted(self.pred_hd95.keys())]).T

        print(dice_scores)
        print(hd95_scores)

        # Save to npy file
        np.save(save_path_dice, dice_scores)
        np.save(save_path_hd95, hd95_scores)
        print(f"Prediction Dice saved to {save_path_dice}")
        return save_path_dice

    @staticmethod
    def save_preds(
        ct_path,
        save_path,
        logits_mask: torch.Tensor | list[torch.Tensor],
        start_coord,
        end_coord,
    ):
        """
        Save the predicted segmentation mask to a NIfTI file.

        Args:
            ct_path (str): Path to the input CT scan NIfTI file.
            save_path (str): Path where the predicted segmentation mask will be saved.
            logits_mask (torch.Tensor): The predicted logits mask from the model.
            start_coord (list of int): The starting coordinates of the region of interest in the CT scan.
            end_coord (list of int): The ending coordinates of the region of interest in the CT scan.

        Returns:
            None
        """
        ct = nib.load(ct_path)

        # Contain in a list in order to collect multiple predictions
        if not isinstance(logits_mask, list):
            logits_mask = (logits_mask,)
        # Create a new tensor with the same shape as the CT
        preds_save = torch.zeros(ct.shape, device=logits_mask[0].device)
        # Change the start and end coordinates
        start_coord = reversed(start_coord)
        end_coord = reversed(end_coord)

        for i, mask in enumerate(logits_mask, start=1):
            # Change to (X, Y, Z) format
            mask = mask.transpose(-1, -3)

            # Fill the tensor with the predictions
            preds_save[
                start_coord[0] : end_coord[0],
                start_coord[1] : end_coord[1],
                start_coord[2] : end_coord[2],
            ] = torch.where(
                torch.sigmoid(mask) > 0.5,
                i,
                preds_save[
                    start_coord[0] : end_coord[0],
                    start_coord[1] : end_coord[1],
                    start_coord[2] : end_coord[2],
                ],
            )
        # Save the predictions
        preds_nii = nib.Nifti1Image(
            preds_save.cpu().numpy().astype(np.uint8),
            affine=ct.affine,
            header=ct.header,
        )
        nib.save(preds_nii, save_path)
        print("Saved predictions to", save_path)
        return save_path

    def on_validation_epoch_end(self):
        current_dice = self.log_dice_val[self.current_epoch, :, 1:].mean().detach()
        if current_dice > self.best_dice:
            self.best_dice = current_dice
            self.save_model()
            
        self.log_dict(self.get_metric_per_class(self.log_dice_val, self.K, self.current_epoch, "dice", "val/") | self.get_metric_per_class(self.log_hd95_val, self.K, self.current_epoch, "hd95", "val/"), logger=True, prog_bar=True, on_epoch=True)

        np.save(self.args.dest / "loss_tra.npy", self.log_loss_tra)
        np.save(self.args.dest / "dice_tra.npy", self.log_dice_tra)
        np.save(self.args.dest / "loss_val.npy", self.log_loss_val)
        np.save(self.args.dest / "dice_val.npy", self.log_dice_val)
        np.save(self.args.dest / "hd95_val.npy", self.log_hd95_val)
        super().on_validation_epoch_end()

    def save_model(self):
        torch.save(self.model, self.args.dest / "bestmodel.pkl")
        torch.save(self.model.state_dict(), self.args.dest / "bestweights.pt")
        # if self.args.wandb_project_name:
        #     self.logger.save(str(self.args.dest / "bestweights.pt"))
