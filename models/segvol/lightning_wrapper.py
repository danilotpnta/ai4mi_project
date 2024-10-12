import os
from lightning.pytorch import LightningModule
import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from torch.utils.data import DataLoader
from models.segvol.base import SegVolConfig
from models.segvol.lora_model import SegVolLoRA
from utils.dataset import VolumetricDataset

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

        if args.loss == "dicefocal":
            print(">>Changed BCELoss to FocalLoss")
            from monai.losses.focal_loss import FocalLoss

            self.model.model.bce_loss = FocalLoss(include_background=False)
        # if True: # meant to be a flag
        #     self.net = torch.compile(self.net)

        # self.loss_fn = get_loss(
        #     args.loss, self.K, include_background=args.include_background
        # )

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
        # TODO: Implement 3D DICE

        # self.log_dice_3d_tra = torch.zeros(
        #     (args.epochs, len(self.gt_shape["train"].keys()), self.K)
        # )
        # self.log_dice_3d_val = torch.zeros(
        #     (args.epochs, len(self.gt_shape["val"].keys()), self.K)
        # )
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

        # self.gt_volumes = {
        #     p: np.zeros((Z, self.K, X, Y), dtype=np.uint8)
        #     for p, (X, Y, Z) in self.gt_shape["val"].items()
        # }

        # self.pred_volumes = {
        #     p: np.zeros((Z, self.K, X, Y), dtype=np.uint8)
        #     for p, (X, Y, Z) in self.gt_shape["val"].items()
        # }

    def _prepare_3d_dice(self, batch_stems, gt, pred_seg):
        for i, seg_class in enumerate(pred_seg):
            stem = batch_stems[i]
            _, patient_n, z = stem.split("_")
            patient_id = f"Patient_{patient_n}"

            X, Y, _ = self.gt_shape["val"][patient_id]

            # self.pred_volumes[patient_id] = resize_and_save_slice(
            #     seg_class, self.K, X, Y, z, self.pred_volumes[patient_id]
            # )
            # self.gt_volumes[patient_id] = resize_and_save_slice(
            #     gt[i], self.K, X, Y, z, self.gt_volumes[patient_id]
            # )

    def forward(self, image, gt):
        # Sanity tests to see we loaded and encoded the data correctly
        raise NotImplementedError
        # assert image.shape[0] == 1, image.shape

        # for k in range(1, self.K):
        #     text_label = self.categories[k]
        #     mask_label = gt[:, k]

        # return self.net(x)

    def on_train_epoch_start(self):
        super().on_train_epoch_start()
        self.train()

    def training_step(self, batch, batch_idx):
        img, gt = batch["image"], batch["label"]

        compound_loss = 0.0
        for k in range(1, self.K):
            text_label = self.categories[k]
            mask_label = gt[:, k]

            # NOTE: We must patch the source code of SegVol to change the loss function
            loss = self.model.forward_train(
                img, train_organs=text_label, train_labels=mask_label, modality="CT"
            )
            compound_loss += loss

        self.log_loss_tra[self.current_epoch, batch_idx] = compound_loss
        # self.log_dice_tra[
        #     self.current_epoch, batch_idx : batch_idx + img.size(0), :
        # ] = dice_coef(pred_seg, gt)

        self.log("train/loss", loss, prog_bar=True, logger=True)
        return compound_loss

    def validation_step(self, batch, batch_idx):
        img, gt = batch["image"], batch["label"]

        dice = {}
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

            dice[f"val/dice/{text_prompt[0]}"] = self.model.processor.dice_score(
                logits_mask[0][0], gt[0, k], self.device
            )

        self.log_dict(dice, logger=True, prog_bar=True, on_epoch=True)
        # self._prepare_3d_dice(batch["stems"], gt, pred_seg)

    def predict_step(self, batch):
        img = batch["image"]
        self.eval()

        logits = []
        log_dict = {}

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

            log_dict[f"val/dice/{text_prompt[0]}"] = (
                self.model.processor.dice_score(
                    logits_mask[0][0], batch["label"][0, k], self.device
                ).detach().cpu().item()
            )

        print(log_dict) # NOTE: Can't log inside predict. FIXME: You can return outputs using the predict's return.


        ct_path = self.val_set.path / batch["stem"][0] / f"{batch['stem'][0]}.nii.gz"
        save_path = os.path.join(self.args.dest, f"{batch['stem'][0]}_pred.nii.gz")
        return self.save_preds(
            ct_path,
            save_path,
            logits,
            start_coord=batch["foreground_start_coord"][0],
            end_coord=batch["foreground_end_coord"][0],
        )

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

        for i, mask in enumerate(logits_mask, start=1):
            # Change to (Z, X, Y) format (# no idea why we do this when the output rn is XYZ and thats what Nifti also has)
            mask = mask.transpose(-1, -3)
            # Also change the start and end coordinates
            start_coord[-1], start_coord[-3] = start_coord[-3], start_coord[-1]
            end_coord[-1], end_coord[-3] = end_coord[-3], end_coord[-1]
            # Fill the tensor with the predictions
            preds_save[
                start_coord[0] : end_coord[0],
                start_coord[1] : end_coord[1],
                : # start_coord[2] : end_coord[2],
            ] = torch.where(
                torch.sigmoid(mask) > 0.5,
                i,
                preds_save[
                    start_coord[0] : end_coord[0],
                    start_coord[1] : end_coord[1],
                    : # start_coord[2] : end_coord[2],
                ],
            )
        # Save the predictions
        preds_nii = nib.Nifti1Image(
            preds_save.cpu().numpy().astype(np.uint8), affine=ct.affine, header=ct.header
        )
        nib.save(preds_nii, save_path)
        print("Saved predictions to", save_path)
        return save_path

    def on_validation_epoch_end(self):
        current_dice = self.log_dice_val[self.current_epoch, :, 1:].mean().detach()
        if current_dice > self.best_dice:
            self.best_dice = current_dice
            self.save_model()

        super().on_validation_epoch_end()

    def save_model(self):
        torch.save(self.model, self.args.dest / "bestmodel.pkl")
        torch.save(self.model.state_dict(), self.args.dest / "bestweights.pt")
        # if self.args.wandb_project_name:
        #     self.logger.save(str(self.args.dest / "bestweights.pt"))
