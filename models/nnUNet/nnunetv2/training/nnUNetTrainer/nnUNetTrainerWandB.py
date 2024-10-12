from typing import List
import wandb
import numpy as np

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper
from nnunetv2.training.loss.dice_loss import DC_and_Focal_loss

class nnUNetTrainerWandB(nnUNetTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        wandb.init(project="nnUNet_SegTHOR", entity="your_wandb_username")
        wandb.init(
            project=args.wandb_project_name,
            config={
                "epochs": self.num_epochs, 
                "dataset": self.plans_manager.dataset_name,  
                "learning_rate": self.initial_lr,  
                "batch_size": self.batch_size,  
                "model": 'nnUNet',
                "loss": 'DiceFocal',  
                "precision": "fp16" if self.grad_scaler is not None else "fp32"  
            },
        )

    def _build_loss(self):
        loss = DC_and_Focal_loss({'batch_dice':self.batch_dice, 'smooth':1e-5,
        	'do_bg':False}, {'alpha':0.5, 'gamma':2, 'smooth':1e-5})

        if self.enable_deep_supervision:
            deep_supervision_scales = self._get_deep_supervision_scales()

            # we give each output a weight which decreases exponentially (division by 2) as the resolution decreases
            # this gives higher resolution outputs more weight in the loss
            weights = np.array([1 / (2 ** i) for i in range(len(deep_supervision_scales))])
            weights[-1] = 0

            # we don't use the lowest 2 outputs. Normalize weights so that they sum to 1
            weights = weights / weights.sum()
            # now wrap the loss
            loss = DeepSupervisionWrapper(loss, weights)
        return loss

    def on_train_epoch_start(self):
        super().on_train_epoch_start()
        wandb.log({"learning_rate": self.optimizer.param_groups[0]['lr'], "epoch": self.current_epoch})

    def train_step(self, batch: dict) -> dict:
        result = super().train_step(batch)
        wandb.log({"train_loss": result['loss'], "epoch": self.current_epoch})
        return result

    def on_validation_epoch_end(self, val_outputs: List[dict]):
        super().on_validation_epoch_end(val_outputs)
        mean_fg_dice = self.logger.my_fantastic_logging['dice_per_class_or_region'][-1]
        wandb.log({
            "val_loss": np.mean([o['loss'] for o in val_outputs]),
            "mean_fg_dice": mean_fg_dice,
            "epoch": self.current_epoch
        })

        for idx, batch in enumerate(self.dataloader_val):
            if idx < 5:  
                data = batch['data']
                output = self.network(data.to(self.device))
                wandb.log({
                    f"val_input_image_{idx}": wandb.Image(data[0].cpu().numpy()[0]),
                    f"val_predicted_segmentation_{idx}": wandb.Image(np.argmax(output.cpu().numpy(), axis=1)[0]),
                    f"val_ground_truth_{idx}": wandb.Image(batch['target'][0].cpu().numpy()[0])
                })

    def save_checkpoint(self, filename: str) -> None:
        super().save_checkpoint(filename)
        # Log the model checkpoint as a W&B artifact
        if self.local_rank == 0 and not self.disable_checkpointing:
            wandb.save(filename)

    def on_epoch_end(self):
        super().on_epoch_end()
        self.logger.plot_progress_png(self.output_folder)  
        wandb.log({"epoch": self.current_epoch})