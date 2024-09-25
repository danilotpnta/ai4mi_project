import os
from typing import Any
import torch
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt
import csv
import warnings
from shutil import copytree, rmtree
import numpy as np
from PIL import Image
from lightning.fabric import Fabric
import argparse
from Metrics import Evaluation_metric as evaluate
from Metrics.Evaluation_metric import get_SegThor_regions
import random
from monai.losses import DiceCELoss
from utils.metrics import dice_batch, dice_coef
from torch.nn.functional import one_hot
from torchvision import transforms
from models import get_model
from pathlib import Path
import torch.nn.functional as F
from dataset import SliceDataset  # If this is in another module, make sure the import is correct
import torch.nn as nn
from torch.utils.data import DataLoader
from utils.losses import get_loss
from utils.tensor_utils import (
    Dcm,
    class2one_hot,
    probs2class,
    probs2one_hot,
    save_images,
    tqdm_,
    print_args,
    set_seed
)
import wandb

def setup_wandb(args):
    wandb.init(
        project=args.wandb_project_name,
        config={
            "epochs": args.epoch_num,
            "dataset": args.dataset,
            "learning_rate": args.lr,
            "batch_size": args.datasets_params[args.dataset]["B"],
            "model": args.model_name,
        },
    )



def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        default="SEGTHOR",
        choices=["SEGTHOR", "TOY2"],
        help="Which dataset to use for the training.",
    )
    parser.add_argument(
        "--include_background",
        action="store_true",
        help="Whether to include the background class in the loss computation.",
    )
    parser.add_argument(
        "--loss",
        choices=["ce", "dice", "dicece", "dicefocal", "ce_torch"],
        default="dicefocal",
        help="Loss function to use for training.",
    )
    parser.add_argument(
        "--wandb_project_name",  # clean code dictates I leave this as "--wandb" but I'm not breaking people's flows yet
        type=str,
        help="Project wandb will be logging run to.",
    )
    parser.add_argument(
        "--data_dir",
        type=Path,
        default="data",
        help="Path to get the GT scan, in order to get the correct number of slices",
    )

    parser.add_argument(
        "--dest",
        type=Path,
        default=None,
        help="Destination directory to save the results (predictions and weights).",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Keep only a fraction (10 samples) of the datasets, "
        "to test the logic around epochs and logging easily.",
    )
    parser.add_argument("--lr", type=float, default=0.0005, help="Learning rate for the optimizer.")
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
    parser.add_argument("--model_name", type=str, default="UDBRNet")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument(
        "--num_workers",
        type=int,
        default=0,
        help="Number of subprocesses to use for data loading. "
        "Default 0 to avoid pickle lambda error.",
    )
    parser.add_argument('--temperature', default=1, type=float)
    parser.add_argument(
        "--seed", default=42, type=int, help="Seed to use for reproducibility."
    )

    args = parser.parse_args()

    if args.dest is None:
        args.dest = Path(f"results/{args.dataset}/{args.model_name}")
    
    args.model = get_model(args.model_name)
    
    args.datasets_params = {
        "TOY2": {"K": 2, "B": 8},  # Ensure batch size is handled here
        "SEGTHOR": {"K": 5, "B": 8},  # Batch size is set to 8 by default
    }

    return args

def out_directory_create(args):
    if not os.path.exists(os.path.join(args.dest, args.dataset)):
        os.mkdir(os.path.join(args.dest, args.dataset))

    if not os.path.exists(os.path.join(args.dest, args.dataset, args.model_name)):
        os.mkdir(os.path.join(args.dest, args.dataset, args.model_name))

    if not os.path.exists(os.path.join(args.dest, args.dataset, args.model_name, "Model")):
        os.mkdir(os.path.join(args.dest, args.dataset, args.model_name, "Model"))

    if not os.path.exists(os.path.join(args.dest, args.dataset, args.model_name, "Image")):
        os.mkdir(os.path.join(args.dest, args.dataset, args.model_name, "Image"))

    return os.path.join(args.dest, args.dataset, args.model_name)

def setup(args) -> tuple[nn.Module, Any, Any, DataLoader, DataLoader, int, Fabric]:
    
    # Seed and Fabric initialization
    set_seed(args.seed)
    fabric = Fabric(precision=args.precision, accelerator="cpu" if args.cpu else "auto")

    # Networks and scheduler
    device = fabric.device
    print(f">> Running on device '{device}'")
    K: int = args.datasets_params[args.dataset]["K"]
    net = args.model(1, K)
    net.init_weights()

    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
    net, optimizer = fabric.setup(net, optimizer)

    # Dataset part
    B: int = args.datasets_params[args.dataset]["B"]  # Batch size
    root_dir: Path = Path(args.data_dir) / str(args.dataset)

    # Transforms for images and ground truth
    img_transform = transforms.Compose(
        [
            lambda img: img.convert("L"), # Convert to grayscale
            transforms.PILToTensor(),     # Convert to tensor 
            lambda img: img / 255,        # Normalize to [0, 1]
        ]
    ) # img_tensor.shape = [1, H, W]
    gt_transform = transforms.Compose(
        [
            lambda img: np.array(img), # img values are in [0, 255]
            # For 2 classes, the classes are mapped to {0, 255}.
            # For 4 classes, the classes are mapped to {0, 85, 170, 255}.
            # For 6 classes, the classes are mapped to {0, 51, 102, 153, 204, 255}.
            lambda nd: nd / (255 / (K - 1)) if K != 5 else nd / 63,  # Normalization
            lambda nd: torch.from_numpy(nd).to(dtype=torch.int64, device=fabric.device)[
                None, ...
            ],  # Add one dimension to simulate batch
            lambda t: class2one_hot(t, K=K)[0], # Tensor: One-hot encoding [B, K, H, W]
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

    train_loader, val_loader = fabric.setup_dataloaders(train_loader, val_loader)

    args.dest.mkdir(parents=True, exist_ok=True)

    # For each patient in dataset, get the ground truth volume shape
    gt_shape = {"train": {}, "val": {}}
    for split in gt_shape:
        directory = root_dir / split / "gt"
        split_patient_ids = set(x.stem.split("_")[1] for x in directory.iterdir())

        for patient_number in split_patient_ids:
            patient_id = f"Patient_{patient_number}"
            patients = list(directory.glob(patient_id + "*"))

            H, W = Image.open(patients[0]).size
            D = len(patients)
            gt_shape[split][patient_id] = (H, W, D)

    return (net, optimizer, device, train_loader, val_loader, K, gt_shape, fabric)

def runTraining(args):
    print(f">>> Setting up to train on '{args.dataset}'")
    net, optimizer, device, train_loader, val_loader, K, gt_shape, fabric = setup(args)
    

    log_loss_tra: Tensor = torch.zeros((args.epochs, len(train_loader)))
    log_loss_val: Tensor = torch.zeros((args.epochs, len(val_loader)))
    log_dice_tra: Tensor = torch.zeros((args.epochs, len(train_loader.dataset), K)) 
    log_dice_val: Tensor = torch.zeros((args.epochs, len(val_loader.dataset), K))  

    best_dice: float = 0
    loss_fn = get_loss(args.loss, K, include_background=args.include_background)

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

            with cm():  # Train: dummy context manager, Val: torch.no_grad 
                j = 0
                tq_iter = tqdm_(enumerate(loader), total=len(loader), desc=desc)
                for i, data in tq_iter:
                    img = data["images"]
                    gt = data["gts"]
                    patient_id = data["patient_ids"]
                    slice_id = data["slice_ids"]
                    
                    if opt:  # So only for training
                        opt.zero_grad()

                    B, _, W, H = img.shape

                    # Get multiple outputs from the network
                    y_pred1, y_pred2, y_pred3, refined = net(img)

                    pred_probs = F.softmax(y_pred1 / args.temperature, dim=1)

                    # Metrics computation, not used for training
                    pred_seg = probs2one_hot(pred_probs)
                    log_dice[e, j : j + B, :] = dice_coef(
                        pred_seg, gt
                    )

                    loss = loss_fn(pred_probs, gt)
                    log_loss[e, i] = (
                        loss.item()
                    )  # One loss value per batch (averaged in the loss)

                    if opt:  # Only for training
                        fabric.backward(loss)
                        opt.step()

                    if m == 'val':
                            with warnings.catch_warnings():
                                warnings.filterwarnings('ignore', category=UserWarning)
                                predicted_class: Tensor = probs2class(pred_probs)
                                mult: int = 63 if K == 5 else (255 / (K - 1))
                                save_images(predicted_class * mult,
                                            data['stems'],
                                            args.dest / f"iter{e:03d}" / m)

                    j += B  # Keep in mind that _in theory_, each batch might have a different size
                    # For the DSC average: do not take the background class (0) into account:
                    postfix_dict: dict[str, str] = {"Dice": f"{log_dice[e, :j, 1:].mean():05.3f}",
                                                    "Loss": f"{log_loss[e, :i + 1].mean():5.2e}"}
                    if K > 2:
                        postfix_dict |= {f"Dice-{k}": f"{log_dice[e, :j, k].mean():05.3f}"
                                            for k in range(1, K)}
                    tq_iter.set_postfix(postfix_dict)

                
                log_dict = {
                    m: {
                        "loss": log_loss[e].mean().item(),
                        "dice": log_dice[e, :, 1:].mean().item(),
                        "dice_class": get_dice_per_class(args, log_dice, K, e),
                    }
                }

                if m == "val":
                    if args.wandb_project_name:
                        wandb.log(log_dict)

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
            if args.wandb_project_name:
                wandb.save(str(args.dest / "bestmodel.pkl"))
                wandb.save(str(args.dest / "bestweights.pt"))


    a = 1
    if a == 0:
        loaders = {"train": train_loader, "valid": val_loader}

        if args.dataset == "SEGTHOR":
            out_channel = 5

        dice_CE = DiceCELoss(softmax=True)

        output_directory = out_directory_create(args)

        epoch_done = 0
        for epoch in tqdm(range(epoch_done, args.epoch_num)):
            print("\n {epc} is running".format(epc=epoch))


            for m in ["train", 'valid']:
                if m == "train":
                    validation_predict = {}
                    validation_true = {}
                    net.train()
                else:
                    net.eval()

                for i, data in enumerate(loaders[m]):
                    x = data["images"]
                    y_true = data["gts"]
                    patient_id = data["patient_ids"]
                    slice_id = data["slice_ids"]

                    optimizer.zero_grad()

                    with torch.set_grad_enabled(m == "train"):
                        y_pred1, y_pred2, y_pred3, refined = net(x)
                        # y_true_one_hot = one_hot(y_true, num_classes=out_channel).permute(0, 3, 1, 2)
                        y_true_one_hot = y_true
                        loss1 = dice_CE(y_pred1, y_true_one_hot)
                        loss2 = dice_CE(y_pred2, y_true_one_hot)
                        loss3 = dice_CE(y_pred3, y_true_one_hot)
                        loss4 = dice_CE(refined, y_true_one_hot)
                        loss = loss1 + loss2 + loss3 + loss4

                        if(i%100 == 0):
                            print(f"Iteration: {i} Loss: {loss}")

                        if m == "valid":
                            patient_id = int(patient_id[0])
                            slice_id = int(slice_id[0])

                            refined = refined.argmax(dim=1)

                            if patient_id not in validation_predict.keys():
                                validation_predict[patient_id] = refined
                                validation_true[patient_id] = y_true
                            else:
                                validation_predict[patient_id] = torch.cat((validation_predict[patient_id], refined))
                                validation_true[patient_id] = torch.cat((validation_true[patient_id], y_true))


                        if m == "train":
                            loss.backward()
                            optimizer.step()


            print(f"Epoch: {epoch} Evaluation of {args.model_name} Validate ")

            # Initialize all_dice dictionary to store organ-wise dice scores
            all_dice = {organ: [] for organ in get_SegThor_regions().keys()}  # Ensure this is initialized

            prev_mean_dice = 0

            for patient in validation_predict.keys():
                print(f"Evaluating Patient {patient}")
                val_patient_pred = validation_predict[patient].squeeze()
                val_patient_true = validation_true[patient].squeeze()

                # Evaluate the dice score for each patient
                dice = evaluate.evaluate_case(val_patient_pred, val_patient_true, out_channel)

                # Split dice results by organ and store in respective lists
                organ_regions = get_SegThor_regions()

                for organ, index in organ_regions.items():
                    if index < dice.size(0):  # Ensure the index is within bounds of the dice tensor
                        organ_dice_score = dice[index].mean().item()  # Compute the mean dice score for this organ
                        all_dice[organ].append(organ_dice_score)
                    else:
                        print(f"Index {index} out of bounds for dice with shape {dice.shape}, skipping {organ}.")

            # Now, compute the average dice score for each organ across all patients
            mean_dice_scores = {organ: np.mean(scores) for organ, scores in all_dice.items()}

            # Convert the mean dice scores to a format suitable for CSV writing
            all = [mean_dice_scores[organ] for organ in get_SegThor_regions().keys()]

            # Write to CSV
            with open(os.path.join(output_directory, f"{args.model_name}_Validation_data.csv"), "a") as f:
                csv_writer = csv.writer(f, delimiter=',')
                csv_writer.writerow(all)

            # Define organ_dice as the overall mean of all organ Dice scores
            organ_dice = torch.tensor([mean_dice_scores[organ] for organ in get_SegThor_regions().keys()])

            # Now calculate curr_mean_dice
            curr_mean_dice = torch.mean(organ_dice)  # This computes the mean Dice score across all organs

            # Save the best model if the current mean dice is better
            model_name = f"{args.model_name}_epoch_{epoch}.pth"
            checkpoint_file = os.path.join(output_directory, "Model", model_name)

            if curr_mean_dice > prev_mean_dice:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': net.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss,
                }, checkpoint_file)
                prev_mean_dice = curr_mean_dice

def get_dice_per_class(args, log, K, e):
    if args.dataset == "SEGTHOR":
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
        dice_per_class = {f"dice_{k}": log[e, :, k].mean().item() for k in range(1, K)}

    return dice_per_class

def main():
    args = get_args()
    if args.wandb_project_name: 
        setup_wandb(args)
    runTraining(args)


if __name__ == "__main__":
    main()
