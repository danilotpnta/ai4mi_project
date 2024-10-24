import argparse
import json
import os
import warnings
from copy import deepcopy
from pathlib import Path

import kornia.morphology as morph
import nibabel as nib
import numpy as np
import torch
from lightning import seed_everything
from skimage.measure import label
from torchvision.transforms import GaussianBlur
from tqdm import tqdm

import wandb

warnings.filterwarnings("ignore")

from utils.metrics import dice_batch, hd95_batch
from utils.tensor_utils import one_hot, one_hot2class, resize_, split_per_class

class_names = ["esophagus", "heart", "trachea", "aorta"]


def main(args):
    seed_everything(42)

    methods = get_methods(args)
    print("Going to run:", methods)
    p, g, p_nibs, patient_ids = load_data(args, debug=False)

    # Initialize ways to save metrics
    wandb.init(
        project="post-processing-ablation-nnunet",
        config={
            "src": args.src,
            "methods": methods,
            "nms": args.nms,
            "gaussian_blur": args.gaussian_blur,
            "opening": args.opening,
        },
    )
    # Append name of run to dest folder to organize resulting volumes
    if args.dest is None:
        args.dest = Path(f"{args.src}/post_processing")
    args.dest = args.dest / wandb.run.name
    args.dest.mkdir(parents=True, exist_ok=True)

    for m in methods[1:]:
        print(f"Performing {m}")
        method_function = get_method_function(m)
        with tqdm(total=len(p)) as pbar:
            for i, p_vol in enumerate(p):
                vol_to_change = get_class_vol(m, p_vol)
                changed_vol = method_function(vol_to_change, m)
                p[i] = stitch_back(m, p_vol, changed_vol)
                save_vol(p[i], p_nibs[i], patient_ids[i], m, args)
                pbar.update()

    if not args.no_gt:
        compute_and_save_metrics(p, g, args.dest)

    wandb.finish()


def opening(vol, m):
    axes = m.split("_")[2:]
    for ax in axes:
        # Permute volume to match instructions
        vol = permute_vol(vol, ax).to("cuda")

        # Perform opening
        kernel = torch.ones(3, 3).to("cuda")
        opened_pred = morph.opening(vol, kernel).cpu().type(torch.int32)

        # Permute back
        vol = ensure_onehot(opened_pred)
        vol = permute_vol(vol, ax)

    return vol


# Define gaussian kernel globally to ensure the same kernel for each class
blur_size, sigma = 51, 1.5
gb_kernel = GaussianBlur((blur_size, blur_size), sigma=sigma)


def gaussianblur(vol, m):
    axes = m.split("_")[2:]
    for ax in axes:
        # Permute volume to match instructions
        vol = permute_vol(vol, ax)

        # Make class dimension the first dimension to avoid cross-class blurring
        to_blur = vol.permute(1, 0, 2, 3)

        # Perform Gaussian Blur
        blurred = gb_kernel(to_blur).permute(1, 0, 2, 3).cpu()

        # Permute back to original shape
        vol = ensure_onehot(blurred)
        vol = permute_vol(vol, ax)

    return vol


def nms(vol, ax):
    # Make sure class dimension is first
    vol = vol.permute(1, 0, 2, 3).cpu()

    # Split and label per class
    split_labeled = [label(x, connectivity=2) for x in vol]

    # Keep only the largest area per class
    _, Z, H, W = vol.shape
    new_background = np.zeros((Z, H, W))
    split_nms = [None] * len(vol)
    for i, l in enumerate(split_labeled):
        indices, counts = np.unique(l, return_counts=True)

        if len(counts) > 1:
            indices, counts = indices[1:], counts[1:]
            most_occuring = indices[np.argmax(counts)]
            selected = np.where(l == most_occuring, 1, 0)

            # Replace prediction with selected area
            split_nms[i] = selected

            # Save inverse of selected area as this should
            # become background in the final result
            new_background += vol[i].numpy() - selected
        else:
            split_nms[i] = l

    # Change the prediction of removed areas to background
    split_nms[0] = np.clip(split_nms[0] + new_background, 0, 1)

    # Combine classes again and correct type
    result = np.stack(split_nms)
    result = torch.from_numpy(result).type(vol.dtype)
    assert result.shape == vol.shape

    # Change back to original shape
    result = result.permute(1, 0, 2, 3)
    return result


def ensure_onehot(vol):
    """Ensures the volume is still onehot after post-processing
    by ensuring voxels without prediction become background
    and checking there are no voxels with an overlapping prediction"""

    # Determine which pixels have turned into background after blurring
    # and update the predicted background
    n_nonbackground = vol.shape[1] - 1
    zero_preds = torch.argwhere(torch.sum(vol, dim=1) < 1)
    new_preds = torch.hstack(
        [torch.ones(len(zero_preds), 1), torch.zeros(len(zero_preds), n_nonbackground)]
    )
    vol[zero_preds[:, 0], :, zero_preds[:, 1], zero_preds[:, 2]] = new_preds.type(
        vol.dtype
    )
    assert len(torch.argwhere(torch.sum(vol, dim=1) < 1)) == 0

    # Remove double predictions -- change to class
    double_preds = torch.argwhere(torch.sum(vol, dim=1) > 1)
    new_preds = torch.hstack(
        [
            torch.zeros(len(double_preds), 1),
            torch.ones(len(double_preds), n_nonbackground),
        ]
    )
    if len(double_preds) > 0:
        vol[double_preds[:, 0], :, double_preds[:, 1], double_preds[:, 2]] = (
            new_preds.type(vol.dtype)
        )
    assert len(torch.argwhere(torch.sum(vol, dim=1) > 1)) == 0

    assert one_hot(vol)
    return vol


def compute_and_save_metrics(pred_vols, gt_vols, dest):
    dice_3d, hd = torch.zeros((2, len(pred_vols), 5))
    for i in tqdm(range(len(pred_vols))):
        p, g = pred_vols[i].to("cuda"), gt_vols[i].to("cuda")
        _3d, _hd = compute_metrics(p, g)

        # 3D Dice
        dice_3d[i, :] = _3d

        # Hausdorff
        hd[i, :] = _hd

    # Save as numpy arrays for plotting
    with open(f"{dest}/dice_3d.npy", "wb") as f:
        np.save(f, dice_3d.cpu().numpy())
    with open(f"{dest}/hd.npy", "wb") as f:
        np.save(f, hd.cpu().numpy())

    # Take the mean for wandb logging
    dice_3d = np.mean(dice_3d.cpu().numpy(), axis=0)
    hd = np.mean(hd.cpu().numpy(), axis=0)

    # Remove background for wandb
    dice_3d, hd = dice_3d[1:], hd[1:]

    dice_3d_class = {name_class: val for name_class, val in zip(class_names, dice_3d)}
    hd_class = {name_class: val for name_class, val in zip(class_names, hd)}
    wandb.log(
        {
            "dice_3d": dice_3d_class,
            "hd": hd_class,
        }
    )


def get_class_vol(method_string, vol):
    """
    Returns the background + relevant class for the method requested.
    For example, '1_nms' will return the first and second dimensions of the volume,
    '1234_nms' will return the full volume.
    """
    classes = method_string.split("_")[0]

    # All classes selected means the full volume is considered
    if len(classes) == 4:
        return vol

    # Ensure we have one class otherwise
    elif len(classes) != 1:
        raise ValueError(f"Invalid selection of classes in {method_string}.")

    return vol[:, [0, int(classes)]]


def stitch_back(method_string, predicted_vol, changed_vol):
    """
    Replaces the dimensions of the predicted vol with the changed vol, based
    on the class indices in the method string
    """
    classes = method_string.split("_")[0]

    # All classes selected means the full volume is considered
    if len(classes) == 4:
        return predicted_vol

    # Ensure we have one class otherwise
    elif len(classes) != 1:
        raise ValueError(f"Invalid selection of classes in {method_string}.")

    # Replace new background and class
    predicted_vol[:, [0, int(classes)]] = changed_vol
    return predicted_vol


def compute_metrics(p, g):
    dice_3d = dice_batch(p, g)
    hd = hd95_batch(
        g[None, ...].permute(0, 2, 3, 4, 1),
        p[None, ...].permute(0, 2, 3, 4, 1),
        include_background=True,
    )
    return dice_3d, hd


def load_data(args, debug=True):
    """
    Expecting data in .nii.gz format. Prediction files should end in 'Patient_{patient_id}.nii.gz' and
    ground truth files should end in 'Patient_{patient_id}_GT.nii.gz'.
    Both the prediction and ground truth files should be containing only
    class labels, so the values [0,1,2,3,4].
    """
    # Select only files ending in .nii.gz
    paths = os.listdir(args.src)
    paths = [x for x in paths if ".nii.gz" in x]
    patient_ids = [x.split("_")[1][:2] for x in paths] if not debug else ["01"]
    print("Found patient ids:", patient_ids)

    # Load nib files
    pred_nibs = [nib.load(f"{args.src}/Patient_{x}.nii.gz") for x in patient_ids]

    gt_nibs = (
        []
        if args.no_gt
        else [nib.load(f"{args.gt_src}/Patient_{x}_GT.nii.gz") for x in patient_ids]
    )

    # Load volumes
    pred_vols = [
        (torch.from_numpy(np.asarray(x.dataobj))).type(torch.int64) for x in pred_nibs
    ]
    gt_vols = (
        []
        if args.no_gt
        else [
            (torch.from_numpy(np.asarray(x.dataobj))).type(torch.int64) for x in gt_nibs
        ]
    )

    # Split volumes per class
    print("Splitting volumes per class...")
    N = len(pred_vols)
    p, g = [None] * N, [None] * N
    for i in tqdm(range(N)):
        p[i] = split_per_class(pred_vols[i])
        if not args.no_gt:
            g[i] = split_per_class(gt_vols[i])

    return p, g, pred_nibs, patient_ids


def permute_vol(vol, ax):
    match ax:
        case "yx":
            return vol
        case "zx":
            return vol.permute(2, 1, 0, 3)
        case "yz":
            return vol.permute(3, 1, 2, 0)


def get_methods(args):
    methods = ["original"]

    # Default case: apply to all axes
    if args.class_config is None:
        if args.nms == "True":
            methods.append("1234_nms")
        if args.gaussian_blur != "None":
            methods.append(f"1234_gblur_{clean_method(args.gaussian_blur)}")
        if args.opening != "None":
            methods.append(f"1234_open_{clean_method(args.opening)}")

    # Class-specific case
    else:
        config = json.load(open(args.class_config, "r"))
        for class_index, method in config.items():
            match method[0]:
                case "NMS":
                    methods.append(f"{class_index}_nms")
                case "GaussianBlur":
                    methods.append(f"{class_index}_gblur_{clean_method(method[1])}")
                case "Opening":
                    methods.append(f"{class_index}_open_{clean_method(method[1])}")
                case "NMS+GaussianBlur":
                    methods.append(f"{class_index}_nms")
                    methods.append(f"{class_index}_gblur_{clean_method(method[1])}")
                case _:
                    raise ValueError(f"Method {method} not implemented.")
    return methods


def clean_method(m):
    m = m.replace("[", "").replace("]", "").replace("'", "")
    m = m.split(", ")
    for x in m:
        assert x in ["yx", "zx", "yz"], "Unsupported axes chosen."
    return "_".join(m)


def get_method_function(m):
    match m.split("_")[1:]:
        case ["nms", *_]:
            return nms
        case ["gblur", *_]:
            return gaussianblur
        case ["open", *_]:
            return opening


def save_vol(vol, nib_, patient_id, method, args):
    # Convert to class representation and correct order of dimensions
    target_vol = deepcopy(vol)
    target_vol = one_hot2class(target_vol, K=5).permute(1, 2, 0)

    # Convert to nifti and save
    H, W, Z = target_vol.shape
    target_vol = resize_(target_vol.cpu().numpy(), (H * 2, W * 2, Z))
    new_nib = nib.nifti1.Nifti1Image(target_vol, affine=nib_.affine, header=nib_.header)
    save_name = f"{args.dest}/Patient_{patient_id}_{method}.nii.gz"
    if args.no_gt:
        save_name = f"{args.dest}/Patient_{patient_id}.nii.gz"
    nib.save(new_nib, save_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--src",
        type=str,
        help="Path to folder containing the volumes to post-process",
    )
    parser.add_argument(
        "--dest", type=Path, help="Path to folder to save processed volumes in."
    )
    parser.add_argument(
        "--gt_src", type=Path, help="Path to folder containing ground truth volumes"
    )
    parser.add_argument(
        "--nms",
        type=str,
        default="False",
        help="Enable to apply Non-Maximum Suppression",
    )
    parser.add_argument(
        "--gaussian_blur",
        default="None",
        help='Enter dimensions to perform gaussian blur over. Multiple options can be selected. Options: ["yx", "zx", "yz"]',
    )
    parser.add_argument(
        "--opening",
        default="None",
        help='Enter dimensions to perform opening over. Multiple options can be selected. Options: ["yx", "zx", "yz"]',
    )
    parser.add_argument(
        "--class_config",
        type=Path,
        default=None,
        help="If enabled, will instead apply the postprocessing technique per class specified in this file, in json format.",
    )
    parser.add_argument(
        "--no_gt",
        action="store_true",
        default=False,
        help="Enable to ignore the ground truth, meaning no metric computation. Useful for test set predictions.",
    )
    args = parser.parse_args()

    main(args)
