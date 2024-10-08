"""
Perform non-maximum suppression in 3D on predicted images
Since this is a post-processing technique, it's in a seperate file
"""

import argparse
import numpy as np
from skimage.measure import label
import nibabel as nib
import torch
import pandas as pd
from pathlib import Path
import os
import wandb

from utils.tensor_utils import tqdm_, split_per_class
from utils.metrics import dice_coef, dice_batch

class_names = ['background', 'esophagus', 'heart', 'trachea', 'aorta']

def main(args):
    # Select only files ending in .nii.gz
    paths = os.listdir(args.src)
    paths = [x for x in paths if '.nii.gz' in x]
    patient_ids = [x.split('_')[1][:2] for x in paths]
    patient_ids = ['01']
    print('Found patient ids:', patient_ids)

    # Load nib files
    pred_nibs = [nib.load(f'{args.src}/Patient_{x}.nii.gz') for x in patient_ids]
    gt_nibs = [nib.load(f'{args.gt_src}/Patient_{x}/GT.nii.gz') for x in patient_ids]

    # Load volumes
    pred_vols = [(torch.from_numpy(np.asarray(x.dataobj)) / 63).type(torch.uint8) for x in pred_nibs]
    gt_vols = [(torch.from_numpy(np.asarray(x.dataobj))).type(torch.uint8) for x in gt_nibs]

    # Split volumes per class
    p = [split_per_class(x) for x in pred_vols]
    g = [split_per_class(x) for x in gt_vols]
    g[0] = g[0].permute(0,1,3,2)
    save_vol(g[0], gt_nibs[0], '01_gt_transpose')
    quit()

    # Initialize dataframe to save metrics
    # methods = ['original', 'nms']
    # iterables = [methods, args.metrics]
    # index = pd.MultiIndex.from_product(iterables, names=['method', 'metric'])
    # result = pd.DataFrame('nan', index, class_names)

    # wandb.init(
    #     project='post-processing-test',
    #     config={
    #         'src' : args.src,
    #         'metrics' : args.metrics,
    #         'methods' : 'nms'
    #     })

    # Compute metrics before post-processing
    # print('Computing metrics...')
    # print(result)
    # d2, d3 = compute_metrics(p, g, args.metrics)
    # save_metrics(result, 'original', d2, d3)

    # Perform NMS
    for i, vol in enumerate(p):
        p[i] = nms(vol)

    # # Recompute metrics and save in datafrmae
    # d2, d3 = compute_metrics(p, g, args.metrics)
    # save_metrics(result, 'nms', d2, d3)

    # # Log metrics
    # log_dict = {m: {
    #     'results': result.loc[m].to_dict(),
    #     'step' : m} for m in methods}
    # wandb.log(log_dict['original'])
    # wandb.log(log_dict['nms']) 

    # Save preds in correct format


def compute_metrics(pred_vols, gt_vols, metrics):
    dice_2d, dice_3d = torch.zeros((2, len(pred_vols), 5))
    for i in tqdm_(range(len(pred_vols))):
        p, g = pred_vols[i].to('cuda'), gt_vols[i].to('cuda')

        # 2D Dice
        if 'dice_2d' in metrics:
            dice_2d[i, :] = torch.mean(dice_coef(g, p), axis=0)

        # 3D Dice
        if 'dice_3d' in metrics:
            dice_3d[i, :] = dice_batch(g, p)

    dice_2d = np.mean(dice_2d.cpu().numpy(), axis=0)
    dice_3d = np.mean(dice_3d.cpu().numpy(), axis=0)

    return dice_2d, dice_3d

def print_metric(name, log):
    print(f'{name} ---')
    for i, c_name in enumerate(class_names):
        print(f'Class {i+1} {c_name}: {round(log[i].item(),3)}')

def nms(vol):
    vol = vol.permute(1,0,2,3)

    # Split and label per class
    split_labeled = [label(x, connectivity=2) for x in vol]

    # Keep only the largest area per class
    _, Z, H, W = vol.shape
    new_background = np.zeros((Z, H, W))
    split_nms = [None]*5
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
    result = torch.from_numpy(result).type(torch.uint8)
    assert result.shape == vol.shape
    result = result.permute(1,0,2,3)
    return result

def save_vol(vol, nib_, patient_id, ):
    for k in range(5):
        vol[:, k] *= k * 63
    vol = torch.sum(vol, dim=1).type(torch.uint8)
    print(vol.shape)
    new_nib = nib.nifti1.Nifti1Image(
        vol, affine=nib_.affine, header=nib_.header
    )
    nib.save(new_nib, f'{args.dest}/Patient_{patient_id}_nms.nii.gz')

def save_metrics(df, method, d2, d3):
    df.loc[(method, 'dice_2d')] = d2
    df.loc[(method, 'dice_3d')] = d3 

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--src',
        type=str,
        help='Path to folder containing the volumes to post-process',
    )
    parser.add_argument(
        '--dest',
        type=Path,
        help='Path to folder to save processed volumes in.'
    )
    parser.add_argument(
        '--gt_src',
        type=Path,
        help='Path to folder containing ground truth volumes'
    )
    parser.add_argument(
        '--metrics',
        nargs='+',
        help='Which metrics to use. Options: "dice_2d", "dice_3d"'
    )
    args = parser.parse_args()

    if args.dest is None:
        args.dest = Path(f"{args.src}/nms")

    args.dest.mkdir(parents=True, exist_ok=True)

    main(args)
