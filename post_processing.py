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
from IPython.display import display

from utils.tensor_utils import tqdm_, split_per_class
from utils.metrics import dice_coef, dice_batch

class_names = ['background', 'esophagus', 'heart', 'trachea', 'aorta']

def main(args):
    # Select only files ending in .nii.gz
    paths = os.listdir(args.src)
    paths = [x for x in paths if '.nii.gz' in x]
    patient_ids = [x.split('_')[1][:2] for x in paths]
    patient_ids = ['01', '02']
    print('Found patient ids:', patient_ids)

    # Load nib files
    pred_nibs = [nib.load(f'{args.src}/{x}') for x in paths]
    gt_nibs = [nib.load(f'{args.gt_src}/Patient_{x}/GT.nii.gz') for x in patient_ids]

    # Load volumes
    pred_vols = [(torch.from_numpy(np.asarray(x.dataobj)) / 63).type(torch.uint8) for x in pred_nibs]
    gt_vols = [(torch.from_numpy(np.asarray(x.dataobj))).type(torch.uint8) for x in gt_nibs]

    # Split volumes per class
    p = [split_per_class(x) for x in pred_vols]
    g = [split_per_class(x) for x in gt_vols]

    # Initialize dataframe to save metrics
    iterables = [['original', 'nms'], class_names]
    index = pd.MultiIndex.from_product(iterables, names=["method", "class"])
    result = pd.DataFrame('nan', index, args.metrics)

    # Compute metrics before post-processing
    print('Computing metrics...')
    d2, d3 = compute_metrics(p, g, args.metrics)
    save_metrics(result, 'original', d2, d3)

    for i, vol in enumerate(p):
        p[i] = nms(vol)

    d2, d3 = compute_metrics(p, g, args.metrics)
    save_metrics(result, 'nms', d2, d3)

    print(result)

def compute_metrics(pred_vols, gt_vols, metrics):
    dice_2d, dice_3d = np.zeros((2, len(pred_vols), 5))
    for i in tqdm_(range(len(pred_vols))):
        p, g = pred_vols[i].to('cuda'), gt_vols[i].to('cuda')

        # 2D Dice
        if 'dice_2d' in metrics:
            dice_2d[i, :] = np.mean(dice_coef(g, p), dim=0)

        # 3D Dice
        if 'dice_3d' in metrics:
            dice_3d[i, :] = dice_batch(g, p)

    dice_2d = np.mean(dice_2d, dim=0)
    dice_3d = np.mean(dice_3d, dim=0)

    return dice_2d, dice_3d

    # if 'dice_2d' in metrics:
    #     print_metric('dice_2d', dice_2d)
    # if 'dice_3d' in metrics:
    #     print_metric('dice_3d', dice_3d)

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

def save_metrics(df, method, d2, d3):
    df.loc[method, 'dice_2d'] = d2
    df.loc[method, 'dice_3d'] = d3 

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
