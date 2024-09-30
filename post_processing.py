"""
Perform non-maximum suppression in 3D on predicted images
Since this is a post-processing technique, it's in a seperate file
"""

import argparse
import numpy as np
from skimage.measure import label
import nibabel as nib
import torch
from pathlib import Path
import os

from utils.tensor_utils import tqdm_, one_hot, split_per_class
from utils.metrics import dice_coef, dice_batch


def main(args):
    # Select only files ending in .nii.gz
    patient_vols = os.listdir(args.src)
    patient_vols = [x for x in patient_vols if ('.nii.gz' in x)]
    patient_ids = [x.split('_')[1][:2] for x in patient_vols]
    print('Found patient ids:', patient_ids)

    # Load volumes
    pred_vols = [nib.load(f'{args.src}/{x}') for x in patient_vols]
    gt_vols = [nib.load(f'{args.gt_src}/Patient_{x}/GT.nii.gz') for x in patient_ids]

    # Compute metrics before post-processing
    print('Computing metrics...')
    compute_metrics(pred_vols, gt_vols)

    # for vol in tqdm_(pred_vols):
    #     result = nms()
    #     nib.save(result, f'{args.dest}/{vol_path}')

def compute_metrics(pred_vols, gt_vols, metrics=['dice_2d', 'dice_3d']):
    dice_2d, dice_3d = [], []
    for i, pred_vol in tqdm_(enumerate(pred_vols)):
        gt_vol = gt_vols[i]
        p, g = [torch.from_numpy(np.asarray(x.dataobj)) for x in [pred_vol, gt_vol]]

        p = (p / 63).type(torch.uint8)
        p, g = split_per_class(p), split_per_class(g)
        print(p.shape, g.shape)
        quit()

        # 2D Dice
        if 'dice_2d' in metrics:
            print(p.shape, g.shape)
            print(dice_coef(g, p).shape)

        # 3D Dice
        if 'dice_3d' in metrics:
            print(dice_batch(g, p).shape)

        quit()

def nms(vol_nib):
    vol = np.asarray(vol_nib.dataobj)

    # Split and label per class
    classes = [0,1,2,3,4]
    vol = vol / 63
    split = [np.where(vol == x, x, 0) for x in classes]
    split_labeled = [label(x, connectivity=2) for x in split]

    # Keep only the largest area per class
    split_nms = [np.zeros(vol.shape)] * 5
    for i, l in enumerate(split_labeled):
        indices, counts = np.unique(l, return_counts=True)

        if len(counts) > 1:
            indices, counts = indices[1:], counts[1:]
            most_occuring = indices[np.argmax(counts)]
            selected = np.where(l == most_occuring, 1, 0)

            # Put the nms processed volume in its right spot
            # and make sure the values match the class index
            split_nms[i] = selected * i * 63
    
    # Combine classes again and correct type
    result = np.sum(split_nms, axis=0)
    result = np.astype(result, np.uint8)
    assert result.shape == vol.shape
    result = nib.nifti1.Nifti1Image(result, affine=vol_nib.affine, header=vol_nib.header)
    return result

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--src',
        type=Path,
        help='Path to the folder containing stitched volumes to perform NMS on')
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
    args = parser.parse_args()

    if args.dest is None:
        args.dest = Path(f"{args.src}/nms")

    args.dest.mkdir(parents=True, exist_ok=True)

    main(args)
