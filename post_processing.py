import argparse
from copy import deepcopy
import numpy as np
from skimage.measure import label
from lightning import seed_everything
import kornia.morphology as morph
import nibabel as nib
import torch
from torchvision.transforms import GaussianBlur
from tqdm import tqdm
from pathlib import Path
import os
import wandb

from utils.tensor_utils import split_per_class, one_hot2class, resize_
from utils.metrics import dice_coef, dice_batch, hd95_batch

class_names = ['esophagus', 'heart', 'trachea', 'aorta']

def main(args):
    seed_everything(42)

    methods = get_methods(args)
    p, g, p_nibs, patient_ids = load_data(args, debug=False)

    # Initialize ways to save metrics
    wandb.init(
        project='post-processing-ablation',
        config={
            'src' : args.src,
            'methods' : methods
        })
    # Append name of run to dest folder to organize resulting volumes
    if args.dest is None:
        args.dest = Path(f"{args.src}/post_processing")
    args.dest = args.dest / wandb.run.name
    args.dest.mkdir(parents=True, exist_ok=True)

    # Compute metrics before post-processing
    print('Computing metrics...')
    compute_and_save_metrics(p, g, 'original')

    for m in methods[1:]:
        print(f'Performing {m}')
        method_function = get_method_function(m)
        with tqdm(total=len(p)) as pbar:
            for i, p_vol in enumerate(p):
                p[i] = method_function(p_vol, m)
                save_vol(p[i], p_nibs[i], patient_ids[i], m)
                pbar.update()
        
        compute_and_save_metrics(p, g, m)

    wandb.finish()
            
def opening(vol, m):
    axes = m.split('_')[1:]
    for ax in axes:
        # Permute volume to match instructions
        vol = permute_vol(vol, ax).to('cuda')

        # Perform opening
        kernel = torch.ones(3,3).to('cuda')
        opened_pred = morph.opening(vol, kernel).cpu().type(torch.int32)

        # Permute back
        vol = ensure_onehot(opened_pred)
        vol = permute_vol(vol, ax)

    return vol

def gaussianblur(vol, m):
    axes = m.split('_')[1:]
    for ax in axes:
        # Permute volume to match instructions
        vol = permute_vol(vol, ax)

        # Set gaussian blur parameters
        blur_size, sigma = 51, 1.5
        blur = GaussianBlur((blur_size,blur_size), sigma=sigma)

        # Make class dimension the first dimension to avoid cross-class blurring
        to_blur = vol.permute(1,0,2,3)

        # Perform Gaussian Blur
        blurred = blur(to_blur).permute(1,0,2,3).cpu()
        
        # Permute back to original shape
        vol = ensure_onehot(blurred)
        vol = permute_vol(vol, ax)

    return vol

def nms(vol, ax):
    # Make sure class dimension is first
    vol = vol.permute(1,0,2,3).cpu()

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
    result = torch.from_numpy(result).type(vol.dtype)
    assert result.shape == vol.shape

    # Change back to original shape
    result = result.permute(1,0,2,3)
    return result

def ensure_onehot(vol):
    '''Ensures the volume is still onehot after post-processing
    by ensuring voxels without prediction become background
    and checking there are no voxels with an overlapping prediction'''

    # Compute which pixels have turned into background after blurring
    # and update the predicted background
    zero_preds = torch.argwhere(torch.sum(vol, dim=1) < 1)
    new_preds = torch.hstack([
        torch.ones(len(zero_preds), 1), 
        torch.zeros(len(zero_preds), 4)
        ])
    vol[zero_preds[:, 0], 
            :, 
            zero_preds[:, 1], 
            zero_preds[:, 2]
        ] = new_preds.type(vol.dtype)

    # Check no double predictions occur
    double_preds = torch.argwhere(torch.sum(vol, dim=1) > 1)
    assert len(double_preds) == 0

    return vol

def compute_and_save_metrics(pred_vols, gt_vols, method):
    dice_2d, dice_3d, hd = torch.zeros((3, len(pred_vols), 4))
    for i in tqdm(range(len(pred_vols))):
        p, g = pred_vols[i].to('cuda'), gt_vols[i].to('cuda')
        _2d, _3d, _hd = compute_metrics(p, g)

        # 2D Dice
        dice_2d[i, :] = _2d

        # 3D Dice
        dice_3d[i, :] = _3d

        # Hausdorff
        hd[i, :] = _hd

    dice_2d = np.mean(dice_2d.cpu().numpy(), axis=0)
    dice_3d = np.mean(dice_3d.cpu().numpy(), axis=0)
    hd = np.mean(hd.cpu().numpy(), axis=0)

    dice_2d_class = {name_class : val for name_class, val in zip(class_names, dice_2d)}
    dice_3d_class = {name_class : val for name_class, val in zip(class_names, dice_3d)}
    hd_class = {name_class : val for name_class, val in zip(class_names, hd)}
    wandb.log({
        'dice_2d' : dice_2d_class,
        'dice_3d' : dice_3d_class,
        'hd' : hd_class,
        'step' : method
    })

def compute_metrics(p, g):
    dice_2d = torch.mean(dice_coef(g, p), axis=0)[1:]
    dice_3d = dice_batch(p,g)[1:]
    hd = hd95_batch(g, p)
    return dice_2d, dice_3d, hd

def load_data(args, debug=False): 
    # Select only files ending in .nii.gz
    paths = os.listdir(args.src)
    paths = [x for x in paths if '.nii.gz' in x]
    patient_ids = [x.split('_')[1][:2] for x in paths] if not debug else ['01']
    print('Found patient ids:', patient_ids)

    # Load nib files
    pred_nibs = [nib.load(f'{args.src}/Patient_{x}.nii.gz') for x in patient_ids]
    gt_nibs = [nib.load(f'{args.gt_src}/Patient_{x}/GT.nii.gz') for x in patient_ids]

    # Load volumes
    pred_vols = [(torch.from_numpy(np.asarray(x.dataobj)) / 63).type(torch.int64) for x in pred_nibs]
    gt_vols = [(torch.from_numpy(np.asarray(x.dataobj))).type(torch.int64) for x in gt_nibs]

    # Split volumes per class
    print('Splitting volumes per class...')
    p = [split_per_class(x) for x in pred_vols]
    g = [split_per_class(x) for x in gt_vols]

    return p, g, pred_nibs, patient_ids

def permute_vol(vol, ax):
    match ax:
        case 'yx':
            return vol
        case 'zx':
            return vol.permute(2,1,0,3)
        case 'yz':
            return vol.permute(3,1,2,0) 

def get_methods(args):
    methods = ['original']
    if args.nms:
        methods.append('nms')
    if args.gaussian_blur is not None:
        methods.append(f'gblur_{"_".join(args.gaussian_blur)}')
    if args.opening is not None:
        methods.append(f'open_{"_".join(args.opening)}')
    return methods

def get_method_function(m):
    match m.split('_'):
        case ['nms', *_]:
            return nms
        case ['gblur', *_]:
            return gaussianblur
        case ['open', *_]:
            return opening

def save_vol(vol, nib_, patient_id, method):
    # Convert to class representation and correct order of dimensions
    target_vol = deepcopy(vol)
    target_vol = one_hot2class(target_vol, K=5).permute(1,2,0)

    # Convert to nifti and save
    H, W, Z = target_vol.shape
    target_vol = resize_(target_vol.numpy(), (H*2, W*2, Z))
    new_nib = nib.nifti1.Nifti1Image(
        target_vol, affine=nib_.affine, header=nib_.header
    )
    nib.save(new_nib, f'{args.dest}/Patient_{patient_id}_{method}.nii.gz')    

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
        '--nms',
        action='store_true',
        default=False,
        help='Enable to apply Non-Maximum Suppression'
    )
    parser.add_argument(
        '--gaussian_blur',
        nargs='+',
        choices=['yx', 'zx', 'yz'],
        help='Enter dimensions to perform gaussian blur over. Multiple options can be selected. Options: ["yx", "zx", "yz"]',
        default=None
    )
    parser.add_argument(
        '--opening',
        nargs='+',
        choices=['yx', 'zx', 'yz'],
        help='Enter dimensions to perform opening over. Multiple options can be selected. Options: ["yx", "zx", "yz"]',
    )
    args = parser.parse_args()

    main(args)