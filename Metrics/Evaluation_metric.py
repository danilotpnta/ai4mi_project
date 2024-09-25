### code is taken from
# https://github.com/nazib/MIDL_code/blob/hpc/evaluate.py

import numpy as np
import torch
from torch.nn.functional import one_hot
from monai.metrics import compute_dice

SMOOTH = 1e-6

def iou_numpy(outputs: np.array, labels: np.array):
    # outputs = outputs.squeeze(1)

    intersection = (outputs & labels).sum((1, 2))
    union = (outputs | labels).sum((1, 2))

    iou = (intersection + SMOOTH) / (union + SMOOTH)

    # thresholded = np.ceil(np.clip(20 * (iou - 0.5), 0, 10)) / 10

    return iou.mean()  # Or thresholded.mean()


def get_SegThor_regions():
    regions = {
        "esophagus": 1,
        "heart": 2,
        "trachea": 3,
        "aorta": 4
    }
    return regions


def get_LCTSC_regions():
    regions = {
        "Esophagus": 1,
        "Spinalcord": 2,
        "Heart": 3,
        "Left-lung": 4,
        "Right-lung": 5
    }
    return regions


def print_SegThor(organ_dice):
    all = []
    dice_dict = {"Esophegus Dice": organ_dice[1].item(), "Heart Dice": organ_dice[2].item(), "Trachea Dice": organ_dice[3].item(),
                 "Aorta Dice": organ_dice[4].item()}

    data = [dice_dict]

    for metric in data:
        for parameter in metric.keys():
            all.append(metric[parameter])
    return all


def print_LCTSC(organ_dice):
    all = []
    dice_dict = {"Esophegus Dice": organ_dice[1].item(), "Spine Dice": organ_dice[2].item(), "Heart Dice": organ_dice[3].item(),
                 "Left Lung Dice": organ_dice[4].item(), "Right Lung Dice": organ_dice[5].item()}

    data = [dice_dict]

    for metric in data:
        for parameter in metric.keys():
            all.append(metric[parameter])
    return all


def create_region_from_mask(mask, join_labels: tuple):
    mask_new = torch.zeros_like(mask)
    mask_new[mask == join_labels] = 1
    '''
    for l in join_labels:
        mask_new[mask == l] = 1
    '''
    mask_new = mask_new[None, :, :, :]
    return mask_new


def evaluate_case(image_gt, image_pred, class_num):

    image_gt = one_hot(image_gt, num_classes=class_num).permute(0,3,1,2)

    image_pred = image_pred[None, :, :, :, :] 
    image_gt = image_gt[None, :, :, :, :]

    print(f"GT shape: {image_gt.shape}, Pred shape: {image_pred.shape}")
    dc = torch.nan if torch.max(image_gt) < 1 and torch.max(image_pred) < 1 else compute_dice(image_pred, image_gt)

    # Initialize an empty list to hold the dice scores for each class
    dice_scores = []
    
    for c in range(1, class_num):  # Skip background class (index 0)
        # Compute Dice score for each class separately
        dice_score = compute_dice(image_pred[:, c], image_gt[:, c])
        dice_scores.append(dice_score.mean().item())  # Append the mean dice score for this class

    # Return the list of mean dice scores for each class (as a tensor if needed)
    return torch.tensor(dice_scores)

