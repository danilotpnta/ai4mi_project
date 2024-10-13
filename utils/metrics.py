import torch
import numpy as np
from medpy import metric
from functools import partial
from torch import Tensor, einsum
from utils.tensor_utils import one_hot, sset
from monai.metrics import HausdorffDistanceMetric
import math


def meta_dice(
    sum_str: str, label: Tensor, pred: Tensor, smooth: float = 1e-8
) -> Tensor:
    assert label.shape == pred.shape
    # assert one_hot(label)
    # assert one_hot(pred)

    inter_size: Tensor = einsum(sum_str, [intersection(label, pred)]).type(
        torch.float32
    )
    sum_sizes: Tensor = (einsum(sum_str, [label]) + einsum(sum_str, [pred])).type(
        torch.float32
    )

    dices: Tensor = (2 * inter_size + smooth) / (sum_sizes + smooth)

    return dices


dice_coef = partial(meta_dice, "bk...->bk")
dice_batch = partial(meta_dice, "bk...->k")  # used for 3d dice


def intersection(a: Tensor, b: Tensor) -> Tensor:
    assert a.shape == b.shape
    assert sset(a, [0, 1])
    assert sset(b, [0, 1])

    res = a & b
    assert sset(res, [0, 1])

    return res


def union(a: Tensor, b: Tensor) -> Tensor:
    assert a.shape == b.shape
    assert sset(a, [0, 1])
    assert sset(b, [0, 1])

    res = a | b
    assert sset(res, [0, 1])

    return res


def iou_coef(label: Tensor, pred: Tensor, smooth: float = 1e-8) -> Tensor:
    assert label.shape == pred.shape

    inter_size: Tensor = intersection(label, pred).sum().float()
    union_size: Tensor = union(label, pred).sum().float()

    return (inter_size + smooth) / (union_size + smooth)


def hd95_batch(label: Tensor, pred: Tensor) -> Tensor:
    """
    label: (batch, Classes, H, W, D) - onehot including background
    pred: (batch, Classes, H, W, D) - onehot including background
    return: (Classes) - Hausdorff Distance 95 reduced over batch, where the distance is nan because of empty pred
                        the biggest possible distance, the diagonal of the volume is used in the mean calculation 
    """
    
    #print("pred.shape", pred.shape)
    #print("label.shape", label.shape)
    
    diagonal = math.sqrt(label.shape[2]**2 + label.shape[3]**2)
    #print("diagonal", diagonal)
    
    hausdorff_metric = HausdorffDistanceMetric(include_background=False, percentile=95)

    score = hausdorff_metric(pred, label)

    #print("score", score)

    score = torch.where(torch.isnan(score), diagonal, score)

    #print("score", score)

    score = torch.mean(score, dim=0)

    #print("score", score)
    return score
