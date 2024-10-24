#!/usr/bin/env python3

# MIT License

# Copyright (c) 2024 Hoel Kervadec

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


from monai.losses import DiceCELoss, DiceFocalLoss, DiceLoss, FocalLoss
from torch import einsum, nn

from utils.tensor_utils import simplex, sset


class CrossEntropy:
    def __init__(self, **kwargs):
        # Self.idk is used to filter out some classes of the target mask. Use fancy indexing
        self.idk = kwargs["idk"]
        print(f"Initialized {self.__class__.__name__} with {kwargs}")

    def __call__(self, pred_softmax, weak_target):
        # assert pred_softmax.shape == weak_target.shape
        # assert simplex(pred_softmax)
        # assert sset(weak_target, [0, 1])

        log_p = (pred_softmax[:, self.idk, ...] + 1e-10).log()
        mask = weak_target[:, self.idk, ...].float()

        loss = -einsum("bkwh,bkwh->", mask, log_p)
        loss /= mask.sum() + 1e-10

        return loss


class PartialCrossEntropy(CrossEntropy):
    def __init__(self, **kwargs):
        super().__init__(idk=[1], **kwargs)


def get_loss(
    loss: str,
    K: int,
    lambda_dice: float = 1,
    lambda_ce: float = 1,
    lambda_focal: float = 1,
    include_background: bool = False,
):
    match loss.lower():
        case "dice":
            return DiceLoss(include_background=include_background)
        case "dicece":
            return DiceCELoss(
                lambda_ce=lambda_ce,
                lambda_dice=lambda_dice,
                include_background=include_background,
            )
        case "dicefocal":
            return DiceFocalLoss(
                sigmoid=True,
                gamma=0.5,
                lambda_dice=lambda_dice,
                include_background=include_background,
            )
        case "focal":
            return FocalLoss(
                weight=None,
                gamma=2,
                include_background=include_background,
            )
        case "ce":
            return CrossEntropy(idk=list(range(int(include_background), K)))
        case "ce_torch":
            return nn.CrossEntropyLoss(ignore_index=-100 if include_background else 0)
        case _:
            raise ValueError(f"Unknown loss: {loss}")
