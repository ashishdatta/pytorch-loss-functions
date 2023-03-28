import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class FocalLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(FocalLoss, self).__init__()

    def forward(
        self,
        y_pred: torch.Tensor,
        y_true: torch.Tensor,
        alpha: Optional[float] = 0.8,
        gamma: Optional[float] = 2.0,
        smooth: Optional[int] = 1,
    ) -> torch.Tensor:
        y_pred = F.sigmoid(y_pred)
        y_pred = y_pred.view(-1)
        y_true = y_true.view(-1)

        logpt = F.binary_cross_entropy(y_pred, y_true, reduction="mean")
        pt = torch.exp(-logpt)
        focal_term = (1.0 - pt).pow(gamma)
        focal_loss = alpha * focal_term * logpt

        return focal_loss


class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(
        self, y_pred: torch.Tensor, y_true: torch.Tensor, smooth: Optional[int] = 1
    ) -> torch.Tensor:
        y_pred = F.sigmoid(y_pred)

        y_pred = y_pred.view(-1)
        y_true = y_true.view(-1)

        intersection = (y_pred * y_true).sum()
        dice = (2.0 * intersection + smooth) / (y_pred.sum() + y_true.sum() + smooth)
        return 1 - dice


class DiceBCELoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceBCELoss, self).__init__()

    def forward(
        self, y_pred: torch.Tensor, y_true: torch.Tensor, smooth: Optional[int] = 1
    ) -> torch.Tensor:
        y_pred = F.sigmoid(y_pred)

        y_pred = y_pred.view(-1)
        y_true = y_true.view(-1)
        intersection = (y_pred * y_true).sum()
        dice = 1 - (2.0 * intersection + smooth) / (
            y_pred.sum() + y_true.sum() + smooth
        )
        BCE = F.binary_cross_entropy(y_pred, y_true, reduction="mean")
        dice_bce = BCE + dice

        return dice_bce


class IoULoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(IoULoss, self).__init__()

    def forward(
        self, y_pred: torch.Tensor, y_true: torch.Tensor, smooth: Optional[int] = 1
    ) -> torch.Tensor:
        y_pred = F.sigmoid(y_pred)

        y_pred = y_pred.view(-1)
        y_true = y_true.view(-1)

        intersection = (y_pred * y_true).sum()
        union = (y_pred + y_true).sum() - intersection
        IoU = (intersection + smooth) / (union + smooth)
        return 1 - IoU


# Note when alpha and beta = 0.5 this is the same as Dice Loss
class TverskyLoss(nn.Module):
    def __init__(self, weights=None, size_average=True):
        super(TverskyLoss, self).__init__()

    def forward(
        self,
        y_pred: torch.Tensor,
        y_true: torch.Tensor,
        smooth: Optional[int] = 1,
        alpha: Optional[float] = 0.5,
        beta: Optional[float] = 0.5,
    ) -> torch.Tensor:
        y_pred = F.sigmoid(y_pred)

        y_pred = y_pred.view(-1)
        y_true = y_true.view(-1)

        TP = (y_pred * y_true).sum()
        FP = ((1 - y_true) * y_pred).sum()
        FN = (y_true * (1 - y_pred)).sum()

        tversky = (TP + smooth) / (TP + alpha * FP + beta * FN + smooth)

        return 1 - tversky


class FocalTverskyLoss(nn.Module):
    pass


class LovaszHingeLoss(nn.Module):
    pass


class ComboLoss(nn.Module):
    pass
