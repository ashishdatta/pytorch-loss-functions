import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F



class FocalLoss():
    pass

class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()
    
    def forward(self, y_pred, y_true, smooth=1):
        y_pred = F.sigmoid(y_pred)

        y_pred = y_pred.view(-1)
        y_true = y_true.view(-1)

        intersection = (y_pred * y_true).sum()
        dice = (2. * intersection + smooth) / (y_pred.sum() + y_true.sum() + smooth)
        return 1 - dice

class DiceBCELoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceBCELoss, self).__init__()

    def forward(self, y_pred, y_true, smooth=1):

        y_pred = F.sigmoid(y_pred)

        y_pred = y_pred.view(-1)
        y_true = y_true.view(-1)
        intersection = (y_pred * y_true).sum()
        dice = 1 - (2. * intersection + smooth) / (y_pred.sum() + y_true.sum() + smooth)
        BCE = F.binary_cross_entropy(y_pred, y_true, reduction='mean')
        dice_bce = BCE + dice

        return dice_bce

class IoULoss():
    pass

class TverskyLoss():
    pass

class FocalTverskyLoss():
    pass

class LovaszHingeLoss():
    pass

class ComboLoss():
    pass
