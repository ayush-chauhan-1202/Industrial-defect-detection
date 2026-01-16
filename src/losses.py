import torch
import torch.nn as nn

class DiceLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, preds, targets, smooth=1e-6):
        preds = torch.sigmoid(preds)
        preds = preds.view(preds.size(0), -1)
        targets = targets.view(targets.size(0), -1)

        intersection = (preds * targets).sum(dim=1)
        dice = (2. * intersection + smooth) / (preds.sum(dim=1) + targets.sum(dim=1) + smooth)
        return 1 - dice.mean()
