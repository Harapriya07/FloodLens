import torch
import torch.nn as nn


class CombinedLoss(nn.Module):
    def __init__(self, bce_weight=0.5, dice_weight=0.5):
        super().__init__()

        self.bce = nn.BCEWithLogitsLoss(reduction="none")
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight

    def forward(self, logits, targets, valid_mask):
        # BCE loss
        bce = self.bce(logits, targets)
        bce = (bce * valid_mask).sum() / valid_mask.sum().clamp(min=1)

        # Dice loss
        probabilities = torch.sigmoid(logits)

        intersection = (probabilities * targets * valid_mask).sum()
        denominator = ((probabilities + targets) * valid_mask).sum()

        dice = (2 * intersection + 1e-6) / (denominator + 1e-6)
        dice_loss = 1 - dice

        return (
            self.bce_weight * bce
            + self.dice_weight * dice_loss
        )