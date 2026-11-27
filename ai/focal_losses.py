
import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalCombinedLoss(nn.Module):
    def __init__(
        self,
        alpha=0.75,
        gamma=2.0,
        focal_weight=0.5,
        dice_weight=0.5
    ):
        super().__init__()

        self.alpha = alpha
        self.gamma = gamma
        self.focal_weight = focal_weight
        self.dice_weight = dice_weight

    def forward(self, logits, targets, valid_mask):
        bce = F.binary_cross_entropy_with_logits(
            logits,
            targets,
            reduction="none"
        )

        probabilities = torch.sigmoid(logits)

        pt = (
            probabilities * targets
            + (1 - probabilities) * (1 - targets)
        )

        alpha_factor = (
            self.alpha * targets
            + (1 - self.alpha) * (1 - targets)
        )

        focal_loss = (
            alpha_factor
            * (1 - pt).pow(self.gamma)
            * bce
        )

        focal_loss = (
            focal_loss * valid_mask
        ).sum() / valid_mask.sum().clamp(min=1)

        intersection = (
            probabilities * targets * valid_mask
        ).sum()

        denominator = (
            (probabilities + targets) * valid_mask
        ).sum()

        dice = (
            2 * intersection + 1e-6
        ) / (
            denominator + 1e-6
        )

        dice_loss = 1 - dice

        return (
            self.focal_weight * focal_loss
            + self.dice_weight * dice_loss
        )
