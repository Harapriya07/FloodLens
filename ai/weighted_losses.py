
import torch
import torch.nn as nn


class WeightedCombinedLoss(nn.Module):
    def __init__(
        self,
        pos_weight=9.52,
        bce_weight=0.5,
        dice_weight=0.5
    ):
        super().__init__()

        self.register_buffer(
            "pos_weight",
            torch.tensor([pos_weight], dtype=torch.float32)
        )

        self.bce = nn.BCEWithLogitsLoss(
            reduction="none",
            pos_weight=self.pos_weight
        )

        self.bce_weight = bce_weight
        self.dice_weight = dice_weight

    def forward(self, logits, targets, valid_mask):
        bce = self.bce(logits, targets)
        bce = (bce * valid_mask).sum() / valid_mask.sum().clamp(min=1)

        probabilities = torch.sigmoid(logits)

        intersection = (
            probabilities * targets * valid_mask
        ).sum()

        denominator = (
            (probabilities + targets) * valid_mask
        ).sum()

        dice = (2 * intersection + 1e-6) / (denominator + 1e-6)
        dice_loss = 1 - dice

        return (
            self.bce_weight * bce
            + self.dice_weight * dice_loss
        )
