
import torch
from ai.weighted_losses import WeightedCombinedLoss


def test_weighted_loss():
    logits = torch.randn(2, 1, 64, 64, requires_grad=True)
    targets = torch.randint(0, 2, (2, 1, 64, 64)).float()
    valid_mask = torch.ones_like(targets, dtype=torch.bool)

    criterion = WeightedCombinedLoss(pos_weight=9.52)

    loss = criterion(logits, targets, valid_mask)

    assert torch.isfinite(loss)
    loss.backward()

    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()

    print("Weighted loss test passed!")
    print(f"Loss: {loss.item():.4f}")


if __name__ == "__main__":
    test_weighted_loss()
