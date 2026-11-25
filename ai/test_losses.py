import torch

from ai.losses import CombinedLoss


def main():
    logits = torch.randn(
    2, 1, 512, 512,
    requires_grad=True
    )
    targets = torch.randint(0, 2, (2, 1, 512, 512)).float()
    valid_mask = torch.ones_like(targets)

    # Simulate some NoData pixels
    valid_mask[:, :, :20, :20] = 0

    criterion = CombinedLoss()
    loss = criterion(logits, targets, valid_mask)

    print("Loss:", loss.item())

    loss.backward()
    print("Backward pass: successful")


if __name__ == "__main__":
    main()