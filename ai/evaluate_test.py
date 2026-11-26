import torch
from torch.utils.data import DataLoader

from ai.dataset import FloodDataset
from ai.unet import UNet


def main():
    device = torch.device("cpu")

    test_dataset = FloodDataset(
        root_dir="data/raw/Sen1Floods11",
        split="test"
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=2,
        shuffle=False
    )

    model = UNet().to(device)
    model.load_state_dict(
        torch.load("best_unet.pth", map_location=device)
    )
    model.eval()

    tp = fp = fn = 0

    with torch.no_grad():
        for batch in test_loader:
            images = batch["image"].to(device)
            targets = batch["mask"].to(device)
            valid = batch["valid_mask"].to(device).bool()

            logits = model(images)
            predictions = torch.sigmoid(logits) >= 0.5

            tp += ((predictions == 1) & (targets == 1) & valid).sum().item()
            fp += ((predictions == 1) & (targets == 0) & valid).sum().item()
            fn += ((predictions == 0) & (targets == 1) & valid).sum().item()

    epsilon = 1e-8

    iou = tp / (tp + fp + fn + epsilon)
    dice = (2 * tp) / (2 * tp + fp + fn + epsilon)
    precision = tp / (tp + fp + epsilon)
    recall = tp / (tp + fn + epsilon)

    print("\nOfficial Test Metrics")
    print("---------------------")
    print(f"IoU:       {iou:.4f}")
    print(f"Dice:      {dice:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")


if __name__ == "__main__":
    main()