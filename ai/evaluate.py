import torch
from torch.utils.data import DataLoader
import csv
import os
from ai.dataset import FloodDataset
from ai.unet import UNet


def main():
    device = torch.device("cpu")

    dataset = FloodDataset(
        root_dir="data/raw/Sen1Floods11",
        split="valid"
    )

    loader = DataLoader(
        dataset,
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
        for batch in loader:
            images = batch["image"].to(device)
            targets = batch["mask"].to(device)
            valid = batch["valid_mask"].to(device).bool()

            logits = model(images)
            predictions = (torch.sigmoid(logits) >= 0.5)

            tp += ((predictions == 1) & (targets == 1) & valid).sum().item()
            fp += ((predictions == 1) & (targets == 0) & valid).sum().item()
            fn += ((predictions == 0) & (targets == 1) & valid).sum().item()

    epsilon = 1e-8
    iou = tp / (tp + fp + fn + epsilon)
    dice = (2 * tp) / (2 * tp + fp + fn + epsilon)
    precision = tp / (tp + fp + epsilon)
    recall = tp / (tp + fn + epsilon)

    os.makedirs("data/processed", exist_ok=True)

    with open("data/processed/validation_metrics.csv", "w", newline="") as file:
        writer = csv.writer(file)

        writer.writerow(["Metric", "Score"])
        writer.writerow(["IoU", iou])
        writer.writerow(["Dice", dice])
        writer.writerow(["Precision", precision])
        writer.writerow(["Recall", recall])

    print("Metrics saved successfully.")

    print("\nValidation Metrics")
    print("------------------")
    print(f"IoU:       {iou:.4f}")
    print(f"Dice:      {dice:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")


if __name__ == "__main__":
    main()