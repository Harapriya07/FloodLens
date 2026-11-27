
import torch
import pandas as pd
from torch.utils.data import DataLoader

from ai.dataset import FloodDataset
from ai.unet import UNet


def main():
    device = torch.device("cpu")
    thresholds = [0.30, 0.40, 0.50, 0.60, 0.70, 0.75, 0.80, 0.85, 0.90]

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
        torch.load(
            "best_focal_unet.pth",
            map_location=device,
            weights_only=True
        )
    )
    model.eval()

    results = []

    with torch.no_grad():
        for threshold in thresholds:
            tp = fp = fn = 0

            for batch in loader:
                images = batch["image"].to(device)
                masks = batch["mask"].to(device)
                valid_masks = batch["valid_mask"].to(device)

                logits = model(images)
                probabilities = torch.sigmoid(logits)

                predictions = probabilities >= threshold
                targets = masks >= 0.5

                tp += (predictions & targets & valid_masks).sum().item()
                fp += (predictions & ~targets & valid_masks).sum().item()
                fn += (~predictions & targets & valid_masks).sum().item()

            iou = tp / (tp + fp + fn + 1e-7)
            dice = (2 * tp) / (2 * tp + fp + fn + 1e-7)
            precision = tp / (tp + fp + 1e-7)
            recall = tp / (tp + fn + 1e-7)

            results.append({
                "threshold": threshold,
                "IoU": iou,
                "Dice": dice,
                "Precision": precision,
                "Recall": recall
            })

            print(
                f"Threshold: {threshold:.2f} | "
                f"IoU: {iou:.4f} | "
                f"Dice: {dice:.4f} | "
                f"Precision: {precision:.4f} | "
                f"Recall: {recall:.4f}"
            )

    df = pd.DataFrame(results)
    df.to_csv(
        "data/processed/focal_threshold_metrics.csv",
        index=False
    )

    print("\nBest threshold by micro IoU:")
    print(df.loc[df["IoU"].idxmax()].to_string())


if __name__ == "__main__":
    main()
