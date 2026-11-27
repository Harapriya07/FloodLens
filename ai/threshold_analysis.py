
import torch
import pandas as pd

from torch.utils.data import DataLoader
from ai.dataset import FloodDataset
from ai.unet import UNet


def main():
    device = torch.device("cpu")
    thresholds = [0.70, 0.75, 0.80, 0.85, 0.90]

    dataset = FloodDataset(
        root_dir="data/raw/Sen1Floods11",
        split="valid"
    )

    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    model = UNet().to(device)
    model.load_state_dict(
        torch.load("best_unet.pth", map_location=device)
    )
    model.eval()

    counts = {
        t: {"tp": 0, "fp": 0, "fn": 0}
        for t in thresholds
    }

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            targets = batch["mask"].to(device)
            valid = batch["valid_mask"].to(device).bool()

            if valid.sum().item() == 0:
                continue

            probabilities = torch.sigmoid(model(images))

            for threshold in thresholds:
                predictions = probabilities >= threshold

                counts[threshold]["tp"] += (
                    (predictions == 1) & (targets == 1) & valid
                ).sum().item()

                counts[threshold]["fp"] += (
                    (predictions == 1) & (targets == 0) & valid
                ).sum().item()

                counts[threshold]["fn"] += (
                    (predictions == 0) & (targets == 1) & valid
                ).sum().item()

    results = []

    for threshold, values in counts.items():
        tp = values["tp"]
        fp = values["fp"]
        fn = values["fn"]

        epsilon = 1e-8

        results.append({
            "threshold": threshold,
            "IoU": tp / (tp + fp + fn + epsilon),
            "Dice": (2 * tp) / (2 * tp + fp + fn + epsilon),
            "Precision": tp / (tp + fp + epsilon),
            "Recall": tp / (tp + fn + epsilon)
        })

    df = pd.DataFrame(results)

    print("\nFine-Grained Threshold Analysis")
    print("--------------------------------")
    print(df.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    ))

    df.to_csv(
        "data/processed/threshold_analysis_fine.csv",
        index=False
    )

    print("\nResults saved successfully.")


if __name__ == "__main__":
    main()
