
import torch
import pandas as pd

from torch.utils.data import DataLoader
from ai.dataset import FloodDataset
from ai.unet import UNet


def main():
    device = torch.device("cpu")
    thresholds = [0.50, 0.85]

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

    results = []

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            targets = batch["mask"].to(device).bool()
            valid = batch["valid_mask"].to(device).bool()

            if valid.sum().item() == 0:
                continue

            probabilities = torch.sigmoid(model(images))

            row = {"image": batch["image_name"][0]}

            for threshold in thresholds:
                predictions = probabilities >= threshold

                tp = (
                    predictions & targets & valid
                ).sum().item()

                fp = (
                    predictions & ~targets & valid
                ).sum().item()

                fn = (
                    ~predictions & targets & valid
                ).sum().item()

                union = tp + fp + fn

                iou = tp / union if union > 0 else float("nan")

                row[f"IoU_{threshold:.2f}"] = iou

            row["IoU_change"] = row["IoU_0.85"] - row["IoU_0.50"]
            results.append(row)

    df = pd.DataFrame(results)

    print("\nPer-image IoU comparison")
    print("------------------------")
    print(df.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    ))

    valid_changes = df["IoU_change"].dropna()

    print("\nSummary")
    print("-------")
    print(f"Images evaluated: {len(df)}")
    print(f"Mean IoU at 0.50: {df['IoU_0.50'].mean():.4f}")
    print(f"Mean IoU at 0.85: {df['IoU_0.85'].mean():.4f}")
    print(f"Images improved: {(valid_changes > 0).sum()}")
    print(f"Images worsened: {(valid_changes < 0).sum()}")
    print(f"Images unchanged: {(valid_changes == 0).sum()}")

    df.to_csv(
        "data/processed/per_image_threshold_comparison.csv",
        index=False
    )

    print("\nReport saved successfully.")


if __name__ == "__main__":
    main()
