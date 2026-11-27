
import torch
import pandas as pd

from torch.utils.data import DataLoader
from ai.dataset import FloodDataset
from ai.unet import UNet


def main():
    device = torch.device("cpu")

    thresholds = [
        0.50, 0.55, 0.60, 0.65, 0.70,
        0.75, 0.80, 0.85, 0.90
    ]

    dataset = FloodDataset(
        root_dir="data/raw/Sen1Floods11",
        split="valid"
    )

    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False
    )

    model = UNet().to(device)
    model.load_state_dict(
        torch.load("best_unet.pth", map_location=device)
    )
    model.eval()

    counts = {
        t: {"tp": 0, "fp": 0, "fn": 0}
        for t in thresholds
    }

    image_ious = {t: [] for t in thresholds}

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            targets = batch["mask"].to(device).bool()
            valid = batch["valid_mask"].to(device).bool()

            if valid.sum().item() == 0:
                continue

            probabilities = torch.sigmoid(model(images))

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

                counts[threshold]["tp"] += tp
                counts[threshold]["fp"] += fp
                counts[threshold]["fn"] += fn

                union = tp + fp + fn

                # Exclude images with no positive pixels
                # in either prediction or ground truth.
                if union > 0:
                    image_ious[threshold].append(tp / union)

    results = []

    for threshold in thresholds:
        tp = counts[threshold]["tp"]
        fp = counts[threshold]["fp"]
        fn = counts[threshold]["fn"]

        micro_union = tp + fp + fn

        micro_iou = (
            tp / micro_union if micro_union > 0 else float("nan")
        )

        macro_iou = (
            sum(image_ious[threshold]) / len(image_ious[threshold])
            if image_ious[threshold]
            else float("nan")
        )

        results.append({
            "threshold": threshold,
            "micro_IoU": micro_iou,
            "macro_IoU": macro_iou,
            "evaluated_images": len(image_ious[threshold])
        })

    df = pd.DataFrame(results)

    print("\nMicro vs Macro IoU")
    print("------------------")
    print(df.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    ))

    df.to_csv(
        "data/processed/micro_macro_thresholds.csv",
        index=False
    )

    print("\nReport saved successfully.")


if __name__ == "__main__":
    main()
