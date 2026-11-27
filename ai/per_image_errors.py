import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from ai.dataset import FloodDataset
from ai.unet import UNet


def create_error_map(target, prediction, valid):
    error_map = np.zeros((*target.shape, 3), dtype=np.float32)

    # True positives and true negatives
    correct = (prediction == target) & valid
    error_map[correct] = [0, 1, 0]

    # False positives
    fp = (prediction == 1) & (target == 0) & valid
    error_map[fp] = [1, 0, 0]

    # False negatives
    fn = (prediction == 0) & (target == 1) & valid
    error_map[fn] = [0, 0, 1]

    return error_map


def main():
    device = torch.device("cpu")

    output_dir = "data/processed/validation_error_visuals"
    os.makedirs(output_dir, exist_ok=True)

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
            image = batch["image"].to(device)
            target = batch["mask"].numpy()[0, 0]
            valid = batch["valid_mask"].numpy()[0, 0].astype(bool)

            if not valid.any():
                continue

            logits = model(image)

            prediction = (
                torch.sigmoid(logits) >= 0.5
            ).numpy()[0, 0]

            fp = ((prediction == 1) & (target == 0) & valid).sum()
            fn = ((prediction == 0) & (target == 1) & valid).sum()

            results.append({
                "image": batch["image_name"][0],
                "sar": image.numpy()[0, 0],
                "target": target.copy(),
                "prediction": prediction.copy(),
                "valid": valid.copy(),
                "fp": int(fp),
                "fn": int(fn),
                "valid_pixels": int(valid.sum())
            })

    if not results:
        raise RuntimeError("No valid validation images found.")

    # Find separate FP-dominant and FN-dominant examples.
    fp_sample = max(
        results,
        key=lambda x: x["fp"] / x["valid_pixels"]
    )

    fn_sample = max(
        results,
        key=lambda x: x["fn"] / x["valid_pixels"]
    )

    for label, sample in [
        ("false_positive", fp_sample),
        ("false_negative", fn_sample)
    ]:
        error_map = create_error_map(
            sample["target"],
            sample["prediction"],
            sample["valid"]
        )

        fig, axes = plt.subplots(1, 4, figsize=(18, 5))

        axes[0].imshow(sample["sar"], cmap="gray")
        axes[0].set_title("SAR Image")

        axes[1].imshow(
            np.ma.masked_where(
                ~sample["valid"], sample["target"]
            ),
            cmap="Blues",
            vmin=0,
            vmax=1
        )
        axes[1].set_title("Ground Truth")

        axes[2].imshow(
            np.ma.masked_where(
                ~sample["valid"], sample["prediction"]
            ),
            cmap="Blues",
            vmin=0,
            vmax=1
        )
        axes[2].set_title("Prediction")

        axes[3].imshow(error_map)
        axes[3].set_title("Error Map")

        for ax in axes:
            ax.axis("off")

        fig.suptitle(
            f"{sample['image']} | FP: {sample['fp']} | FN: {sample['fn']}"
        )

        plt.tight_layout()
        plt.savefig(
            os.path.join(output_dir, f"{label}.png"),
            dpi=150
        )
        plt.close(fig)

        print(f"\n{label.upper()}: {sample['image']}")
        print("False positives:", sample["fp"])
        print("False negatives:", sample["fn"])

    print("\nVisualizations saved in:", output_dir)


if __name__ == "__main__":
    main()