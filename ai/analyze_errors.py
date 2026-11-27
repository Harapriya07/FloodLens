import torch
import numpy as np
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from ai.dataset import FloodDataset
from ai.unet import UNet


def main():
    device = torch.device("cpu")

    dataset = FloodDataset(
        root_dir="data/raw/Sen1Floods11",
        split="test"
    )

    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    model = UNet().to(device)
    model.load_state_dict(
        torch.load("best_unet.pth", map_location=device)
    )
    model.eval()

    worst_sample = None
    highest_errors = -1

    total_fp = 0
    total_fn = 0

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

            total_fp += fp
            total_fn += fn

            errors = fp + fn

            if errors > highest_errors:
                highest_errors = errors
                worst_sample = {
                    "name": batch["image_name"][0],
                    "image": image.numpy()[0, 0],
                    "target": target.copy(),
                    "prediction": prediction.copy(),
                    "valid": valid.copy(),
                    "fp": fp,
                    "fn": fn
                }

    print("Total False Positives:", total_fp)
    print("Total False Negatives:", total_fn)

    if worst_sample is None:
        raise RuntimeError("No valid test samples found.")

    print("\nSample with most errors:", worst_sample["name"])
    print("False Positives:", worst_sample["fp"])
    print("False Negatives:", worst_sample["fn"])

    target = worst_sample["target"]
    prediction = worst_sample["prediction"]
    valid = worst_sample["valid"]

    error_map = np.zeros((*target.shape, 3), dtype=np.float32)

    # False positives: red
    error_map[(prediction == 1) & (target == 0) & valid] = [1, 0, 0]

    # False negatives: blue
    error_map[(prediction == 0) & (target == 1) & valid] = [0, 0, 1]

    # Correct predictions: green
    error_map[(prediction == target) & valid] = [0, 1, 0]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(worst_sample["image"], cmap="gray")
    axes[0].set_title("SAR Image")

    axes[1].imshow(np.ma.masked_where(~valid, target), cmap="Blues", vmin=0, vmax=1)
    axes[1].set_title("Ground Truth")

    axes[2].imshow(error_map)
    axes[2].set_title("Error Map (Green=Correct, Red=FP, Blue=FN)")

    for ax in axes:
        ax.axis("off")

    plt.tight_layout()
    plt.savefig("error_analysis.png", dpi=150)
    plt.show()


if __name__ == "__main__":
    main()