import torch
import matplotlib.pyplot as plt
import numpy as np

from torch.utils.data import DataLoader
from ai.dataset import FloodDataset
from ai.unet import UNet


def main():
    device = torch.device("cpu")

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

    batch = None

    for candidate in loader:
        if candidate["valid_mask"].sum().item() > 0:
            batch = candidate
            break

    if batch is None:
        raise RuntimeError("No validation samples contain valid pixels.")

    image = batch["image"].to(device)
    target = batch["mask"].numpy()[0, 0]
    valid = batch["valid_mask"].numpy()[0, 0].astype(bool)

    with torch.no_grad():
        logits = model(image)
        prediction = (
            torch.sigmoid(logits) >= 0.5
        ).cpu().numpy()[0, 0]

    print("Image:", batch["image_name"])
    print("Valid pixels:", valid.sum())
    print("Target values:", np.unique(target, return_counts=True))
    print("Predicted water pixels:", prediction[valid].sum())

    sar_image = image.cpu().numpy()[0, 0]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(sar_image, cmap="gray")
    axes[0].set_title("SAR Image (VV)")

    axes[1].imshow(np.ma.masked_where(~valid, target), cmap="Blues", vmin=0, vmax=1)
    axes[1].set_title("Ground Truth")

    axes[2].imshow(np.ma.masked_where(~valid, prediction), cmap="Blues", vmin=0, vmax=1)
    axes[2].set_title("U-Net Prediction")

    for ax in axes:
        ax.axis("off")

    plt.tight_layout()
    plt.savefig("prediction_comparison.png", dpi=150)
    plt.show()


if __name__ == "__main__":
    main()