
from pathlib import Path

import numpy as np
import rasterio
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

ROOT = Path("data/raw/Sen1Floods11")

IMAGE_PATH = ROOT / "S1Hand" / "Bolivia_103757_S1Hand.tif"
LABEL_PATH = ROOT / "LabelHand" / "Bolivia_103757_LabelHand.tif"

OUTPUT_DIR = Path("data/processed/visualizations")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def display_stretched_band(band):
    """Apply percentile stretching for visualization only."""

    valid = np.isfinite(band)

    low, high = np.percentile(band[valid], [2, 98])

    stretched = np.clip(
        (band - low) / (high - low),
        0,
        1
    )

    return np.ma.masked_where(~valid, stretched)


def visualize():

    with rasterio.open(IMAGE_PATH) as src:
        vv = src.read(1)
        vh = src.read(2)

    with rasterio.open(LABEL_PATH) as src:
        mask = src.read(1)

    assert vv.shape == vh.shape == mask.shape, (
        "Image and mask dimensions do not match!"
    )

    vv_display = display_stretched_band(vv)
    vh_display = display_stretched_band(vh)

    # -1 = NoData, 0 = Non-water, 1 = Water
    label_display = np.ma.masked_where(mask == -1, mask)

    cmap = ListedColormap(["black", "cyan"])

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    axes[0].imshow(vv_display, cmap="gray")
    axes[0].set_title("Sentinel-1 VV")

    axes[1].imshow(vh_display, cmap="gray")
    axes[1].set_title("Sentinel-1 VH")

    axes[2].imshow(
        label_display,
        cmap=cmap,
        vmin=0,
        vmax=1
    )
    axes[2].set_title("Ground Truth Water Mask")

    for ax in axes:
        ax.axis("off")

    fig.suptitle("FloodLens - Sen1Floods11 Sample")

    plt.tight_layout()

    output_path = OUTPUT_DIR / "bolivia_sample.png"

    plt.savefig(output_path, dpi=200, bbox_inches="tight")

    print("Visualization saved to:", output_path)

    plt.show()


if __name__ == "__main__":
    visualize()
