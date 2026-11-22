
from pathlib import Path

import numpy as np
import rasterio
import matplotlib.pyplot as plt

ROOT = Path("data/raw/Sen1Floods11")

image_path = ROOT / "S1Hand" / "Bolivia_103757_S1Hand.tif"
label_path = ROOT / "LabelHand" / "Bolivia_103757_LabelHand.tif"


with rasterio.open(image_path) as src:
    vv = src.read(1)
    image_valid = np.isfinite(vv)
    image_transform = src.transform
    image_crs = src.crs

with rasterio.open(label_path) as src:
    labels = src.read(1)
    label_transform = src.transform
    label_crs = src.crs

assert vv.shape == labels.shape

assert image_crs == label_crs

assert np.allclose(
    tuple(image_transform),
    tuple(label_transform),
    rtol=0,
    atol=1e-12
)

print("Geospatial alignment verified!")

low, high = np.percentile(vv[image_valid], [2, 98])
vv_display = np.clip((vv - low) / (high - low), 0, 1)

fig, ax = plt.subplots(figsize=(9, 9))

ax.imshow(vv_display, cmap="gray")

water = np.ma.masked_where(
    (labels != 1) | ~image_valid,
    labels
)

ax.imshow(
    water,
    cmap="autumn",
    alpha=0.55,
    vmin=0,
    vmax=1
)

ax.set_title("FloodLens: VV + Water Label Overlay")
ax.axis("off")

plt.tight_layout()

output = Path("data/processed/visualizations")
output.mkdir(parents=True, exist_ok=True)

plt.savefig(output / "water_overlay.png", dpi=200)
plt.show()
