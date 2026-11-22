
from pathlib import Path
import rasterio

ROOT = Path("data/raw/Sen1Floods11")

image_path = ROOT / "S1Hand" / "Bolivia_103757_S1Hand.tif"
label_path = ROOT / "LabelHand" / "Bolivia_103757_LabelHand.tif"

with rasterio.open(image_path) as image, \
     rasterio.open(label_path) as label:

    print("\n========== IMAGE METADATA ==========")
    print("Shape:", image.shape)
    print("CRS:", image.crs)
    print("Transform:", image.transform)
    print("Bounds:", image.bounds)
    print("Resolution:", image.res)

    print("\n========== LABEL METADATA ==========")
    print("Shape:", label.shape)
    print("CRS:", label.crs)
    print("Transform:", label.transform)
    print("Bounds:", label.bounds)
    print("Resolution:", label.res)

    print("\n========== COMPARISON ==========")
    print("Same dimensions:", image.shape == label.shape)
    print("Same CRS:", image.crs == label.crs)
    print("Same transform:", image.transform == label.transform)
    print("Same bounds:", image.bounds == label.bounds)

    print("\nTransform difference:")
    for name, img_value, label_value in zip(
        ["a", "b", "c", "d", "e", "f"],
        tuple(image.transform),
        tuple(label.transform)
    ):
        print(
            f"{name}: image={img_value}, "
            f"label={label_value}, "
            f"difference={img_value - label_value}"
        )
