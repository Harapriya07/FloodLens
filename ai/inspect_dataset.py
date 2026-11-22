
from pathlib import Path
import re

import numpy as np
import pandas as pd
import rasterio


ROOT = Path("data/raw/Sen1Floods11")

IMAGE_DIR = ROOT / "S1Hand"
LABEL_DIR = ROOT / "LabelHand"
SPLIT_DIR = ROOT / "splits"


def normalize_name(path):
    """Normalize image and label filenames for matching."""
    return re.sub(
        r"_(S1Hand|LabelHand)$",
        "",
        path.stem,
        flags=re.IGNORECASE
    )


def inspect_files():
    images = list(IMAGE_DIR.rglob("*.tif"))
    labels = list(LABEL_DIR.rglob("*.tif"))

    image_map = {normalize_name(p): p for p in images}
    label_map = {normalize_name(p): p for p in labels}

    common = sorted(set(image_map) & set(label_map))

    print("\n========== FILE AUDIT ==========")
    print("SAR images:", len(images))
    print("Label masks:", len(labels))
    print("Matched pairs:", len(common))
    print("Unmatched images:", len(images) - len(common))
    print("Unmatched labels:", len(labels) - len(common))

    if not common:
        print("\nNo matching filenames found.")
        print("Sample image names:", [p.name for p in images[:5]])
        print("Sample label names:", [p.name for p in labels[:5]])
        return

    return image_map[common[0]], label_map[common[0]], common


def inspect_raster(path, title):
    print(f"\n========== {title} ==========")

    with rasterio.open(path) as src:
        print("File:", path.name)
        print("Dimensions:", src.width, "x", src.height)
        print("Bands:", src.count)
        print("Data types:", src.dtypes)
        print("CRS:", src.crs)
        print("NoData metadata:", src.nodata)
        print("Band descriptions:", src.descriptions)

        for i in range(1, src.count + 1):
            band = src.read(i, masked=True)
            valid = band.compressed()

            if valid.size:
                print(f"\nBand {i}")
                print("Minimum:", float(valid.min()))
                print("Maximum:", float(valid.max()))
                print("Mean:", float(valid.mean()))
                print("Standard deviation:", float(valid.std()))

        if title == "LABEL MASK":
            mask = src.read(1)
            values, counts = np.unique(mask, return_counts=True)

            print("\nLabel values and pixel counts:")
            for value, count in zip(values, counts):
                print(f"{value}: {count:,} pixels")


def inspect_all_labels():
    print("\n========== AGGREGATE LABEL DISTRIBUTION ==========")

    total_pixels = 0
    class_counts = {}

    for path in LABEL_DIR.rglob("*.tif"):
        with rasterio.open(path) as src:
            mask = src.read(1)
            values, counts = np.unique(mask, return_counts=True)

            for value, count in zip(values, counts):
                class_counts[value] = (
                    class_counts.get(value, 0) + int(count)
                )

            total_pixels += mask.size

    print("Total pixels:", f"{total_pixels:,}")

    for value, count in sorted(class_counts.items()):
        percentage = 100 * count / total_pixels
        print(f"Class {value}: {count:,} ({percentage:.3f}%)")


def inspect_splits():
    print("\n========== DATASET SPLITS ==========")

    for path in sorted(SPLIT_DIR.glob("*.csv")):
        df = pd.read_csv(path)

        print(f"\n{path.name}")
        print("Rows:", len(df))
        print("Columns:", list(df.columns))
        print(df.head(3).to_string(index=False))


if __name__ == "__main__":
    result = inspect_files()

    if result:
        image_path, label_path, pairs = result

        inspect_raster(image_path, "SAR IMAGE")
        inspect_raster(label_path, "LABEL MASK")

        inspect_all_labels()
        inspect_splits()
