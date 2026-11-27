
from ai.dataset import FloodDataset


def analyze_split(split):
    dataset = FloodDataset(
        root_dir="data/raw/Sen1Floods11",
        split=split
    )

    total_water = 0
    total_background = 0

    for sample in dataset:
        mask = sample["mask"]
        valid_mask = sample["valid_mask"]

        water = (mask == 1) & valid_mask
        background = (mask == 0) & valid_mask

        total_water += water.sum().item()
        total_background += background.sum().item()

    total = total_water + total_background

    water_percentage = 100 * total_water / total
    background_percentage = 100 * total_background / total

    print(f"\n{split.upper()} SPLIT")
    print(f"Water pixels:      {total_water:,}")
    print(f"Background pixels: {total_background:,}")
    print(f"Water percentage:  {water_percentage:.2f}%")
    print(f"Background percentage: {background_percentage:.2f}%")


if __name__ == "__main__":
    analyze_split("train")
    analyze_split("valid")
