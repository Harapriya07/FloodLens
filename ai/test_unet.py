import torch
from torch.utils.data import DataLoader

from ai.dataset import FloodDataset
from ai.unet import UNet


def main():
    dataset = FloodDataset(
    root_dir="data/raw/Sen1Floods11",
    split="train"
    )

    loader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=True
    )

    batch = next(iter(loader))

    images = batch["image"]
    masks = batch["mask"]

    model = UNet()

    outputs = model(images)

    print("Input shape:", images.shape)
    print("Target mask shape:", masks.shape)
    print("Model output shape:", outputs.shape)


if __name__ == "__main__":
    main()