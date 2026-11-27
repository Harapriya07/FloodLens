
import argparse
import torch
from torch.utils.data import DataLoader

from ai.dataset import FloodDataset
from ai.unet import UNet
from ai.losses import CombinedLoss
from ai.reproducibility import set_seed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)

    device = torch.device("cpu")
    print("Using device:", device)

    root_dir = "data/raw/Sen1Floods11"

    train_dataset = FloodDataset(root_dir=root_dir, split="train")
    val_dataset = FloodDataset(root_dir=root_dir, split="valid")

    generator = torch.Generator()
    generator.manual_seed(args.seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=2,
        shuffle=True,
        generator=generator
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=2,
        shuffle=False
    )

    model = UNet().to(device)
    criterion = CombinedLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    epochs = 5
    best_val_loss = float("inf")

    checkpoint_path = f"best_unet_seed_{args.seed}.pth"

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0

        for batch in train_loader:
            images = batch["image"].to(device)
            masks = batch["mask"].to(device)
            valid_mask = batch["valid_mask"].to(device)

            optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, masks, valid_mask)

            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for batch in val_loader:
                images = batch["image"].to(device)
                masks = batch["mask"].to(device)
                valid_mask = batch["valid_mask"].to(device)

                outputs = model(images)
                loss = criterion(outputs, masks, valid_mask)

                val_loss += loss.item()

        val_loss /= len(val_loader)

        print(
            f"Epoch [{epoch + 1}/{epochs}] "
            f"Train Loss: {train_loss:.4f} "
            f"Validation Loss: {val_loss:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss

            torch.save(model.state_dict(), checkpoint_path)
            print(f"Checkpoint saved: {checkpoint_path}")

    print(f"\nBest validation loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    main()
