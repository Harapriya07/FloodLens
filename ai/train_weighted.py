
import torch
from torch.utils.data import DataLoader

from ai.dataset import FloodDataset
from ai.unet import UNet
from ai.weighted_losses import WeightedCombinedLoss


def main():
    device = torch.device("cpu")
    root_dir = "data/raw/Sen1Floods11"

    # Load datasets
    train_dataset = FloodDataset(root_dir=root_dir, split="train")
    valid_dataset = FloodDataset(root_dir=root_dir, split="valid")

    train_loader = DataLoader(
        train_dataset,
        batch_size=2,
        shuffle=True
    )

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=2,
        shuffle=False
    )

    # Initialize model, weighted loss and optimizer
    model = UNet().to(device)

    criterion = WeightedCombinedLoss(
        pos_weight=9.52,
        bce_weight=0.5,
        dice_weight=0.5
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.001
    )

    epochs = 5
    best_valid_loss = float("inf")

    # Training loop
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0

        for batch in train_loader:
            images = batch["image"].to(device)
            masks = batch["mask"].to(device)
            valid_masks = batch["valid_mask"].to(device)

            optimizer.zero_grad()

            logits = model(images)
            loss = criterion(logits, masks, valid_masks)

            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        # Validation loop
        model.eval()
        valid_loss = 0.0

        with torch.no_grad():
            for batch in valid_loader:
                images = batch["image"].to(device)
                masks = batch["mask"].to(device)
                valid_masks = batch["valid_mask"].to(device)

                logits = model(images)
                loss = criterion(logits, masks, valid_masks)

                valid_loss += loss.item()

        valid_loss /= len(valid_loader)

        print(
            f"Epoch [{epoch + 1}/{epochs}] "
            f"Train Loss: {train_loss:.4f} "
            f"Validation Loss: {valid_loss:.4f}"
        )

        # Save the best weighted model
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss

            torch.save(
                model.state_dict(),
                "best_weighted_unet.pth"
            )

            print("Best weighted model saved!")

    print(f"\nBest validation loss: {best_valid_loss:.4f}")


if __name__ == "__main__":
    main()
