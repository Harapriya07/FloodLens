
import torch
from torch.utils.data import DataLoader

from ai.dataset import FloodDataset
from ai.unet import UNet


def main():
    device = torch.device("cpu")
    root_dir = "data/raw/Sen1Floods11"

    dataset = FloodDataset(root_dir=root_dir, split="valid")
    loader = DataLoader(dataset, batch_size=2, shuffle=False)

    model = UNet().to(device)
    model.load_state_dict(
        torch.load(
            "best_weighted_unet.pth",
            map_location=device,
            weights_only=True
        )
    )
    model.eval()

    tp = 0
    fp = 0
    fn = 0

    threshold = 0.50

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            masks = batch["mask"].to(device)
            valid_masks = batch["valid_mask"].to(device)

            logits = model(images)
            probabilities = torch.sigmoid(logits)
            predictions = probabilities >= threshold
            targets = masks >= 0.5

            tp += (predictions & targets & valid_masks).sum().item()
            fp += (predictions & ~targets & valid_masks).sum().item()
            fn += (~predictions & targets & valid_masks).sum().item()

    iou = tp / (tp + fp + fn + 1e-7)
    dice = (2 * tp) / (2 * tp + fp + fn + 1e-7)
    precision = tp / (tp + fp + 1e-7)
    recall = tp / (tp + fn + 1e-7)

    print("\nWEIGHTED MODEL — VALIDATION RESULTS")
    print(f"Threshold: {threshold}")
    print(f"IoU:       {iou:.4f}")
    print(f"Dice:      {dice:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")


if __name__ == "__main__":
    main()
