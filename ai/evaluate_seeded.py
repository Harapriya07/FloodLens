
import argparse
import torch
from torch.utils.data import DataLoader

from ai.dataset import FloodDataset
from ai.unet import UNet


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["baseline", "focal"], required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--threshold", type=float, default=0.50)
    args = parser.parse_args()

    device = torch.device("cpu")

    checkpoint = (
        f"best_unet_seed_{args.seed}.pth"
        if args.model == "baseline"
        else f"best_focal_unet_seed_{args.seed}.pth"
    )

    dataset = FloodDataset(
        root_dir="data/raw/Sen1Floods11",
        split="valid"
    )

    loader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=False
    )

    model = UNet().to(device)
    model.load_state_dict(
        torch.load(
            checkpoint,
            map_location=device,
            weights_only=True
        )
    )
    model.eval()

    tp = fp = fn = 0

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            masks = batch["mask"].to(device)
            valid_mask = batch["valid_mask"].to(device)

            probabilities = torch.sigmoid(model(images))
            predictions = probabilities >= args.threshold
            targets = masks >= 0.5

            tp += (predictions & targets & valid_mask).sum().item()
            fp += (predictions & ~targets & valid_mask).sum().item()
            fn += (~predictions & targets & valid_mask).sum().item()

    iou = tp / (tp + fp + fn + 1e-7)
    dice = 2 * tp / (2 * tp + fp + fn + 1e-7)
    precision = tp / (tp + fp + 1e-7)
    recall = tp / (tp + fn + 1e-7)

    print(f"\n{args.model.upper()} — SEED {args.seed}")
    print(f"Threshold: {args.threshold:.2f}")
    print(f"IoU:       {iou:.4f}")
    print(f"Dice:      {dice:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")


if __name__ == "__main__":
    main()
