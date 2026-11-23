
from torch.utils.data import DataLoader
from dataset import FloodDataset

dataset = FloodDataset(
    root_dir="data/raw/Sen1Floods11",
    split="train"
)

print("Dataset size:", len(dataset))

sample = dataset[0]
print("Normalized image min:", sample["image"].min().item())
print("Normalized image max:", sample["image"].max().item())

for channel, name in enumerate(["VV", "VH"]):
    values = sample["image"][channel]
    print(
        f"{name} mean: {values.mean().item():.4f}, "
        f"std: {values.std().item():.4f}"
    )

print("Image shape:", sample["image"].shape)
print("Mask shape:", sample["mask"].shape)
print("Valid mask shape:", sample["valid_mask"].shape)

print("Image dtype:", sample["image"].dtype)
print("Mask values:", sample["mask"].unique())
print("Valid pixels:", sample["valid_mask"].sum().item())

loader = DataLoader(
    dataset,
    batch_size=4,
    shuffle=True
)

batch = next(iter(loader))

print("\nBatch image shape:", batch["image"].shape)
print("Batch mask shape:", batch["mask"].shape)
print("Batch valid mask shape:", batch["valid_mask"].shape)
