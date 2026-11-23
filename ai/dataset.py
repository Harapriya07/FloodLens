from pathlib import Path
import numpy as np
import pandas as pd
import rasterio
import torch
from torch.utils.data import Dataset

class FloodDataset(Dataset):

    def __init__(self, root_dir, split="train"):

        self.root_dir = Path(root_dir)

        self.image_dir = self.root_dir / "S1Hand"
        self.label_dir = self.root_dir / "LabelHand"

        split_file = (
            self.root_dir / "splits" / f"flood_{split}_data.csv"
        )

        self.samples = pd.read_csv(split_file, header=None)

    def __len__(self):
        return len(self.samples)

    
    def __getitem__(self, index):

        image_name = self.samples.iloc[index, 0]
        label_name = self.samples.iloc[index, 1]

        image_path = self.image_dir / image_name
        label_path = self.label_dir / label_name

        with rasterio.open(image_path) as src:
         image = src.read().astype(np.float32)

        with rasterio.open(label_path) as src:
         mask = src.read(1).astype(np.int64)

    # Identify pixels where both SAR channels are finite
        image_valid = np.isfinite(image).all(axis=0)

    # Training-derived VV and VH parameters
        lower = np.array([-23.793211, -34.037746], dtype=np.float32)
        upper = np.array([-2.4567175, -9.528764], dtype=np.float32)
        means = np.array([-10.379914, -17.186045], dtype=np.float32)
        stds = np.array([3.8586926, 4.449738], dtype=np.float32)

    # Clip each channel independently
        image = np.clip(
        image,
        lower[:, None, None],
        upper[:, None, None]
        ) 

    # Replace invalid SAR pixels with channel means
    # so they become zero after standardization
        image[:, ~image_valid] = means[:, None]

    # Standardize VV and VH independently
        image = (image - means[:, None, None]) / stds[:, None, None]

    # Exclude invalid labels and invalid SAR pixels
        valid_mask = (mask != -1) & image_valid

    # Replace ignored labels with background placeholder
        mask = np.where(valid_mask, mask, 0)

    # Convert to tensors
        image = torch.from_numpy(image.copy())
        mask = torch.from_numpy(mask).unsqueeze(0).float()
        valid_mask = torch.from_numpy(valid_mask).unsqueeze(0)

        return {
        "image": image,
        "mask": mask,
        "valid_mask": valid_mask,
        "image_name": image_name
         }
