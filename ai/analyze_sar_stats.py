
from pathlib import Path

import numpy as np
import rasterio
import pandas as pd

ROOT = Path("data/raw/Sen1Floods11")
CSV = ROOT / "splits" / "flood_train_data.csv"

samples = pd.read_csv(CSV, header=None)

# Collect a reproducible sample of training pixels
rng = np.random.default_rng(42)
channel_samples = [[], []]

for _, row in samples.iterrows():
    path = ROOT / "S1Hand" / row[0]

    with rasterio.open(path) as src:
        image = src.read().astype(np.float32)

    for channel in range(2):
        values = image[channel].ravel()
        values = values[np.isfinite(values)]

        if values.size:
            selected = rng.choice(
                values,
                size=min(5000, values.size),
                replace=False
            )
            channel_samples[channel].append(selected)

for name, chunks in zip(["VV", "VH"], channel_samples):
    values = np.concatenate(chunks)

    lower = np.percentile(values, 1)
    upper = np.percentile(values, 99)

    clipped = np.clip(values, lower, upper)

    print(f"\n{name} normalization parameters")
    print("Clip minimum:", lower)
    print("Clip maximum:", upper)
    print("Mean after clipping:", clipped.mean())
    print("Std after clipping:", clipped.std())