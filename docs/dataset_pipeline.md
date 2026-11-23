
# FloodLens – Dataset Pipeline

## Dataset
- Source: Sen1Floods11
- Training samples: 252
- SAR channels: VV and VH
- Image dimensions: 512 × 512
- Image format: GeoTIFF
- Label format: GeoTIFF

## Label Encoding
- 0: Non-water
- 1: Water
- -1: NoData (excluded from training)

## Preprocessing
- Read VV and VH as separate channels.
- Replace invalid SAR values safely.
- Clip each channel using training-set 1st and 99th percentiles.
- Standardize VV and VH independently.
- Exclude invalid SAR and label pixels using a validity mask.

## Normalization Parameters

| Parameter | VV | VH |
|---|---:|---:|
| Clip minimum | -23.793211 | -34.037746 |
| Clip maximum | -2.4567175 | -9.528764 |
| Mean | -10.379914 | -17.186045 |
| Standard deviation | 3.8586926 | 4.449738 |

These are estimated from sampled training pixels.

## PyTorch Pipeline
- Custom FloodDataset implementation
- DataLoader with batch size 4
- Image tensor: [B, 2, 512, 512]
- Mask tensor: [B, 1, 512, 512]

## Validation
- 252 training samples loaded successfully.
- VV and VH channels verified.
- Mask labels verified.
- NoData exclusion verified.
- Batch loading verified.
