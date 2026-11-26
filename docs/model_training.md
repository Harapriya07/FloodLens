# FloodLens — U-Net Model Training

## 1. Overview

Implemented and trained a baseline U-Net model using PyTorch for satellite image water segmentation with the Sen1Floods11 dataset.

## 2. Model Architecture

- Architecture: U-Net
- Input channels: 2 (Sentinel-1 VV and VH)
- Input dimensions: 512 × 512
- Encoder: 3 convolution blocks
- Bottleneck: 128 channels
- Decoder: 3 upsampling stages
- Skip connections: Used to preserve spatial details
- Output: 1 channel containing water segmentation logits

## 3. Training Configuration

| Parameter | Value |
|---|---|
| Framework | PyTorch |
| Device | CPU |
| Optimizer | Adam |
| Learning rate | 0.001 |
| Batch size | 2 |
| Epochs | 5 |
| Loss function | Combined BCE + Dice Loss |
| Training samples | 252 |
| Validation samples | 89 |

The validity mask excludes NoData pixels from loss calculations.

## 4. Training Results

| Epoch | Training Loss | Validation Loss |
|---|---:|---:|
| 1 | 0.5798 | 0.4384 |
| 2 | 0.4211 | 0.3747 |
| 3 | 0.4028 | 0.3771 |
| 4 | 0.3983 | 0.3573 |
| 5 | 0.3859 | 0.3563 |

Best validation loss: **0.3563**

The best model checkpoint was saved as `best_unet.pth`.

## 5. Validation Metrics

| Metric | Score |
|---|---:|
| IoU | 0.5497 |
| Dice | 0.7094 |
| Precision | 0.6544 |
| Recall | 0.7745 |

Metrics were calculated using a probability threshold of 0.5 and excluding invalid pixels.

## 6. Prediction Visualization

Generated a comparison showing:

- Sentinel-1 SAR input (VV)
- Ground-truth water mask
- U-Net predicted water mask

The visualization helped inspect segmentation quality and identify differences around water boundaries and smaller regions.

## 7. Generated Artifacts

- `best_unet.pth` — best model checkpoint
- `data/processed/validation_metrics.csv` — validation metrics
- `prediction_comparison.png` — prediction visualization

These generated artifacts are excluded from Git unless intentionally versioned.

## 8. Limitations

- The model was trained for only 5 epochs.
- Evaluation so far uses the validation split.
- Results are from a baseline model and require independent test-set evaluation.
- Water segmentation should not automatically be interpreted as newly flooded area.

## 9. Next Steps

- Evaluate on the official test split.
- Analyse false positives and false negatives.
- Improve model performance through controlled experiments.
- Compare the baseline with an alternative segmentation architecture.