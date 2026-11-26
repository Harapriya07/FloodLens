# FloodLens — Official Test Evaluation

## 1. Overview

Evaluated the trained U-Net model on the official Sen1Floods11 test split.

- Test samples: 90
- Model: U-Net
- Framework: PyTorch
- Device: CPU
- Probability threshold: 0.5
- Checkpoint: `best_unet.pth`

## 2. Test Results

| Metric | Score |
|---|---:|
| IoU | 0.5878 |
| Dice | 0.7404 |
| Precision | 0.6969 |
| Recall | 0.7896 |

## 3. Validation vs Test

| Metric | Validation | Test |
|---|---:|---:|
| IoU | 0.5497 | 0.5878 |
| Dice | 0.7094 | 0.7404 |
| Precision | 0.6544 | 0.6969 |
| Recall | 0.7745 | 0.7896 |

## 4. Observations

- Test IoU reached 0.5878.
- Test Dice reached 0.7404.
- Recall was higher than precision.
- Test metrics were higher than validation metrics.

## 5. Limitations

- These metrics represent aggregate pixel-level performance.
- Results do not establish performance on Odisha-specific flood events.
- Water segmentation does not automatically distinguish permanent water from newly flooded areas.
- Further geographic and event-level validation is required.

## 6. Next Steps

- Perform false-positive and false-negative analysis.
- Inspect predictions across multiple test samples.
- Investigate class imbalance and segmentation errors.
- Conduct controlled model improvement experiments.
