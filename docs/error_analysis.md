# FloodLens — Model Error Analysis

## 1. Objective

Analyze U-Net segmentation errors on the Sen1Floods11 validation and test datasets.

The analysis focuses on:
- False positives (FP)
- False negatives (FN)
- Per-image segmentation performance
- Classification threshold selection

## 2. Test Set Error Analysis

At threshold 0.50:

| Error | Pixel count |
|---|---:|
| False positives | 863,528 |
| False negatives | 529,095 |

The worst-performing sample by total error count was `Pakistan_849790_S1Hand.tif`.

- False positives: 3,832
- False negatives: 128,660

## 3. Validation Error Analysis

The validation set revealed different error patterns across geographical regions.

Examples:

| Image | False positives | False negatives |
|---|---:|---:|
| Somalia_12849 | 88,114 | 9,445 |
| Somalia_167787 | 96,407 | 2 |
| Ghana_1033830 | 660 | 56,636 |

These examples demonstrate that the model experiences both over-segmentation and missed-water errors.

## 4. Threshold Analysis

Threshold experiments were performed exclusively on the validation set.

### Initial threshold comparison

| Threshold | IoU | Dice | Precision | Recall |
|---|---:|---:|---:|---:|
| 0.30 | 0.5291 | 0.6920 | 0.6082 | 0.8026 |
| 0.40 | 0.5407 | 0.7019 | 0.6327 | 0.7881 |
| 0.50 | 0.5497 | 0.7094 | 0.6544 | 0.7745 |
| 0.60 | 0.5571 | 0.7156 | 0.6758 | 0.7604 |
| 0.70 | 0.5639 | 0.7212 | 0.7007 | 0.7428 |

### Fine-grained threshold comparison

| Threshold | IoU | Dice | Precision | Recall |
|---|---:|---:|---:|---:|
| 0.70 | 0.5639 | 0.7212 | 0.7007 | 0.7428 |
| 0.75 | 0.5675 | 0.7240 | 0.7180 | 0.7302 |
| 0.80 | 0.5714 | 0.7273 | 0.7430 | 0.7121 |
| 0.85 | 0.5733 | 0.7288 | 0.7789 | 0.6847 |
| 0.90 | 0.5652 | 0.7222 | 0.8291 | 0.6397 |

## 5. Micro vs Macro IoU

| Threshold | Micro IoU | Macro IoU |
|---|---:|---:|
| 0.50 | 0.5497 | 0.3405 |
| 0.55 | 0.5536 | 0.3402 |
| 0.60 | 0.5571 | 0.3396 |
| 0.65 | 0.5606 | 0.3388 |
| 0.70 | 0.5639 | 0.3372 |
| 0.75 | 0.5675 | 0.3348 |
| 0.80 | 0.5714 | 0.3304 |
| 0.85 | 0.5733 | 0.3226 |
| 0.90 | 0.5652 | 0.3201 |

Macro IoU excludes images where the union of predicted and ground-truth water pixels is zero.

At thresholds 0.50–0.85, 85 images were included. At 0.90, 82 images were included.

## 6. Findings

- Higher thresholds improve aggregate precision and micro IoU up to 0.85.
- Higher thresholds reduce recall.
- Macro IoU decreases as the threshold increases.
- Threshold 0.85 achieves the highest tested micro IoU.
- Threshold 0.50 achieves the highest tested macro IoU.

## 7. Current Decision

Retain 0.50 as the baseline threshold.

Keep 0.85 as an alternative threshold for applications prioritizing aggregate pixel-level IoU and precision.

No threshold has been selected as universally optimal.

## 8. Limitations

- Error analysis does not establish the causes of geographical performance differences.
- The validation results do not establish generalization to Odisha.
- The current model segments labelled water; it does not distinguish permanent water from newly flooded areas.
- Threshold experiments were conducted on validation data. The official test set was not used for threshold selection.

## 9. Next Steps

1. Investigate class imbalance.
2. Evaluate improvements to the training loss.
3. Compare model performance on the validation set.
4. Preserve the official test set for final evaluation.
