# Loss Function Experiments — FloodLens

## 1. Objective

Evaluate whether alternative loss functions improve water segmentation performance under class imbalance.

Three loss configurations were investigated:

1. BCE + Dice Loss (baseline)
2. Weighted BCE + Dice Loss
3. Focal Loss + Dice Loss

The experiments use the Sen1Floods11 hand-labelled dataset and a U-Net segmentation architecture.

## 2. Class Imbalance Analysis

The training and validation datasets contain substantially more background pixels than water pixels.

| Split | Water pixels | Background pixels | Water percentage |
|---|---:|---:|---:|
| Train | 5,446,678 | 51,832,579 | 9.51% |
| Validation | 2,231,206 | 18,042,948 | 11.01% |

The training set has approximately 9.52 background pixels for every water pixel.

This imbalance motivated experiments with class weighting and focal loss.

## 3. Experiment 1 — BCE + Dice Loss

**Configuration**

- Loss: Combined BCE and Dice Loss
- BCE weight: 0.5
- Dice weight: 0.5
- Optimizer: Adam
- Learning rate: 0.001
- Batch size: 2
- Epochs: 5

This configuration served as the baseline.

## 4. Experiment 2 — Weighted BCE + Dice Loss

**Configuration**

- Loss: Weighted BCE + Dice Loss
- Positive class weight: 9.52
- BCE weight: 0.5
- Dice weight: 0.5

The positive class weight was based on the training-set background-to-water pixel ratio.

**Validation results at threshold 0.50**

| Metric | Result |
|---|---:|
| IoU | 0.5158 |
| Dice | 0.6805 |
| Precision | 0.5851 |
| Recall | 0.8132 |

**Observation**

Weighted BCE increased recall but reduced precision and overall IoU compared with the original baseline.

The model detected more water pixels but also generated more false positives.

## 5. Experiment 3 — Focal Loss + Dice Loss

**Configuration**

- Alpha: 0.75
- Gamma: 2.0
- Focal weight: 0.5
- Dice weight: 0.5
- Optimizer: Adam
- Learning rate: 0.001
- Batch size: 2
- Epochs: 5

Focal loss reduces the contribution of confidently classified pixels and places greater emphasis on difficult examples.

### Initial validation result

At threshold 0.50:

| Metric | Result |
|---|---:|
| IoU | 0.5573 |
| Dice | 0.7157 |
| Precision | 0.6628 |
| Recall | 0.7778 |

A validation threshold sweep showed that 0.70 achieved an IoU of 0.5659. However, the multi-seed comparison below uses the common threshold of 0.50 for both models.

## 6. Reproducibility Experiments

To reduce dependence on a single random initialization and data-shuffling sequence, both the baseline and focal models were trained using three random seeds:

- 42
- 123
- 2026

All models used the same architecture, training configuration, validation split and evaluation threshold.

### Individual seed results

**Baseline — BCE + Dice**

| Seed | IoU | Dice | Precision | Recall |
|---|---:|---:|---:|---:|
| 42 | 0.5741 | 0.7294 | 0.8187 | 0.6577 |
| 123 | 0.5746 | 0.7299 | 0.7320 | 0.7277 |
| 2026 | 0.5533 | 0.7124 | 0.7721 | 0.6612 |

**Focal + Dice**

| Seed | IoU | Dice | Precision | Recall |
|---|---:|---:|---:|---:|
| 42 | 0.5477 | 0.7077 | 0.6387 | 0.7936 |
| 123 | 0.5345 | 0.6966 | 0.6207 | 0.7938 |
| 2026 | 0.5538 | 0.7128 | 0.6490 | 0.7906 |

### Mean and standard deviation

| Metric | Baseline | Focal |
|---|---:|---:|
| IoU | 0.5673 ± 0.0122 | 0.5453 ± 0.0099 |
| Dice | 0.7239 ± 0.0100 | 0.7057 ± 0.0083 |
| Precision | 0.7743 ± 0.0434 | 0.6361 ± 0.0143 |
| Recall | 0.6822 ± 0.0394 | 0.7927 ± 0.0018 |

All metrics were calculated using the validation set at threshold 0.50. Standard deviations are across the three seeds.

## 7. Comparative Analysis

The experiments revealed a consistent precision-recall trade-off.

- The baseline achieved higher mean IoU and Dice.
- The baseline achieved substantially higher mean precision.
- Focal loss achieved higher mean recall across all three seeds.
- Focal loss produced more false-positive predictions.
- Weighted BCE + Dice also increased recall but did not improve overall IoU.

Although focal loss produced lower validation loss during training, this did not translate into better IoU or Dice.

## 8. Final Model Selection

**Selected primary model: U-Net with BCE + Dice Loss.**

The baseline was selected because it achieved better mean IoU, Dice and precision across three independent seeds.

Focal loss remains a useful alternative for applications where missing water regions is particularly costly and higher recall is preferred.

Weighted BCE + Dice was not selected because its overall validation IoU was lower than the baseline.

## 9. Limitations

- Only three random seeds were evaluated.
- Training was limited to five epochs per experiment.
- Experiments were performed on CPU.
- Threshold 0.50 was used for the controlled multi-seed comparison.
- Results are based on the Sen1Floods11 validation split.
- These experiments do not establish deployment readiness or performance on unseen geographic regions.
- Further experiments are needed to evaluate longer training, augmentation and generalization across geographic regions.

The test split was not used for loss-function or threshold selection.

## 10. Conclusion

The experiments demonstrate that changing the loss function affects the precision-recall balance of the FloodLens segmentation model.

For the current experimental setup, BCE + Dice provides the strongest overall validation performance and remains the primary model.