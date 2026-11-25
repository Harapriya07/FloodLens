# Milestone 3: U-Net Architecture and Loss Functions

## 1. U-Net Architecture

Implemented a U-Net segmentation model using PyTorch.

- Input: 2 channels (Sentinel-1 VV and VH)
- Input dimensions: 512 × 512
- Encoder: 3 convolution blocks
- Bottleneck: 128 feature channels
- Decoder: 3 upsampling stages
- Skip connections: Connect encoder and decoder feature maps
- Output: 1 channel (water segmentation logits)

## 2. Convolution Block

Each block contains:
- Two 3×3 convolution layers
- ReLU activation functions
- Padding to preserve spatial dimensions

## 3. Loss Function

Implemented a combined loss:

- Binary Cross-Entropy with Logits
- Dice Loss
- Validity mask to exclude NoData pixels

## 4. Testing

Verified:
- U-Net output shape matches target mask shape.
- Model works with actual dataset batches.
- Combined loss produces a finite value.
- Backward pass completes successfully.

## 5. Next Step

Implement the training loop and evaluate the model.