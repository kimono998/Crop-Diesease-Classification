# Cassava Disease Image Classification

## Problem Statement

This project involves approximating a function that classifies images \( X \) of cassava leaves into the correct disease categories \( Y \).

## Dataset

The dataset consists of images labeled into five classes:

- **"0"**: Cassava Bacterial Blight (CBB)
- **"1"**: Cassava Brown Streak Disease (CBSD)
- **"2"**: Cassava Green Mottle (CGM)
- **"3"**: Cassava Mosaic Disease (CMD)
- **"4"**: Healthy

The dataset includes the following number of samples per class:

- CBB: 921
- CBSD: 1831
- CGM: 1993
- CMD: 11027
- Healthy: 2166

- **Input Dimensions**: \( X \) -> [17938 x 3 x 800 x 600]
- **Output Labels**: \( Y \) -> [17938 x 5]

## Preprocessing

- Images are resized to 256x192 to reduce computational cost.
- The `.csv` file containing image-label pairs was refactored for better usability.

## Models

### State-of-the-Art Architectures

1. **ResNet50**
    - 50-layer deep convolutional neural network (CNN).
    - Pretrained on ImageNet.
    - Uses residual connections to prevent overfitting and improve training.
  
2. **RexNet150**
    - 150-layer deep CNN.
    - Pretrained on ImageNet.
    - Lightweight architecture using 1x1 bottleneck kernels with expanded iterations, which helps reduce model complexity and training time.

### Custom Models

Locally implemented custom models using different combinations of convolutional blocks and fully connected layers.

## Training Parameters

The following hyperparameters were used for training the models:

- **Number of Filters**: 64
- **Kernel Size**: 3
- **Pooling Size**: 2
- **Dense Layer Size**: 512
- **Stride**: 1
- **Learning Rate**: 1e-5
- **Weight Decay**: 1e-5
- **Optimizer**: Adam
- **Loss Weights**:
  \[
  [19.5588,  9.8417,  9.0510,  1.6351,  8.3200]
  \]

## Hyperparameter Search - Bayesian Optimization

Bayesian optimization was used to find the optimal hyperparameters. The best parameters were:

- **Learning Rate**: 0.00024526126311336793
- **Optimizer**: Adam
- **Number of Epochs**: 19
- **Weight Decay**: 0.00021830968390524624
- **Optimal F1 Score**: 0.6623861743301297
