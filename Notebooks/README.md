# Custom CNN, ResNet and DenseNet

In these notebooks, we build three different deep learning models for skin lesion classification using the HAM10000 dataset, as already mentioned. For this project, we have implemented and tested the following models, each for balance and imbalance dataset.

## Custom CNN

The first model we implemented is a custom Convolutional Neural Network (CNN) architecture. CNNs are the fundamental architecture for image classification tasks, making them a natural baseline choice for skin lesion analysis. Their key strength lies in their ability to automatically learn hierarchical feature representations from image data by detecting basic features like edges and textures in early layers, while deeper layers identify more complex patterns specific to different types of skin lesions.

Our custom CNN architecture consists of two main parts:

![Custom CNN Architecture](images_nb/DeepL-4.png)

### Convolutional Layers (Feature Extraction)

Starting from 224x224 RGB image, our architecture employed 4 convolutional blocks, each with progressively increasing filter counts.

* 4 convolutional layers, with 3x3 kernels
* Each layer progressively increases the number of filters (64 → 128 → 256 → 512)
* This gradual expansion of feature channels allows the network to learn from simple textures in early layers to complex lesion structures in deeper layers.
* Batch Normalization is applied after each convolutional layer to stabilize activations and improve convergence
* ReLU activation introduces non-linearity, enabling the network to learn complex patterns
* MaxPooling (2x2, stride=2) reduces spatial dimensions while retaining key information, effectively downsampling the feature maps

### Fully Connected Layers (Classification)

After feature extraction, the final feature map (512x14x14) is flattened and passed through:

* FC1: 512x14x14 → 512 neurons - Extracts high-level lesion patterns
* FC2: 512 → 128 neurons - Refines learned representations for better generalization
* FC3: 128 → 7 neurons - Outputs class probabilities for the 7 skin lesion types.

