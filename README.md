# DCGAN on Fashion-MNIST

A Deep Convolutional Generative Adversarial Network (DCGAN) implemented using TensorFlow/Keras to generate realistic Fashion-MNIST grayscale images.

The project includes complete GAN training, model visualization, generated-image analysis, loss analysis, and advanced quantitative evaluation using a Fashion-MNIST-specific feature extractor.

---

## Project Overview

Generative Adversarial Networks (GANs) are deep learning models consisting of two neural networks:

- **Generator** – generates synthetic images from random latent vectors.

- **Discriminator** – distinguishes between real images and generated images.

This project implements a **\*\*Deep Convolutional GAN (DCGAN)\*\*** using the Fashion-MNIST dataset.

The Generator learns to create 28 × 28 grayscale fashion images, while the Discriminator learns to distinguish real Fashion-MNIST images from generated samples.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Objectives](#objectives)
- [Dataset](#dataset)
- [Technologies Used](#technologies-used)
- [DCGAN Architecture](#dcgan-architecture)
- [Discriminator Architecture](#discriminator-architecture)
- [GAN Training](#gan-training)
- [Loss Functions](#loss-functions)
- [Optimizer](#optimizer)
- [Training Configuration](#training-configuration)
- [Training Analysis](#training-analysis)
- [Model Evaluation](#model-evaluation)
- [Advanced Evaluation Metrics](#advanced-evaluation-metrics)
- [Generated Image Analysis](#generated-image-analysis)
- [Project Workflow](#project-workflow)
- [Project Outputs](#project-outputs)
- [Important Notebook Sections](#important-notebook-sections)
- [Applications](#applications)
- [Limitations](#limitations)
- [Future Scope](#future-scope)
- [Conclusion](#conclusion)
- [Requirements](#requirements)
- [How to Run](#how-to-run)
- [Project Structure](#project-structure)
- [Author](#author)
- [Keywords](#keywords)

## Objectives

The main objectives of this project are:

- Understand the working principle of Generative Adversarial Networks.

- Implement a DCGAN using TensorFlow/Keras.

- Design and train a Generator and Discriminator.

- Generate synthetic Fashion-MNIST images.

- Analyze Generator and Discriminator losses.

- Compare real and generated images visually.

- Save trained Generator and Discriminator models.

- Evaluate generated images using feature-based quantitative metrics.

- Analyze image diversity and nearest-neighbor relationships.

---

## Dataset

### Fashion-MNIST

The Fashion-MNIST dataset contains grayscale images of fashion products.

| Property | Description |

|---|---|

| Dataset | Fashion-MNIST |

| Training Images | 60,000 |

| Image Size | 28 × 28 |

| Channels | 1 (Grayscale) |

| Classes | 10 |

| Pixel Range | 0–255 originally |

| Preprocessed Range | -1 to 1 |

The dataset is loaded directly using TensorFlow/Keras.

---

## Technologies Used

- Python

- TensorFlow

- Keras

- NumPy

- Matplotlib

- Pandas

- SciPy

- Jupyter Notebook

---

# DCGAN Architecture

## Generator

The Generator takes a random latent vector of size **100** and progressively transforms it into a **28 × 28 grayscale image**.

### Generator Architecture

```text
Latent Vector (100)
        ↓
Dense Layer
        ↓
Batch Normalization
        ↓
ReLU
        ↓
Reshape (7 × 7 × 128)
        ↓
Conv2DTranspose (64 filters)
        ↓
Batch Normalization
        ↓
ReLU
        ↓
Conv2DTranspose (32 filters)
        ↓
Batch Normalization
        ↓
ReLU
        ↓
Conv2DTranspose (1 filter)
        ↓
Tanh
        ↓
Generated Image (28 × 28 × 1)
```

### Generator Configuration

| Parameter              | Value       |
| ---------------------- | ----------- |
| Latent Dimension       | 100         |
| Output Size            | 28 × 28 × 1 |
| Activation             | ReLU        |
| Output Activation      | Tanh        |
| Batch Normalization    | Used        |
| Transposed Convolution | Used        |

---

## Discriminator Architecture

The Discriminator receives a **28 × 28 grayscale image** and produces a prediction score indicating whether the image is real or generated.

### Discriminator Architecture

```text
Input Image (28 × 28 × 1)
        ↓
Conv2D (64 filters)
        ↓
LeakyReLU
        ↓
Dropout
        ↓
Conv2D (128 filters)
        ↓
LeakyReLU
        ↓
Dropout
        ↓
Flatten
        ↓
Dense (1)
        ↓
Discriminator Score
```

### Discriminator Configuration

| Parameter            | Value        |
| -------------------- | ------------ |
| Convolutional Layers | 2            |
| Filters              | 64 and 128   |
| Activation           | LeakyReLU    |
| Dropout              | 0.3          |
| Final Output         | Single logit |
| Sigmoid Activation   | Not used     |

The Discriminator uses logits together with binary cross-entropy loss configured with `from_logits=True`.

---

## GAN Training

The GAN is trained using an adversarial learning process.

### Training Process

```text
Random Noise
     ↓
Generator
     ↓
Fake Images
     ↓
Discriminator
     ↓
Fake Score
```

```text
Real Images
     ↓
Discriminator
     ↓
Real Score
```

### During Training

* The Generator creates synthetic images from random noise.
* The Discriminator receives both real and generated images.
* The Discriminator learns to distinguish real from fake images.
* The Generator learns to produce images that can fool the Discriminator.
* Both networks are updated repeatedly through adversarial training.

---

## Loss Functions

Binary Cross-Entropy is used for both networks.

### Discriminator Loss

The Discriminator learns from:

* Real images with target label **1**
* Generated images with target label **0**

**Discriminator Loss:**

```text
Real Image Loss + Fake Image Loss
```

### Generator Loss

The Generator attempts to make generated images appear real to the Discriminator.

**Generator Loss:**

```text
Binary Cross-Entropy(Fake Images, Real Labels)
```

---

## Optimizer

The project uses the **Adam optimizer**.

| Parameter     |  Value |
| ------------- | -----: |
| Learning Rate | 0.0002 |
| Beta 1        |    0.5 |

These settings are used for both the Generator and Discriminator.

---

## Training Configuration

| Parameter        | Value         |
| ---------------- | ------------- |
| Dataset          | Fashion-MNIST |
| Image Size       | 28 × 28       |
| Batch Size       | 128           |
| Latent Dimension | 100           |
| Epochs           | 20            |
| Learning Rate    | 0.0002        |
| Adam Beta 1      | 0.5           |
| Image Channels   | 1             |
| Generator Output | 28 × 28 × 1   |

A fixed random-noise vector is also used to visualize Generator progress consistently during training.

---

## Training Analysis

The notebook analyzes GAN training using:

* Generator loss
* Discriminator loss
* Combined loss
* Training progress images
* Real vs. generated image comparison

The loss graphs are saved for further analysis.

Generated images are also saved at selected training stages:

* Epoch 001
* Epoch 005
* Epoch 010
* Epoch 015
* Epoch 020

This allows visual inspection of how image generation develops during training.

---

## Model Evaluation

Visual inspection alone is not sufficient to evaluate generated images.

Therefore, this project includes feature-based evaluation using a **CNN feature extractor trained specifically on Fashion-MNIST**.

The feature extractor produces a **128-dimensional feature representation** for each image.

Both real and generated images are converted into these feature representations for quantitative comparison.

---

## Advanced Evaluation Metrics

The project evaluates generated images using three main feature-based measures.

### 1. Domain-Specific Fréchet Feature Distance

A Fréchet-style distance is calculated between the feature distributions of real and generated Fashion-MNIST images.

**Final Result:**

```text
Fréchet Feature Distance: 1.599898
```

This is a domain-specific feature distance based on the Fashion-MNIST-trained feature extractor.

> **Note:** It should not be interpreted as a standard ImageNet/Inception FID value.

### 2. Feature Diversity

The average standard deviation of generated feature representations is used as a diversity measure.

**Result:**

```text
Average Feature Standard Deviation: 0.768950
```

This provides a quantitative view of variation among generated samples in feature space.

### 3. Nearest-Neighbor Analysis

Nearest-neighbor analysis compares generated samples with real Fashion-MNIST samples in feature space.

**Results:**

| Metric                            |      Value |
| --------------------------------- | ---------: |
| Mean Nearest-Neighbor Distance    |  2.6802032 |
| Minimum Nearest-Neighbor Distance | 0.49948868 |
| Maximum Nearest-Neighbor Distance |  6.4414353 |

This analysis helps examine how close generated samples are to real samples in the learned feature space.

---

## Generated Image Analysis

The project performs several forms of visual analysis:

* Real image visualization
* Generated image visualization
* Real vs. generated comparison
* Training progress visualization
* Final generated sample grid
* Nearest-neighbor distance visualization

These visualizations complement the quantitative evaluation.

---

## Project Workflow

```text
Load Fashion-MNIST
        ↓
Explore Dataset
        ↓
Preprocess Images
        ↓
Create Training Dataset
        ↓
Build Generator
        ↓
Build Discriminator
        ↓
Define Loss Functions
        ↓
Define Optimizers
        ↓
Create Training Step
        ↓
Train DCGAN
        ↓
Save Models
        ↓
Generate Images
        ↓
Analyze Training Loss
        ↓
Compare Real and Generated Images
        ↓
Build Feature Extractor
        ↓
Extract Feature Representations
        ↓
Calculate Fréchet Feature Distance
        ↓
Analyze Diversity
        ↓
Perform Nearest-Neighbor Analysis
        ↓
Save Evaluation Metrics
```

---

## Project Outputs

The project generates and saves the following outputs.

### Models

```text
models/
├── generator/
│   └── dcgan_generator.keras
└── discriminator/
    └── dcgan_discriminator.keras
```

### Evaluation and Visualization Outputs

```text
generator_loss.png
discriminator_loss.png
combined_loss.png
real_vs_generated.png
final_generated_samples.png
evaluation_metrics.csv
```

Training progress images are also generated at selected epochs.

---

## Important Notebook Sections

The notebook is organized into the following major stages:

1. Project Title
2. Objective
3. Problem Statement
4. Introduction
5. GAN Concept
6. Why DCGAN?
7. Project Workflow
8. Dataset Description
9. Libraries and Environment
10. Load Fashion-MNIST
11. Dataset Exploration
12. Real Image Visualization
13. Data Preprocessing
14. Training Dataset Preparation
15. Generator Definition
16. Generator Architecture Analysis
17. Discriminator Definition
18. Discriminator Architecture
19. Model Summaries
20. Loss Functions
21. Optimizers
22. Training Step
23. Fixed Noise and Image Generation
24. DCGAN Training
25. Save Trained Models
26. Final Generated Images
27. Generator Loss Analysis
28. Discriminator Loss Analysis
29. Combined Loss Analysis
30. Save Loss Graphs
31. Real vs. Generated Comparison
32. Save Comparison
33. Pixel Statistics
34. Advanced Evaluation Preparation
35. Evaluation Dataset
36. Feature Extractor
37. Feature Extractor Training
38. Feature Representation Extraction
39. Fréchet Feature Distance
40. Diversity Analysis
41. Nearest-Neighbor Analysis
42. Nearest-Neighbor Visualization
43. Evaluation Metrics Summary
44. Save Metrics to CSV
45. Training Progress Visualization
46. Final Generated Sample Grid
47. Final Model Evaluation Summary
48. Applications
49. Limitations
50. Future Scope
51. Conclusion
52. Final Project Output Checklist

---

## Applications

GAN-based image generation can be applied to areas such as:

* Synthetic image generation
* Data augmentation
* Computer vision research
* Image synthesis
* Generative AI
* Creative content generation
* Simulation and experimentation
* Research in deep generative models

---

## Limitations

The project has several limitations:

* Fashion-MNIST images are low resolution.
* Generated images may contain artifacts or unclear structures.
* GAN training can be unstable.
* Generator and Discriminator losses may fluctuate during training.
* Training quality depends on architecture and hyperparameter selection.
* Quantitative evaluation depends on the feature extractor used.
* The domain-specific Fréchet feature distance is not directly comparable with standard Inception FID values.

---

## Future Scope

Possible improvements include:

* Training for more epochs.
* Hyperparameter tuning.
* Increasing Generator and Discriminator capacity.
* Experimenting with different latent dimensions.
* Using higher-resolution datasets.
* Implementing Wasserstein GAN (WGAN).
* Implementing WGAN-GP.
* Exploring conditional GANs.
* Applying advanced image-quality metrics.
* Using larger and more diverse datasets.
* Improving training stability.

---

## Conclusion

This project demonstrates the implementation of a **Deep Convolutional Generative Adversarial Network (DCGAN)** using TensorFlow/Keras and the Fashion-MNIST dataset.

The project covers the complete workflow from dataset preprocessing and DCGAN architecture design to adversarial training, image generation, visualization, model saving, and quantitative evaluation.

The additional feature-based evaluation provides a more systematic analysis of generated samples beyond visual inspection and GAN training losses.

Overall, the project provides practical experience with deep generative modeling, convolutional neural networks, adversarial learning, image synthesis, and quantitative model evaluation.

---

## Requirements

Install the required Python libraries using:

```bash
pip install tensorflow numpy matplotlib pandas scipy jupyter
```

## How to Run

### Step 1: Clone or Download the Project

```bash
git clone <repository-url>

Step 2: Open the project directory

cd \<project-directory>

Step 3: Start Jupyter Notebook

```bash
jupyter notebook
```

Step 4: Open the DCGAN notebook

Run the notebook cells sequentially from the beginning.

The notebook will:

Load Fashion-MNIST.

Preprocess the images.

Build the Generator.

Build the Discriminator.

Train the DCGAN.

Generate synthetic images.

Save trained models.

Generate loss graphs.

Perform advanced evaluation.

Save evaluation results.

## Project Structure
DCGAN-Fashion-MNIST/

│

├── DCGAN_Fashion_MNIST.ipynb

│

├── models/

│   ├── generator/

│   │   └── dcgan_generator.keras

│   │

│   └── discriminator/

│       └── dcgan_discriminator.keras

│

├── generator_loss.png

├── discriminator_loss.png

├── combined_loss.png

├── real_vs_generated.png

├── final_generated_samples.png

├── evaluation_metrics.csv

│

└── README.md

## Author

**Armi Sherathiya**

M.Tech – Artificial Intelligence and Machine Learning

## Keywords

`GAN` `DCGAN` `Generative AI` `Deep Learning` `Fashion-MNIST` `TensorFlow` `Keras` `Computer Vision` `Image Generation` `Image Synthesis` `CNN` `Generative Models`

