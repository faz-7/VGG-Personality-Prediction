# VGG Model for Big-Five Personality Prediction

## Overview
This repository contains an implementation of a **VGG-based deep learning model** for predicting the **Big-Five personality traits** from audio data. The model is trained and evaluated on the **ChaLearn First Impression Tiny dataset**, which consists of **100 video clips**, each labeled with personality trait scores.

## Dataset
The dataset used in this project is a subset of the **ChaLearn First Impression dataset (2016)**. This dataset contains **short video clips** labeled with the **Big-Five personality traits**:
- **Openness**
- **Conscientiousness**
- **Extraversion**
- **Agreeableness**
- **Neuroticism**

### Dataset Access
You can access the dataset and its details from the official ChaLearn website:  
[ChaLearn First Impression Dataset](https://chalearnlap.cvc.uab.cat/dataset/24/description/)

### Data Preprocessing
- **Audio Extraction:** Extracts raw audio files from the video dataset.
- **Feature Extraction:** Uses **VGGish** to convert raw audio into feature embeddings.
- **Normalization & Padding:** Ensures consistent input size for training.

## Model Architecture
The model is based on **VGG** (a convolutional neural network originally designed for image classification) and has been adapted to work with **audio spectrogram features**.

### Key Components:
- **5 convolutional blocks** with ReLU activations and max pooling.
- **Fully connected layers** leading to a **5-dimensional output** for personality trait predictions.
- **Sigmoid activation** in the output layer to scale predictions between 0 and 1.

## Installation
To set up the environment, clone this repository and install the required dependencies:
```bash
# Clone repository
git clone https://github.com/yourusername/vgg-personality-prediction.git
cd vgg-personality-prediction

# Install dependencies
pip install -r requirements.txt
```

## Training the Model
To train the model, run:
```bash
python train.py
```
### Hyperparameters:
Due to **Google Colab limitations** and the **large dataset size**, the batch size had to be kept small to fit within the memory constraints.
- **Batch size:** 4 (training), 2 (validation), 2 (test)
- **Learning rate:** 0.01
- **Optimizer:** SGD
- **Loss Function:** MSE for training, L1 loss for evaluation

## Evaluation Metrics
The model is evaluated using:
- **Mean Absolute Error (MAE)**
- **R-squared (R2) Score**
- **Concordance Correlation Coefficient (CCC)**
- **Pearson Correlation Coefficient (PCC)**

## Results & Analysis
- The model outputs **predicted trait scores** for each sample.
- A **regression-to-the-mean analysis** is performed to check model bias.
- Model interpretability is explored using **SHAP (SHapley Additive Explanations)**.

## Usage
To make predictions on new data:
```python
from model import Vgg
import torch

model = Vgg(numChannels=1, classes=5)
model.load_state_dict(torch.load("best_model.pth"))
model.eval()
```
