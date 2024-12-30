# Pneumonia Detection using Deep Learning

This project aims to classify chest X-ray images into two categories: **Normal** and **Pneumonia** using a convolutional neural network (CNN). The model is trained on a labeled dataset and is designed to assist in the early detection of pneumonia.

---

## Table of Contents
- [Overview](#overview)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [How to Use](#how-to-use)
- [Requirements](#requirements)
- [Acknowledgments](#acknowledgments)

---

## Overview
Pneumonia is a respiratory infection that affects the lungs, and early detection is critical for treatment. This project utilizes deep learning to automate the detection process, potentially assisting medical professionals in diagnosis.

Key features of this project include:
- Pre-trained VGG16 architecture for feature extraction.
- Data augmentation for better generalization.
- Flask integration for user-friendly web-based predictions.

---

## Dataset
The dataset contains chest X-ray images classified into two categories:
1. **Normal**
2. **Pneumonia**

- **Training Data**: 5232 images
- **Testing Data**: 624 images

Data augmentation techniques such as rescaling, zooming, and horizontal flipping are applied during training to improve model performance.

---

## Model Architecture
The project employs a modified version of the **VGG16** architecture:
- Input Layer: 224x224 RGB images.
- Intermediate Layers: Pre-trained VGG16 layers for feature extraction.
- Fully Connected Layer: Custom dense layers for classification.
- Output Layer: Softmax activation for binary classification.

The model is trained using:
- **Optimizer**: Adam
- **Loss Function**: Categorical Crossentropy
- **Evaluation Metric**: Accuracy

---

## How to Use
### 1. Clone the Repository
```bash
git clone https://github.com/your-repo/pneumonia-detection.git
cd pneumonia-detection
```

### 2. Install Dependencies
Install the required libraries using:
```bash
pip install -r requirements.txt
```

### 3. Train the Model
To train the model from scratch, open the `Pneumonia Detection using Deep Learning.ipynb` notebook and execute the cells.

### 4. Use the Pre-trained Model
A pre-trained model file (`our_model.h5`) is included in the project. Use this file for predictions without retraining.

### 5. Run the Web Application
Launch the Flask-based web application for user-friendly predictions:
```bash
python app.py
```
Visit `http://127.0.0.1:5000` in your web browser to upload X-ray images and receive predictions.

---

## Requirements
- Python 3.x
- TensorFlow
- Keras
- Flask
- NumPy
- Matplotlib

---

## Acknowledgments
- The Chest X-ray dataset used in this project.
- TensorFlow and Keras libraries for enabling deep learning implementations.

