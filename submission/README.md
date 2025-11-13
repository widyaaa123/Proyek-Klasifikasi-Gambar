# Butterfly Image Classification Project 🦋

This project focuses on classifying **40 species of butterflies** using a Convolutional Neural Network (CNN) built with TensorFlow and Keras.

## Dataset
The dataset is sourced from Kaggle:
> [Butterfly Images - 40 Species](https://www.kaggle.com/datasets/gpiosenka/butterfly-images40-species)

After downloading, all images from the original train, test, and validation sets were merged and re-split into:
- **70%** training data  
- **15%** validation data  
- **15%** testing data  

## Model Performance
- **Train Accuracy:** 92.48%
- **Validation Accuracy:** 87.81%
- **Test Accuracy:** 88.49%
- **Test Loss:** 0.4565

The model achieves strong accuracy across datasets, showing good generalization on unseen butterfly images.

## Model Architecture
The CNN model includes:
- Multiple **Conv2D** layers with ReLU activation and **BatchNormalization**
- **MaxPooling2D** for downsampling
- **GlobalAveragePooling2D** before the dense layers
- **Dropout (0.4)** for regularization
- **Softmax output layer** for 40-class classification

**Optimizer:** `Adam (lr=5e-4)`  
**Loss Function:** `Categorical Crossentropy`

## Data Augmentation
To improve robustness and prevent overfitting, the training data was augmented using:
- Rotation (±25°)
- Width/height shift (±10%)
- Zoom (±20%)
- Shear transformation
- Horizontal flip
- Brightness adjustment (0.8–1.2)

## Example Prediction
Example uploaded image (`contoh.jpg`) was predicted as:
> **Class: PEACOCK**

The image was also displayed in Colab with the predicted label overlayed.

## Model Export
The trained model was exported to multiple formats for deployment:
- **TensorFlow SavedModel** → `submission/saved_model/`
- **TensorFlow Lite (.tflite)** → `submission/tflite/model.tflite`
- **TensorFlow.js** → `submission/tfjs_model/`

Each format allows easy integration across platforms (mobile, web, or backend).

## Summary
- The CNN successfully recognizes 40 butterfly species with ~88% accuracy.
- Augmentation and Batch Normalization improved model stability.
- The model is ready for deployment in both **mobile (TFLite)** and **web (TF.js)** environments.

---

**Built with:**
- Python 🐍  
- TensorFlow + Keras 🤖  
- Google Colab ☁️  
- Kaggle Dataset 📊
