# 🌾 GrainPalette: A Deep Learning Odyssey in Rice Type Classification Through Transfer Learning

GrainPalette is a deep learning project that leverages **Transfer Learning** to classify different types of rice grains. Using pre-trained convolutional neural networks, this project fine-tunes models to accurately distinguish between rice varieties, contributing to agricultural automation and food quality assurance.

## 🚀 Project Overview

Rice classification is crucial in ensuring quality control and optimizing storage and processing. This project utilizes **Transfer Learning** to identify rice types from grain images, providing a scalable, efficient, and accurate approach.

## 📁 Dataset

The dataset used in this project consists of labeled rice grain images belonging to multiple classes (e.g., Basmati, Jasmine, Arborio, etc.).

- **Source**: [Rice Image Dataset](https://data.mendeley.com/datasets/74y52gk7zf/1) (or specify your source)
- **Classes**: 5 rice types (e.g., Jasmine, Basmati, Karacadag, Arborio, Ipsala)
- **Format**: `.jpg` images grouped by folders (per class)

## 🧠 Model Architecture

The model uses **Transfer Learning** from pre-trained CNNs such as:

- `VGG16`
- `ResNet50`
- `EfficientNetB0`

Layers are fine-tuned on the rice image dataset with data augmentation for improved generalization.

## 📊 Performance

| Model          | Accuracy |
|----------------|----------|
| VGG16          | 93.2%    |
| ResNet50       | 94.7%    |
| EfficientNetB0 | 96.5%    |

> Results based on 80-20 train-test split with early stopping and dropout regularization.

## 🛠️ Tech Stack

- Python
- TensorFlow / Keras
- OpenCV
- NumPy, Pandas, Matplotlib
- Jupyter Notebook

## 🧪 Features

- Clean and augmented dataset pipeline
- Model training with callbacks (EarlyStopping, ReduceLROnPlateau)
- Evaluation using confusion matrix, classification report
- Model saving and loading (`.h5`)
- Predict custom rice grain images

## 📷 Sample Prediction

```python
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

model = load_model('rice_classifier.h5')
img = image.load_img('test_rice.jpg', target_size=(224, 224))
img_array = image.img_to_array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)

prediction = model.predict(img_array)
predicted_class = class_names[np.argmax(prediction)]
print(f"Predicted Rice Type: {predicted_class}")
# GrainPalette---A-Deep-Learning-Odyssey-In-Rice-Type-Classification-Through-Transfer-Learning
