# CIFAR-10 Object Recognition using ResNet-50

This project implements a deep learning-based image classification model using **ResNet-50** to recognize objects from the **CIFAR-10** dataset. CIFAR-10 consists of 60,000 32x32 color images across 10 classes. The model is built using **TensorFlow/Keras** and leverages **transfer learning** via a pre-trained ResNet-50 architecture.

<p align="center">
  <img src="https://upload.wikimedia.org/wikipedia/commons/3/3f/CIFAR-10_image.png" alt="CIFAR-10 Sample" width="600"/>
</p>

---

## 🔍 Project Overview

The goal of this project is to classify small-scale images into one of the following 10 categories:

- Airplane
- Automobile
- Bird
- Cat
- Deer
- Dog
- Frog
- Horse
- Ship
- Truck

Using **ResNet-50** (trained on ImageNet), we fine-tune the model on the CIFAR-10 dataset, optimizing for validation accuracy while avoiding overfitting.

---

## 🧠 Model Architecture

- **Base Model**: ResNet-50 (with pre-trained ImageNet weights)
- **Input Size**: Resized CIFAR-10 images to 224x224 (to match ResNet-50 input requirements)
- **Frozen Layers**: All convolutional layers frozen for feature extraction
- **Trainable Head**:
  - Global Average Pooling
  - Dense(512, ReLU) + Dropout
  - Dense(10, Softmax)

---

## 🛠️ Setup Instructions

1. **Install Dependencies**
   ```bash
   pip install tensorflow numpy matplotlib sklearn
Download Dataset (Kaggle)
To use the dataset via Kaggle API:
kaggle datasets download -d tensorflow/cifar10
Run Training
python main.py
⚠️ Important Note: Use Cloud Computing

Training on CIFAR-10 using ResNet-50 is computationally intensive and may take several hours on CPU or even mid-tier GPUs.
We strongly recommend using cloud platforms with GPU support such as:

Google Colab (Free NVIDIA Tesla T4)
Kaggle Kernels
Amazon SageMaker
Paperspace
To run on Colab:

# Add to the top of your notebook
!pip install kaggle
from google.colab import drive
drive.mount('/content/drive')
📈 Training Details

Epochs: 20 (can be adjusted)
Batch Size: 32
Optimizer: Adam
Loss Function: Categorical Crossentropy
Augmentation: Horizontal Flip, Random Rotation, Normalization
📊 Performance Metrics

Metric	Value
Accuracy	~86–90%
Loss	Reduced steadily with early stopping
Confusion Matrix	Included for detailed class-level analysis
📁 Project Structure

├── main.py                  # Training script
├── models/                  # Saved model and weights
├── utils/                   # Preprocessing & augmentation functions
├── outputs/                 # Logs, confusion matrix, plots
└── README.md
📌 Key Learnings

Transfer Learning with ResNet-50 improves performance significantly.
Resizing CIFAR-10 to match ResNet input (224x224) is essential.
Cloud training with GPU acceleration is almost mandatory for practical training time.
Model evaluation through both metrics and visualization helps in interpretability.
