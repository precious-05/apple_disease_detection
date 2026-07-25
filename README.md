# Apple Disease Detection using Deep Learning

A comprehensive deep learning solution for automated detection and classification of apple leaf diseases using transfer learning with MobileNetV2. This project achieves high accuracy in identifying four distinct apple disease categories: Normal, Scab, Blotch, and Rot.

---

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Performance Metrics](#performance-metrics)
- [Installation](#installation)
- [Usage](#usage)
- [Results & Visualizations](#results--visualizations)
- [Technologies Used](#technologies-used)
- [Key Features](#key-features)
- [Future Enhancements](#future-enhancements)

---

## Overview

Apple leaf diseases pose significant threats to crop yields and agricultural productivity. Early and accurate detection is crucial for timely intervention and disease management. This project leverages deep learning to automate the classification process, achieving robust performance across multiple disease categories.

**Project Highlights:**
- Implements transfer learning using MobileNetV2
- Processes 1,600+ images across 4 disease categories
- Achieves 90%+ accuracy on test dataset
- Optimized for inference speed and model efficiency
- Production-ready classification pipeline

---

## Dataset

### Data Distribution

![Dataset Distribution](images/dataset_distribution.png)

**Dataset Composition:**

| Class | Training Samples | Testing Samples | Total |
|-------|-----------------|-----------------|-------|
| Normal Apple | 400 | 100 | 500 |
| Scab Apple | 402 | 100 | 502 |
| Blotch Apple | 400 | 100 | 500 |
| Rot Apple | 400 | 100 | 500 |
| **Total** | **1,602** | **400** | **2,002** |

### Dataset Characteristics

**Source:** Kaggle - Apple Disease Classification Dataset

**Image Properties:**
- Format: JPG/PNG
- Resolution: Variable (standardized to 224×224 for model input)
- Color Space: RGB
- Total Size: ~181 MB

**Disease Categories:**

1. **Normal Apple** - Healthy leaves with no visible disease symptoms
2. **Scab Apple** - Fungal infection causing dark scab-like lesions
3. **Blotch Apple** - Brown or black irregular patches on leaf surface
4. **Rot Apple** - Tissue decay and necrosis indicating advanced disease

**Data Augmentation:**
- Random rotation (20°)
- Horizontal and vertical flipping
- Random zoom (0.8-1.2x)
- Brightness and contrast adjustments
- Normalize to ImageNet statistics

---

## Model Architecture

![Model Architecture](images/model_architecture.png)

### Architecture Overview

**Base Model:** MobileNetV2 (Pre-trained on ImageNet)

```
Input Layer (224, 224, 3)
    ↓
MobileNetV2 Convolutional Base (53 layers)
    ├── 32 initial filters
    ├── Depthwise Separable Convolutions
    └── Bottleneck Residual Blocks
    ↓
Global Average Pooling
    ↓
Custom Dense Layers:
    ├── Dense(256, activation='relu', L2=0.0001)
    ├── Dropout(0.5)
    ├── Dense(128, activation='relu', L2=0.0001)
    ├── Dropout(0.3)
    └── Dense(4, activation='softmax')
    ↓
Output: Disease Classification (4 classes)
```

### Why MobileNetV2?

- **Lightweight:** 3.5M parameters vs 25M+ for VGG16
- **Fast Inference:** Optimized for mobile and edge devices
- **Transfer Learning:** Pre-trained on 1.4M ImageNet images
- **Accuracy:** 71.9% top-1 accuracy on ImageNet
- **Efficiency:** Depthwise separable convolutions reduce computation

### Model Parameters

| Component | Configuration |
|-----------|---------------|
| Base Model | MobileNetV2 (frozen weights) |
| Fine-tuning Layers | Last 20 layers (unfrozen) |
| Optimizer | Adam (lr=0.001) |
| Loss Function | Categorical Crossentropy |
| Batch Size | 32 |
| Epochs | 50 |
| Early Stopping | Patience=5 (validation loss) |
| Regularization | L2=0.0001, Dropout=0.5/0.3 |

---

## Performance Metrics

### Training Performance Curves

![Training Performance](images/training_performance.png)

**Observed Patterns:**
- Training accuracy reaches ~97% by epoch 50
- Validation accuracy stabilizes at ~93%
- Minimal overfitting gap (training-validation: 4%)
- Training loss converges smoothly to ~0.15
- Validation loss plateaus around ~0.25

### Classification Results

![Classification Metrics](images/classification_metrics.png)

**Per-Class Performance on Test Set:**

| Disease Class | Precision | Recall | F1-Score | Accuracy |
|---------------|-----------|--------|----------|----------|
| Normal Apple | 0.92 | 0.90 | 0.91 | 92% |
| Scab Apple | 0.88 | 0.91 | 0.89 | 88% |
| Blotch Apple | 0.90 | 0.87 | 0.88 | 90% |
| Rot Apple | 0.91 | 0.92 | 0.91 | 91% |
| **Weighted Avg** | **0.90** | **0.90** | **0.90** | **90.3%** |

### Confusion Matrix Analysis

![Confusion Matrix](images/confusion_matrix.png)

**Insights:**
- High diagonal values indicate strong classification accuracy
- Most confusion occurs between Scab and Blotch classes (visually similar)
- Normal apples are correctly identified 90% of the time
- Rot disease has the highest recall (92%)

### Per-Class Accuracy

![Per-Class Accuracy](images/per_class_accuracy.png)

**Key Observations:**
- Normal Apple: 92% - Clearly distinguishable from diseased samples
- Scab Apple: 88% - Slightly lower due to similarity with Blotch
- Blotch Apple: 90% - Moderate accuracy with some confusion
- Rot Apple: 91% - High accuracy, advanced disease stage is distinctive

---

## Installation

### Prerequisites

- Python 3.8 or higher
- GPU recommended (NVIDIA CUDA 11.0+)
- 4GB RAM minimum

### Dependencies

```bash
requirements:
  - tensorflow >= 2.11.0
  - keras >= 2.11.0
  - numpy >= 1.21.0
  - pandas >= 1.3.0
  - scikit-learn >= 1.0.0
  - matplotlib >= 3.5.0
  - seaborn >= 0.12.0
  - pillow >= 9.0.0
  - kagglehub >= 1.0.0
```

### Setup Instructions

**1. Clone the Repository**

```bash
git clone https://github.com/precious-05/apple_disease_detection.git
cd apple_disease_detection
```

**2. Create Virtual Environment**

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

**3. Install Dependencies**

```bash
pip install -r requirements.txt
```

**4. Download Dataset (Optional)**

The notebook includes automatic dataset download via Kaggle Hub. Alternatively:

```bash
kagglehub dataset download "manzurulislam101413/apple-disease-classification-by-ab9d"
```

---

## Usage

### Running the Notebook

```bash
jupyter notebook Apple_Disease.ipynb
```

### Training the Model

The notebook includes complete training pipeline:

1. **Data Loading & Preprocessing**
   - Download dataset from Kaggle
   - Split into train/validation/test
   - Apply augmentation

2. **Model Training**
   - Initialize MobileNetV2 with ImageNet weights
   - Add custom classification head
   - Train with early stopping
   - Save best model checkpoint

3. **Evaluation**
   - Generate confusion matrix
   - Calculate per-class metrics
   - Plot performance curves
   - Visualize predictions

### Making Predictions

```python
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

# Load trained model
model = load_model('apple_disease_model.h5')

# Load and preprocess image
img = image.load_img('sample_leaf.jpg', target_size=(224, 224))
img_array = image.img_to_array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)

# Make prediction
prediction = model.predict(img_array)
class_names = ['Normal Apple', 'Scab Apple', 'Blotch Apple', 'Rot Apple']
predicted_class = class_names[np.argmax(prediction[0])]
confidence = np.max(prediction[0]) * 100

print(f"Predicted: {predicted_class}")
print(f"Confidence: {confidence:.2f}%")
```

---

## Results & Visualizations

### Key Results Summary

✓ **Overall Accuracy:** 90.3%  
✓ **Weighted F1-Score:** 0.90  
✓ **Training Time:** ~15 minutes (GPU)  
✓ **Inference Time:** ~50ms per image  
✓ **Model Size:** 9.2 MB  

### Dataset Distribution

The dataset is well-balanced across all disease categories with sufficient samples for robust training:
- Training: 1,602 images
- Testing: 400 images
- Validation: Included in notebook cross-validation

### Model Efficiency

- **Parameters:** 3.5M (lightweight)
- **Memory:** ~35 MB (inference)
- **FLOPs:** ~300M per image
- **Deployment:** Mobile, Edge, Cloud-ready

---

## Technologies Used

### Deep Learning Framework

**TensorFlow/Keras** - Industry-standard deep learning library
- Version: 2.11+
- Model architecture: Functional API
- Training: High-level Keras API
- Export: SavedModel & H5 formats

### Computer Vision

**OpenCV & PIL** - Image processing
- Preprocessing and augmentation
- Format conversion
- Batch processing

**Matplotlib & Seaborn** - Visualization
- Training curves
- Confusion matrices
- Classification reports
- Statistical plots

### Data & Utilities

**NumPy & Pandas** - Numerical computing
- Array operations
- Data manipulation
- Statistical analysis

**Scikit-learn** - Machine learning utilities
- Metrics calculation
- Train-test splitting
- Preprocessing pipelines

**Kaggle Hub** - Dataset management
- Automatic download
- Version control
- Dataset integration

---

## Key Features

### 1. Transfer Learning
- Pre-trained MobileNetV2 on ImageNet
- Fine-tuning on apple disease dataset
- Reduces training time significantly
- Improves generalization

### 2. Data Augmentation
- Random rotation and flipping
- Zoom and brightness adjustments
- Prevents overfitting
- Increases effective dataset size

### 3. Regularization
- L2 weight regularization (0.0001)
- Dropout layers (0.5 and 0.3)
- Early stopping on validation loss
- Batch normalization in base model

### 4. Model Optimization
- MobileNetV2 architecture (lightweight)
- Efficient depthwise separable convolutions
- Optimized for inference speed
- Mobile and edge device compatible

### 5. Comprehensive Evaluation
- Precision, Recall, F1-Score per class
- Confusion matrix analysis
- Training curve visualization
- Per-class accuracy breakdown

### 6. Production-Ready
- Model serialization (SavedModel format)
- Reproducible results with random seeds
- Comprehensive documentation
- Easy deployment pipeline

---

## Future Enhancements

### Model Improvements

1. **Ensemble Methods**
   - Combine multiple model architectures (EfficientNet, ResNet50)
   - Weighted voting for final predictions
   - Expected improvement: +2-3% accuracy

2. **Advanced Architectures**
   - Vision Transformer (ViT) integration
   - Hybrid CNN-Attention models
   - Multi-scale feature extraction

3. **Semi-Supervised Learning**
   - Leverage unlabeled data
   - Pseudo-labeling strategies
   - Self-training approaches

### Dataset Expansion

1. **Additional Disease Classes**
   - Include more apple disease types
   - Regional disease variants
   - Environmental stress factors

2. **Multi-View Analysis**
   - Process multiple angles per leaf
   - 3D reconstruction capabilities
   - Disease stage classification

### Deployment & Scalability

1. **Mobile Deployment**
   - TensorFlow Lite conversion
   - Quantization for edge devices
   - Real-time on-device inference

2. **API Development**
   - REST API for predictions
   - Batch processing endpoints
   - Real-time monitoring dashboard

3. **Cloud Integration**
   - AWS SageMaker deployment
   - Google Cloud AI Platform
   - Model serving infrastructure

### Explainability

1. **Interpretability**
   - Grad-CAM visualization
   - Feature importance analysis
   - Decision boundary exploration

2. **Uncertainty Quantification**
   - Bayesian neural networks
   - Monte Carlo dropout
   - Confidence calibration

---

## Usage in Production

### Batch Inference

```python
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

model = load_model('apple_disease_model.h5')
image_folder = 'path/to/images/'
results = []

for img_file in os.listdir(image_folder):
    if img_file.endswith(('.jpg', '.jpeg', '.png')):
        img = image.load_img(os.path.join(image_folder, img_file), 
                             target_size=(224, 224))
        img_array = image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        prediction = model.predict(img_array, verbose=0)
        results.append({
            'image': img_file,
            'class': class_names[np.argmax(prediction[0])],
            'confidence': np.max(prediction[0])
        })

print("Batch inference completed!")
```

---

## Contributing

Contributions are welcome! Please feel free to submit pull requests for:
- Bug fixes
- Performance improvements
- Additional features
- Documentation enhancements

---

## License

This project is licensed under the MIT License - see LICENSE file for details.

---

## Citation

If you use this project in your research, please cite:

```bibtex
@dataset{apple_disease_classification,
  title={Apple Disease Classification Dataset},
  author={Islam, Manzur},
  year={2024},
  url={https://www.kaggle.com/datasets/manzurulislam101413/apple-disease-classification-by-ab9d}
}

@inproceedings{sandler2018mobilenetv2,
  title={Mobilenetv2: Inverted residuals and linear bottlenecks},
  author={Sandler, Mark and Howard, Andrew and Zhu, Menglong and Zhmoginov, Andrey and Chen, Liang-Chieh},
  booktitle={CVPR},
  year={2018}
}
```

---

## Contact & Support

**Author:** precious-05  
**Repository:** https://github.com/precious-05/apple_disease_detection  
**Issues:** For bug reports and feature requests, please open an issue on GitHub  

---

## Acknowledgments

- Kaggle for the comprehensive apple disease dataset
- TensorFlow/Keras team for excellent deep learning framework
- MobileNetV2 authors for efficient architecture
- Community contributors and researchers

---

**Last Updated:** July 2026  
**Status:** Production Ready
