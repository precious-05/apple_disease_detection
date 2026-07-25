# Setup Guide - Apple Disease Detection

Complete step-by-step guide to set up and run the Apple Disease Detection project.

---

## Table of Contents

1. [System Requirements](#system-requirements)
2. [Installation Steps](#installation-steps)
3. [Dataset Setup](#dataset-setup)
4. [Generating Visualizations](#generating-visualizations)
5. [Running the Notebook](#running-the-notebook)
6. [Troubleshooting](#troubleshooting)

---

## System Requirements

### Hardware Requirements

**Minimum:**
- CPU: Intel i5 or equivalent
- RAM: 4 GB
- Storage: 2 GB (for dataset + model)

**Recommended:**
- CPU: Intel i7 or AMD Ryzen 5+
- GPU: NVIDIA GPU with CUDA support (highly recommended)
- RAM: 8 GB+
- Storage: 10 GB SSD

### Software Requirements

- Python 3.8 or higher
- pip (Python package manager)
- Git (for cloning repository)
- CUDA 11.0+ (optional, for GPU acceleration)
- cuDNN 8.0+ (optional, for GPU acceleration)

### Operating Systems

- Windows 10/11
- macOS 10.14+
- Ubuntu 18.04+
- Any system with Python 3.8+ support

---

## Installation Steps

### Step 1: Clone the Repository

```bash
git clone https://github.com/precious-05/apple_disease_detection.git
cd apple_disease_detection
```

### Step 2: Create Virtual Environment

**On Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**On macOS/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

**Verify activation:**
```bash
# You should see (venv) in your terminal prompt
which python  # Should show path to venv python
```

### Step 3: Upgrade pip

```bash
python -m pip install --upgrade pip
```

### Step 4: Install Dependencies

```bash
pip install -r requirements.txt
```

**Expected output:**
```
Successfully installed tensorflow-2.11.0 keras-2.11.0 numpy-1.21.0 ...
```

### Step 5: Verify Installation

```bash
python -c "import tensorflow as tf; print(tf.__version__)"
python -c "import keras; print(keras.__version__)"
```

**Expected output:**
```
2.11.0
2.11.0
```

### Step 6: (Optional) GPU Setup

If you have an NVIDIA GPU and want to use it:

**Install CUDA Toolkit (v11.8):**
- Download from: https://developer.nvidia.com/cuda-11-8-0-download-archive
- Follow installation instructions for your OS

**Install cuDNN (v8.6):**
- Download from: https://developer.nvidia.com/cudnn
- Extract and add to system PATH

**Verify GPU detection:**
```bash
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

---

## Dataset Setup

### Automatic Download (Recommended)

The notebook will automatically download the dataset using Kaggle Hub:

```bash
# No manual setup needed - runs automatically in notebook
# The notebook will download to: ~/.cache/kagglehub/
```

### Manual Download

**Option 1: Using Kaggle Hub CLI**

```bash
# Install kagglehub (already in requirements.txt)
pip install kagglehub

# Download dataset
python -c "import kagglehub; kagglehub.dataset_download('manzurulislam101413/apple-disease-classification-by-ab9d')"
```

**Option 2: From Kaggle Website**

1. Visit: https://www.kaggle.com/datasets/manzurulislam101413/apple-disease-classification-by-ab9d
2. Click "Download" button
3. Extract to `./data/` directory
4. Update path in notebook if needed

### Dataset Structure

After download/extraction:

```
data/
├── apple_disease_classification/
│   ├── Train/
│   │   ├── Normal_Apple/      (400 images)
│   │   ├── Scab_Apple/        (402 images)
│   │   ├── Blotch_Apple/      (400 images)
│   │   └── Rot_Apple/         (400 images)
│   └── Test/
│       ├── Normal_Apple/      (100 images)
│       ├── Scab_Apple/        (100 images)
│       ├── Blotch_Apple/      (100 images)
│       └── Rot_Apple/         (100 images)
```

---

## Generating Visualizations

### Generate README Assets

The project includes a script to generate all visualization images:

```bash
python generate_readme_assets.py
```

**Expected output:**
```
✓ Saved: dataset_distribution.png
✓ Saved: model_architecture.png
✓ Saved: training_performance.png
✓ Saved: confusion_matrix.png
✓ Saved: classification_metrics.png
✓ Saved: per_class_accuracy.png

✓ All visualization assets generated successfully!
✓ Images saved to: /path/to/apple_disease_detection/images
```

### Generated Files

Images are saved to `./images/` directory:

- `dataset_distribution.png` - Dataset composition chart
- `model_architecture.png` - Model architecture diagram
- `training_performance.png` - Training curves
- `confusion_matrix.png` - Confusion matrix heatmap
- `classification_metrics.png` - Per-class metrics
- `per_class_accuracy.png` - Accuracy comparison

---

## Running the Notebook

### Start Jupyter Lab/Notebook

```bash
# Option 1: Jupyter Lab (recommended)
jupyter lab

# Option 2: Jupyter Notebook
jupyter notebook
```

This will open a browser window at `http://localhost:8888`

### Open the Project Notebook

1. Navigate to `Apple_Disease.ipynb` in the Jupyter interface
2. Click to open the notebook
3. Select kernel: Python 3 (from venv)

### Execute Notebook Cells

**Method 1: Sequential Execution**
- Click on first cell
- Press `Shift + Enter` to execute each cell sequentially

**Method 2: Run All Cells**
- Menu: `Cell` → `Run All Cells`

**Method 3: Step Through**
- Use the `Run` button in the toolbar for individual cells

### Expected Execution Flow

1. **Setup (Cells 1-3)**
   - Install kagglehub
   - Import libraries
   - Display environment info

2. **Data Loading (Cells 4-6)**
   - Download dataset from Kaggle
   - Verify dataset structure
   - Display sample counts

3. **Preprocessing (Cells 7-10)**
   - Load and augment images
   - Create train/validation splits
   - Display sample images

4. **Model Building (Cells 11-13)**
   - Load MobileNetV2 pre-trained model
   - Add custom classification head
   - Compile model

5. **Training (Cells 14-16)**
   - Train model (10-20 minutes on GPU)
   - Monitor loss and accuracy
   - Apply early stopping

6. **Evaluation (Cells 17-20)**
   - Calculate metrics
   - Generate confusion matrix
   - Plot performance curves

7. **Visualization (Cells 21-25)**
   - Display predictions
   - Show class distribution
   - Export results

### Expected Execution Times

| Step | CPU | GPU |
|------|-----|-----|
| Data loading | 2-3 min | 2-3 min |
| Model compilation | 10 sec | 5 sec |
| Training (50 epochs) | 45-60 min | 10-15 min |
| Evaluation | 3-5 min | 1-2 min |
| **Total** | **~60 min** | **~20 min** |

---

## Troubleshooting

### Common Issues & Solutions

#### 1. Module Import Errors

**Error:** `ModuleNotFoundError: No module named 'tensorflow'`

**Solution:**
```bash
# Ensure virtual environment is activated
source venv/bin/activate  # macOS/Linux
# or
venv\Scripts\activate  # Windows

# Reinstall dependencies
pip install --upgrade -r requirements.txt
```

#### 2. Kernel Not Available

**Error:** `No module named 'ipykernel'`

**Solution:**
```bash
python -m ipykernel install --user --name venv --display-name "Python (venv)"
```

Then restart Jupyter and select the new kernel.

#### 3. CUDA/GPU Not Detected

**Error:** `Could not load dynamic library 'libcudart.so'`

**Solution:**
```bash
# Verify CUDA installation
nvcc --version

# If not found, reinstall CUDA toolkit
# See GPU Setup section above

# Use CPU as fallback (works but slower)
# No action needed - TensorFlow will use CPU automatically
```

#### 4. Memory Issues

**Error:** `ResourceExhaustedError: OOM when allocating tensor`

**Solution:**
```python
# In notebook, reduce batch size before training:
BATCH_SIZE = 16  # Instead of 32
# or
import tensorflow as tf
tf.config.run_functions_eagerly(True)
```

#### 5. Dataset Download Fails

**Error:** `Connection error downloading dataset`

**Solution:**
```bash
# Option 1: Retry with timeout
python generate_readme_assets.py

# Option 2: Manual download
# Visit: https://www.kaggle.com/datasets/...
# Download and extract manually to ./data/

# Option 3: Clear cache
rm -rf ~/.cache/kagglehub/
# Then retry
```

#### 6. Jupyter Port Already in Use

**Error:** `Address already in use`

**Solution:**
```bash
# Use different port
jupyter notebook --port 8889

# or kill existing process
# Windows: taskkill /PID <process_id> /F
# Linux/Mac: kill -9 <process_id>
```

#### 7. Low Disk Space

**Error:** `No space left on device`

**Solution:**
```bash
# Check available space
df -h  # Linux/Mac
dir C:\  # Windows

# Clean up:
# - Remove ~/.cache/kagglehub/ (safe to delete)
# - Remove unused virtual environments
# - Clear pip cache: pip cache purge
```

### Performance Optimization

**For faster training:**

```python
# Use mixed precision training
from tensorflow.keras import mixed_precision
policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_global_policy(policy)

# Reduce input size
IMG_SIZE = 192  # Instead of 224

# Increase batch size (if memory allows)
BATCH_SIZE = 64  # Instead of 32

# Enable multi-threading
tf.config.threading.set_inter_op_parallelism_threads(4)
```

**For lower memory usage:**

```python
# Reduce batch size
BATCH_SIZE = 8

# Use float16 instead of float32
model = load_model(..., dtype='float16')

# Generator-based data loading
# (Already implemented in notebook)
```

---

## Verification Checklist

After setup, verify everything works:

- [ ] Virtual environment activated
- [ ] All dependencies installed (`pip list`)
- [ ] Python version 3.8+ (`python --version`)
- [ ] TensorFlow importable (`python -c "import tensorflow"`)
- [ ] Jupyter accessible (`jupyter --version`)
- [ ] Dataset available (manual or auto)
- [ ] Visualizations generated (`ls images/`)
- [ ] Notebook opens without errors

---

## Next Steps

1. **Run the notebook:** `jupyter lab Apple_Disease.ipynb`
2. **Review results:** Check generated visualizations in `./images/`
3. **Modify parameters:** Experiment with different architectures/hyperparameters
4. **Deploy model:** See README.md for production deployment

---

## Support

**If you encounter issues:**

1. Check [Troubleshooting](#troubleshooting) section
2. Review error messages carefully
3. Search existing GitHub issues
4. Open new issue with:
   - Error message (full traceback)
   - Your system info (`python --version`, OS, etc.)
   - Steps to reproduce
   - Relevant code snippet

---

## Additional Resources

- TensorFlow Docs: https://www.tensorflow.org/guide
- Keras API: https://keras.io/
- Kaggle Datasets: https://www.kaggle.com/datasets/
- NVIDIA CUDA Setup: https://docs.nvidia.com/cuda/

---

**Last Updated:** July 2026  
**Status:** Production Ready
