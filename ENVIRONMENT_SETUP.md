# 🛠️ Environment Setup - Machine Learning Deep Learning Cookbook

This document explains how to set up your environment to run all notebooks in the MLDL Cookbook.

## 🎯 Requirements

- **Python**: 3.10 (compatible with TensorFlow 2.10)
- **TensorFlow**: 2.10.0
- **NumPy**: 1.23.5 (compatible with TensorFlow 2.10)
- **Scikit-learn**: 1.1.3

## 📋 Setup Options

### Option 1: Conda Environment (Recommended)

Using the provided `environment.yml` file:

```bash
# Create environment
conda env create -f environment.yml

# Activate environment
conda activate mldl-cookbook

# Verify installation
python -c "import tensorflow as tf; print('TensorFlow version:', tf.__version__)"
python -c "import numpy as np; print('NumPy version:', np.__version__)"
```

### Option 2: Pip Installation

Using the provided `requirements.txt` file:

```bash
# Create virtual environment
python3.10 -m venv mldl-cookbook-env

# Activate environment
# On Windows:
mldl-cookbook-env\Scripts\activate
# On macOS/Linux:
source mldl-cookbook-env/bin/activate

# Install requirements
pip install -r requirements.txt

# Verify installation
python -c "import tensorflow as tf; print('TensorFlow version:', tf.__version__)"
```

### Option 3: Manual Installation

```bash
# Core packages
pip install tensorflow==2.10.0 numpy==1.23.5 scikit-learn==1.1.3

# Data and visualization
pip install pandas==1.5.3 matplotlib==3.6.3 seaborn==0.12.2

# Jupyter
pip install jupyter jupyterlab

# Additional packages as needed
```

## 🧪 Testing Your Installation

Run this test script to verify everything works:

```python
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

print("🔧 ENVIRONMENT TEST")
print("=" * 50)
print(f"✅ TensorFlow version: {tf.__version__}")
print(f"✅ NumPy version: {np.__version__}")
print(f"✅ Pandas version: {pd.__version__}")

# Test GPU (if available)
gpus = tf.config.list_physical_devices('GPU')
print(f"🎮 GPU devices: {len(gpus)}")
if gpus:
    for i, gpu in enumerate(gpus):
        print(f"  GPU {i}: {gpu.name}")
else:
    print("  ⚠️  No GPU detected, using CPU")

print("\n✅ Environment setup successful!")
```

## 📦 Package Breakdown

### Core ML/DL
- **tensorflow==2.10.0**: Main deep learning framework
- **numpy==1.23.5**: Numerical computing (TF 2.10 compatible)
- **scikit-learn==1.1.3**: Traditional ML algorithms

### Data Science
- **pandas==1.5.3**: Data manipulation and analysis
- **matplotlib==3.6.3**: Basic plotting
- **seaborn==0.12.2**: Statistical visualization
- **plotly==5.17.0**: Interactive plotting

### Jupyter Environment
- **jupyter==1.0.0**: Jupyter notebooks
- **jupyterlab==3.6.1**: Enhanced Jupyter interface
- **ipykernel==6.19.4**: Python kernel for Jupyter

### Specialized
- **Pillow==9.4.0**: Image processing
- **opencv-python==4.7.0.72**: Computer vision
- **pygame==2.1.3**: Game development (Snake AI project)
- **openai>=0.27.0**: LLM/AI projects
- **h5py==3.7.0**: Model saving/loading

## 🔧 Compatibility Notes

### TensorFlow 2.10 Compatibility
- **Python**: 3.7-3.10 (we use 3.10)
- **NumPy**: 1.19.2-1.24.3 (we use 1.23.5)
- **CUDA**: 11.2 (for GPU support)
- **cuDNN**: 8.1 (for GPU support)

### Version Constraints
- All versions are specifically chosen for compatibility with TensorFlow 2.10
- NumPy 1.23.5 provides optimal compatibility with both TensorFlow 2.10 and scikit-learn
- Pandas 1.5.3 ensures compatibility with NumPy 1.23.5

## 🚀 Getting Started

After setting up your environment:

1. **Activate your environment** (conda or venv)
2. **Start Jupyter**: `jupyter lab` or `jupyter notebook`
3. **Navigate to notebooks** in your browser
4. **Run the test cell** in any notebook to verify setup

## 🆘 Troubleshooting

### Common Issues

**TensorFlow not detecting GPU:**
```python
import tensorflow as tf
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
```

**NumPy compatibility warnings:**
- Ensure NumPy 1.23.5 is installed
- Restart Python kernel after installation

**Import errors:**
- Verify environment is activated
- Check package versions: `pip list` or `conda list`

**Module not found:**
- Ensure Jupyter kernel matches your environment
- Restart Jupyter after installing packages

### Getting Help

If you encounter issues:
1. Check package versions match requirements
2. Verify Python 3.10 is being used
3. Ensure environment is properly activated
4. Try creating a fresh environment

## 📋 Environment Management

### Updating Packages
```bash
# Conda
conda env update -f environment.yml

# Pip
pip install -r requirements.txt --upgrade
```

### Removing Environment
```bash
# Conda
conda env remove -n mldl-cookbook

# Pip
rm -rf mldl-cookbook-env  # or delete folder manually
```

---

🎓 **Happy Learning with the MLDL Cookbook!** 