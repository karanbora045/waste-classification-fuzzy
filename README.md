# ♻️ Smart Waste Classification Using Convolutional Neural Networks

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-1.25%2B-red)


## 🎯 Project Introduction

Waste management is a critical environmental challenge. Proper waste segregation at source enables efficient recycling and reduces landfill burden. This project develops an **AI-powered waste classification system** that uses computer vision and deep learning to automatically identify waste categories from images.

The system classifies waste into **six categories**:
- 📦 **Cardboard** - Boxes, cartons, packaging
- 🍾 **Glass** - Bottles, jars, containers
- 🥫 **Metal** - Cans, foil, containers
- 📄 **Paper** - Newspapers, magazines, documents
- 🥤 **Plastic** - Bottles, containers, packaging
- 🗑️ **Trash** - General non-recyclable waste

---

## ❓ Problem Statement

Manual waste sorting is:
- **Labor-intensive** and costly
- **Error-prone** due to human fatigue
- **Inconsistent** across different workers
- **Slow** for high-volume waste streams

Automated classification using CNNs offers:
- ✅ High accuracy with transfer learning
- ✅ Real-time inference capability
- ✅ Consistent performance 24/7
- ✅ Scalable to multiple sorting stations

---

## 🎯 Project Objectives

1. **Build a CNN classifier** using MobileNetV2 transfer learning
2. **Achieve >85% test accuracy** on six waste categories
3. **Create a professional web interface** using Streamlit
4. **Provide comprehensive evaluation** with metrics and visualizations
5. **Ensure reproducible, modular code** suitable for academic submission

---

## 🛠️ Technologies Used

| Category | Technologies |
|----------|-------------|
| **Language** | Python 3.8+ |
| **Deep Learning** | TensorFlow 2.x, Keras |
| **Model Architecture** | MobileNetV2 (Transfer Learning) |
| **Image Processing** | OpenCV, Pillow (PIL) |
| **Data Processing** | NumPy, Pandas, Scikit-learn |
| **Visualization** | Matplotlib, Seaborn, Plotly |
| **Web Framework** | Streamlit |
| **Development** | Jupyter Notebook |

---

## 🧠 CNN & Transfer Learning Explanation

### What is Transfer Learning?

Transfer learning leverages a **pre-trained model** (trained on a large dataset like ImageNet) and adapts it to a new, related task with a smaller dataset.

### Why MobileNetV2?

- **Lightweight**: ~3.5M parameters (vs 138M for ResNet-152)
- **Fast**: Optimized for mobile/edge deployment
- **Effective**: Depthwise separable convolutions
- **Pre-trained**: ImageNet weights (1.4M images, 1000 classes)

### Architecture

```
Input Image (224×224×3)
         │
         ▼
┌─────────────────────────────┐
│   MobileNetV2 Base Model    │  ← Frozen initially (pre-trained)
│   (Feature Extractor)       │
│   Output: 7×7×1280          │
└─────────────────────────────┘
         │
         ▼
┌─────────────────────────────┐
│  Global Average Pooling 2D  │  → Reduces to 1280 features
│  (No parameters)            │
└─────────────────────────────┘
         │
         ▼
┌─────────────────────────────┐
│       Dropout (0.3)         │  ← Regularization
└─────────────────────────────┘
         │
         ▼
┌─────────────────────────────┐
│    Dense (6, Softmax)       │  ← Classification Head
│    (Waste Categories)       │
└─────────────────────────────┘
         │
         ▼
   Probabilities [6]
```

### Two-Phase Training

| Phase | Layers Trained | Learning Rate | Epochs |
|-------|----------------|---------------|--------|
| **1. Feature Extraction** | Classification head only | 1e-3 | 15 |
| **2. Fine-tuning** | Last 20 base layers + head | 1e-5 | 10 |

---

## 📊 Dataset Information

### TrashNet Dataset

The **TrashNet** dataset (by Gary Thung & Mindy Yang) contains ~2,500 images across 6 categories.

| Category | Images | Percentage |
|----------|--------|------------|
| Cardboard | 403 | 16% |
| Glass | 501 | 20% |
| Metal | 410 | 16% |
| Paper | 594 | 24% |
| Plastic | 482 | 19% |
| Trash | 137 | 5% |
| **Total** | **~2,527** | **100%** |

### Download Instructions

1. **Original Source**: [TrashNet on GitHub](https://github.com/garythung/trashnet)
2. **Direct Download**: [Google Drive](https://drive.google.com/drive/folders/0B3P9oOwA3RvUY0NsZXVfc01xSlU) (original)
3. **Kaggle Mirror**: [TrashNet on Kaggle](https://www.kaggle.com/datasets/mostafaabla/garbage-classification)

### Dataset Organization

After downloading, organize the dataset as follows:

```
dataset/
├── cardboard/
│   ├── cardboard1.jpg
│   ├── cardboard2.jpg
│   └── ...
├── glass/
│   ├── glass1.jpg
│   └── ...
├── metal/
├── paper/
├── plastic/
└── trash/
```

**Windows PowerShell commands to organize:**

```powershell
# Create directory structure
mkdir dataset\cardboard, dataset\glass, dataset\metal, dataset\paper, dataset\plastic, dataset\trash

# Move downloaded images to respective folders
# Example (adjust paths as needed):
move trashnet\cardboard\* dataset\cardboard\
move trashnet\glass\* dataset\glass\
move trashnet\metal\* dataset\metal\
move trashnet\paper\* dataset\paper\
move trashnet\plastic\* dataset\plastic\
move trashnet\trash\* dataset\trash\
```

---

## 📁 Project Structure

```
waste-classification/
│
├── app.py                      # Streamlit web application
├── requirements.txt            # Python dependencies
├── config.py                   # Configuration & hyperparameters
├── README.md                   # This file
│
├── dataset/                    # Organized waste images (gitignored)
│   ├── cardboard/
│   ├── glass/
│   ├── metal/
│   ├── paper/
│   ├── plastic/
│   └── trash/
│
├── models/                     # Trained models (gitignored)
│   ├── best_model.keras        # Best model checkpoint
│   ├── class_indices.json      # Class mapping
│   └── model_architecture.json # Model architecture
││
├── src/                        # Source code modules
│   ├── data_preprocessing.py   # Data loading, splitting, augmentation
│   ├── model.py                # Model architecture & compilation
│   ├── train.py                # Training pipeline
│   ├── evaluate.py             # Evaluation & metrics
│   └── predict.py              # Inference interface
│
├── results/                    # Evaluation outputs
│   ├── confusion_matrix.png
│   ├── accuracy_graph.png
│   ├── loss_graph.png
│   ├── precision_graph.png
│   ├── recall_graph.png
│   ├── per_class_metrics.png
│   ├── prediction_distribution.png
│   ├── classification_report.txt
│   └── evaluation_metrics.json
│
└── utils/
    └── image_utils.py          # Image processing utilities
```

---

## ⚙️ Installation

### Prerequisites

- **Windows 10/11** (tested)
- **Python 3.8+** (recommended: 3.10 or 3.11)
- **Git** (for cloning)
- **GPU** (optional but recommended for training)

### Step 1: Clone Repository

```powershell
git clone <repository-url>
cd waste-classification
```

### Step 2: Create Virtual Environment

```powershell
# Create virtual environment
python -m venv venv

# Activate (Windows PowerShell)
venv\Scripts\Activate.ps1

# Or Command Prompt
venv\Scripts\activate.bat
```

### Step 3: Install Dependencies

```powershell
# Upgrade pip
python -m pip install --upgrade pip

# Install requirements
pip install -r requirements.txt
```

### Step 4: Verify Installation

```powershell
# Test imports
python -c "import tensorflow as tf; print(f'TF: {tf.__version__}'); import streamlit; print('Streamlit OK')"
```

### Step 5: Download Dataset

Follow the [Dataset Instructions](#-dataset-information) to download and organize the TrashNet dataset in the `dataset/` folder.

---

## 🏋️ Model Training

### Option 1: Using Python Script (Recommended)

```powershell
# Activate environment first
venv\Scripts\Activate.ps1

# Run training
python -m src.train
```

This will:
1. Check dataset existence
2. Split data (70/15/15) if not already split
3. Build MobileNetV2 model
4. **Phase 1**: Train classification head (15 epochs)
5. **Phase 2**: Fine-tune last 20 layers (10 epochs)
6. Save best model to `models/best_model.keras`
7. Generate training curves in `results/`

### Option 2: Using Jupyter Notebook (Interactive)

```powershell
# Start Jupyter
jupyter notebook notebooks/model_training.ipynb
```

Run cells sequentially for step-by-step training with visualizations.

### Training Configuration

Key hyperparameters in `config.py`:

```python
TRAIN_CONFIG = {
    "epochs_initial": 15,      # Phase 1 epochs
    "epochs_fine_tune": 10,    # Phase 2 epochs
    "early_stopping_patience": 5,
    "reduce_lr_patience": 3,
    "model_checkpoint_path": "models/best_model.keras"
}

MODEL_CONFIG = {
    "learning_rate": 1e-3,           # Phase 1 LR
    "fine_tune_learning_rate": 1e-5, # Phase 2 LR
    "fine_tune_layers": 20,          # Layers to unfreeze
    "dropout_rate": 0.3
}
```

## 📈 Model Evaluation

After training, run the evaluation script:

```powershell
python -m src.evaluate
```

This generates:

### Metrics Computed

- **Test Accuracy** - Overall classification accuracy
- **Precision** (per-class, macro, weighted)
- **Recall** (per-class, macro, weighted)
- **F1-Score** (per-class, macro, weighted)
- **Confusion Matrix** - Normalized and raw

### Output Files

| File | Description |
|------|-------------|
| `results/evaluation_metrics.json` | All metrics in JSON |
| `results/classification_report.txt` | Sklearn classification report |
| `results/confusion_matrix.png` | Normalized confusion matrix heatmap |
| `results/accuracy_graph.png` | Training/validation accuracy curves |
| `results/loss_graph.png` | Training/validation loss curves |
| `results/per_class_metrics.png` | Bar chart of per-class P/R/F1 |
| `results/prediction_distribution.png` | Confidence distribution |

### Sample Output

```
Test Accuracy: 0.9123
Macro F1: 0.9087
Weighted F1: 0.9123

Per-Class Metrics:
  cardboard  : P=0.9400 R=0.9200 F1=0.9300 (n=60)
  glass      : P=0.9600 R=0.9800 F1=0.9700 (n=75)
  metal      : P=0.9300 R=0.9100 F1=0.9200 (n=62)
  paper      : P=0.8800 R=0.8900 F1=0.8850 (n=89)
  plastic    : P=0.9100 R=0.9000 F1=0.9050 (n=72)
  trash      : P=0.8500 R=0.8200 F1=0.8350 (n=21)
```

---

## 🚀 Running the Application

Launch the Streamlit web application:

```powershell
# Activate environment
venv\Scripts\Activate.ps1

# Run Streamlit app
streamlit run app.py
```

The app will open at **http://localhost:8501**

### Using the Classifier

1. Go to **🔍 Classify Waste**
2. Upload an image (JPG/PNG) or use camera
3. View prediction with confidence percentage
4. See probability breakdown for all 6 categories
5. Low confidence warnings appear automatically

---
