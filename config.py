
import os

# Project paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
MODELS_DIR = os.path.join(BASE_DIR, "models")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
NOTEBOOKS_DIR = os.path.join(BASE_DIR, "notebooks")

# Dataset configuration
CLASS_NAMES = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]
NUM_CLASSES = len(CLASS_NAMES)
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32

# Data split ratios
TRAIN_SPLIT = 0.70
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15

# Data augmentation parameters
AUGMENTATION_CONFIG = {
    "rotation_range": 20,
    "width_shift_range": 0.2,
    "height_shift_range": 0.2,
    "shear_range": 0.2,
    "zoom_range": 0.2,
    "horizontal_flip": True,
    "brightness_range": [0.8, 1.2],
    "fill_mode": "nearest"
}

# Model configuration
MODEL_CONFIG = {
    "input_shape": (224, 224, 3),
    "base_model_trainable": False,  # Initially freeze base model
    "dropout_rate": 0.3,
    "learning_rate": 1e-3,
    "fine_tune_learning_rate": 1e-5,
    "fine_tune_layers": 20  # Number of layers to unfreeze for fine-tuning
}

# Training configuration
TRAIN_CONFIG = {
    "epochs_initial": 15,
    "epochs_fine_tune": 10,
    "early_stopping_patience": 5,
    "reduce_lr_patience": 3,
    "reduce_lr_factor": 0.2,
    "model_checkpoint_path": os.path.join(MODELS_DIR, "best_model.keras")
}

# Evaluation configuration
EVAL_CONFIG = {
    "confusion_matrix_path": os.path.join(RESULTS_DIR, "confusion_matrix.png"),
    "accuracy_graph_path": os.path.join(RESULTS_DIR, "accuracy_graph.png"),
    "loss_graph_path": os.path.join(RESULTS_DIR, "loss_graph.png"),
    "classification_report_path": os.path.join(RESULTS_DIR, "classification_report.txt"),
    "evaluation_metrics_path": os.path.join(RESULTS_DIR, "evaluation_metrics.json")
}

# Streamlit configuration
STREAMLIT_CONFIG = {
    "page_title": "Smart Waste Classification",
    "page_icon": "♻️",
    "layout": "wide",
    "initial_sidebar_state": "expanded"
}

# Class descriptions for UI
CLASS_DESCRIPTIONS = {
    "cardboard": "Cardboard boxes, cartons, and packaging materials",
    "glass": "Glass bottles, jars, and containers",
    "metal": "Metal cans, foil, and containers",
    "paper": "Paper, newspapers, magazines, and cardboard",
    "plastic": "Plastic bottles, containers, and packaging",
    "trash": "General waste that cannot be recycled"
}

# Supported image formats
SUPPORTED_FORMATS = ["jpg", "jpeg", "png"]

# Confidence threshold for low confidence warning
CONFIDENCE_THRESHOLD = 0.5