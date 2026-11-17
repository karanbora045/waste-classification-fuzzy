"""
Evaluation Module for Waste Classification
Comprehensive model evaluation with metrics and visualizations
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
from sklearn.metrics import (
    classification_report, confusion_matrix,
    precision_recall_fscore_support, accuracy_score
)

import tensorflow as tf
from tensorflow.keras.models import load_model

from config import (
    CLASS_NAMES, NUM_CLASSES, IMAGE_SIZE, BATCH_SIZE,
    EVAL_CONFIG, RESULTS_DIR, MODELS_DIR
)
from src.data_preprocessing import create_data_generators, load_class_indices


def load_trained_model(model_path: str = None) -> tf.keras.Model:
    if model_path is None:
        model_path = os.path.join(MODELS_DIR, "best_model.keras")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}")

    print(f"Loading model from {model_path}...")
    model = load_model(model_path)
    return model


def get_predictions(
    model: tf.keras.Model,
    test_dir: str = "dataset/test",
    batch_size: int = BATCH_SIZE
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    # Create test generator
    _, _, test_gen = create_data_generators(
        train_dir="dataset/train",
        val_dir="dataset/val",
        test_dir=test_dir,
        batch_size=batch_size
    )

    # Get predictions
    print("Generating predictions...")
    y_pred_proba = model.predict(test_gen, verbose=1)
    y_pred = np.argmax(y_pred_proba, axis=1)
    y_true = test_gen.classes

    return y_true, y_pred, y_pred_proba


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_pred_proba: np.ndarray = None
) -> Dict:
    # Basic metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=0
    )

    # Macro and weighted averages
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='macro', zero_division=0
    )
    weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='weighted', zero_division=0
    )

    metrics = {
        "accuracy": float(accuracy),
        "per_class": {
            CLASS_NAMES[i]: {
                "precision": float(precision[i]),
                "recall": float(recall[i]),
                "f1_score": float(f1[i]),
                "support": int(support[i])
            }
            for i in range(NUM_CLASSES)
        },
        "macro_avg": {
            "precision": float(macro_precision),
            "recall": float(macro_recall),
            "f1_score": float(macro_f1)
        },
        "weighted_avg": {
            "precision": float(weighted_precision),
            "recall": float(weighted_recall),
            "f1_score": float(weighted_f1)
        }
    }

    return metrics


def generate_classification_report_text(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> str:
    report = classification_report(
        y_true, y_pred,
        target_names=CLASS_NAMES,
        digits=4,
        zero_division=0
    )
    return report


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    save_path: str = EVAL_CONFIG["confusion_matrix_path"],
    normalize: bool = True
) -> None:
    cm = confusion_matrix(y_true, y_pred)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2f'
        title = "Normalized Confusion Matrix"
    else:
        fmt = 'd'
        title = "Confusion Matrix"

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt=fmt,
        cmap='Blues',
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
        cbar_kws={'label': 'Proportion' if normalize else 'Count'}
    )
    plt.title(title, fontsize=14, fontweight='bold', pad=20)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Confusion matrix saved to {save_path}")


def plot_per_class_metrics(
    metrics: Dict,
    save_path: str = os.path.join(RESULTS_DIR, "per_class_metrics.png")
) -> None:
    classes = CLASS_NAMES
    precision = [metrics["per_class"][c]["precision"] for c in classes]
    recall = [metrics["per_class"][c]["recall"] for c in classes]
    f1 = [metrics["per_class"][c]["f1_score"] for c in classes]

    x = np.arange(len(classes))
    width = 0.25

    fig, ax = plt.subplots(figsize=(12, 6))
    bars1 = ax.bar(x - width, precision, width, label='Precision', color='#3498db', alpha=0.8)
    bars2 = ax.bar(x, recall, width, label='Recall', color='#2ecc71', alpha=0.8)
    bars3 = ax.bar(x + width, f1, width, label='F1-Score', color='#e74c3c', alpha=0.8)

    ax.set_xlabel('Waste Category', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Per-Class Performance Metrics', fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(classes, rotation=45, ha='right')
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')

    # Add value labels on bars
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.2f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Per-class metrics plot saved to {save_path}")


def plot_prediction_distribution(
    y_pred_proba: np.ndarray,
    y_true: np.ndarray,
    save_path: str = os.path.join(RESULTS_DIR, "prediction_distribution.png")
) -> None:
    # Get max probability for each prediction
    max_probs = np.max(y_pred_proba, axis=1)
    correct = np.argmax(y_pred_proba, axis=1) == y_true

    plt.figure(figsize=(10, 6))
    plt.hist(max_probs[correct], bins=30, alpha=0.7, label='Correct', color='#2ecc71', density=True)
    plt.hist(max_probs[~correct], bins=30, alpha=0.7, label='Incorrect', color='#e74c3c', density=True)
    plt.xlabel('Prediction Confidence', fontsize=12)
    plt.ylabel('Density', fontsize=12)
    plt.title('Prediction Confidence Distribution', fontsize=14, fontweight='bold', pad=20)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Prediction distribution plot saved to {save_path}")


def save_evaluation_results(
    metrics: Dict,
    classification_report_text: str,
    save_dir: str = RESULTS_DIR
) -> None:
    os.makedirs(save_dir, exist_ok=True)

    # Save metrics as JSON
    with open(os.path.join(save_dir, "evaluation_metrics.json"), 'w') as f:
        json.dump(metrics, f, indent=4)

    # Save classification report as text
    with open(EVAL_CONFIG["classification_report_path"], 'w') as f:
        f.write(classification_report_text)

    print(f"Evaluation results saved to {save_dir}")


def load_evaluation_results(load_dir: str = RESULTS_DIR) -> Dict:
    metrics_path = os.path.join(load_dir, "evaluation_metrics.json")
    if os.path.exists(metrics_path):
        with open(metrics_path, 'r') as f:
            return json.load(f)
    return {}


def run_full_evaluation(
    model_path: str = None,
    test_dir: str = "dataset/test"
) -> Dict:
    print("="*50)
    print("Running Full Model Evaluation")
    print("="*50)

    # Load model
    model = load_trained_model(model_path)

    # Check if test directory exists
    if not os.path.exists(test_dir):
        print(f"Test directory not found: {test_dir}")
        print("Please run dataset splitting first.")
        return {}

    # Get predictions
    y_true, y_pred, y_pred_proba = get_predictions(model, test_dir)

    # Compute metrics
    metrics = compute_metrics(y_true, y_pred, y_pred_proba)

    # Generate classification report
    report_text = generate_classification_report_text(y_true, y_pred)

    # Print results
    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)
    print(f"\nTest Accuracy: {metrics['accuracy']:.4f}")
    print(f"Macro F1: {metrics['macro_avg']['f1_score']:.4f}")
    print(f"Weighted F1: {metrics['weighted_avg']['f1_score']:.4f}")

    print("\nPer-Class Metrics:")
    for class_name in CLASS_NAMES:
        m = metrics["per_class"][class_name]
        print(f"  {class_name:10s}: P={m['precision']:.4f} R={m['recall']:.4f} F1={m['f1_score']:.4f} (n={m['support']})")

    print("\nClassification Report:")
    print(report_text)

    # Generate visualizations
    print("\nGenerating visualizations...")
    plot_confusion_matrix(y_true, y_pred)
    plot_per_class_metrics(metrics)
    plot_prediction_distribution(y_pred_proba, y_true)

    # Save results
    save_evaluation_results(metrics, report_text)

    print("\n" + "="*50)
    print("Evaluation Complete!")
    print("="*50)

    return metrics


if __name__ == "__main__":
    run_full_evaluation()