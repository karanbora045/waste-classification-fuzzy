import os
import json
import time
from typing import Tuple, Dict, Optional

import tensorflow as tf
import matplotlib.pyplot as plt

from config import (
    TRAIN_CONFIG, MODEL_CONFIG, CLASS_NAMES, NUM_CLASSES,
    MODELS_DIR, RESULTS_DIR
)
from src.data_preprocessing import (
    check_dataset_exists, split_dataset, create_data_generators,
    save_class_indices, verify_no_data_leakage
)
from src.model import (
    build_model, compile_model, get_callbacks,
    unfreeze_base_model, recompile_for_fine_tuning,
    save_model_architecture, count_parameters
)


def train_model(
    train_dir: str = "dataset/train",
    val_dir: str = "dataset/val",
    epochs_initial: int = TRAIN_CONFIG["epochs_initial"],
    epochs_fine_tune: int = TRAIN_CONFIG["epochs_fine_tune"],
    batch_size: int = 32,
    fine_tune: bool = True
) -> Tuple[tf.keras.Model, Dict]:
    # Check dataset
    if not check_dataset_exists():
        raise FileNotFoundError(
            "Dataset not found! Please download and organize the TrashNet dataset "
            f"in {os.path.abspath('dataset')} directory."
        )

    # Split dataset if not already split
    if not (os.path.exists(train_dir) and os.path.exists(val_dir)):
        print("Splitting dataset...")
        split_dataset()

    # Verify no data leakage
    test_dir = "dataset/test"
    if os.path.exists(test_dir):
        verify_no_data_leakage(train_dir, val_dir, test_dir)

    # Create data generators
    print("Creating data generators...")
    train_gen, val_gen, _ = create_data_generators(
        train_dir=train_dir,
        val_dir=val_dir,
        batch_size=batch_size
    )

    # Save class indices
    save_class_indices(train_gen)

    # Build and compile model
    print("Building model...")
    model = build_model()
    model = compile_model(model)

    # Print model info
    params = count_parameters(model)
    print(f"\nModel Parameters:")
    print(f"  Trainable: {params['trainable']:,}")
    print(f"  Non-trainable: {params['non_trainable']:,}")
    print(f"  Total: {params['total']:,}")

    # Get callbacks
    callbacks = get_callbacks()

    # Phase 1: Initial training (frozen base)
    print(f"\n{'='*50}")
    print(f"Phase 1: Initial Training ({epochs_initial} epochs)")
    print(f"{'='*50}")

    start_time = time.time()
    history_initial = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=epochs_initial,
        callbacks=callbacks,
        verbose=1
    )
    initial_time = time.time() - start_time
    print(f"Initial training completed in {initial_time:.1f} seconds")

    # Phase 2: Fine-tuning (unfreeze last layers)
    history_fine = None
    if fine_tune:
        print(f"\n{'='*50}")
        print(f"Phase 2: Fine-tuning ({epochs_fine_tune} epochs)")
        print(f"{'='*50}")

        # Unfreeze base model layers
        model = unfreeze_base_model(model)

        # Recompile with lower learning rate
        model = recompile_for_fine_tuning(model)

        # Print updated parameter counts
        params = count_parameters(model)
        print(f"\nUpdated Model Parameters:")
        print(f"  Trainable: {params['trainable']:,}")
        print(f"  Non-trainable: {params['non_trainable']:,}")

        # Continue training
        start_time = time.time()
        history_fine = model.fit(
            train_gen,
            validation_data=val_gen,
            epochs=epochs_fine_tune,
            callbacks=callbacks,
            verbose=1
        )
        fine_tune_time = time.time() - start_time
        print(f"Fine-tuning completed in {fine_tune_time:.1f} seconds")

    # Combine histories
    combined_history = combine_histories(history_initial, history_fine)

    # Save model architecture
    save_model_architecture(model)

    # Save training history
    save_training_history(combined_history)

    # Plot training curves
    plot_training_history(combined_history)

    return model, combined_history


def combine_histories(history_initial, history_fine) -> Dict:
    """
    Combine initial training and fine-tuning histories.

    Args:
        history_initial: History from initial training
        history_fine: History from fine-tuning (can be None)

    Returns:
        Combined history dictionary
    """
    combined = {
        "accuracy": list(history_initial.history["accuracy"]),
        "val_accuracy": list(history_initial.history["val_accuracy"]),
        "loss": list(history_initial.history["loss"]),
        "val_loss": list(history_initial.history["val_loss"]),
        "precision": list(history_initial.history.get("precision", [])),
        "val_precision": list(history_initial.history.get("val_precision", [])),
        "recall": list(history_initial.history.get("recall", [])),
        "val_recall": list(history_initial.history.get("val_recall", []))
    }

    if history_fine:
        for key in combined:
            if key in history_fine.history:
                combined[key].extend(history_fine.history[key])

    return combined


def save_training_history(history: Dict, save_path: str = "results/training_history.json") -> None:
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'w') as f:
        json.dump(history, f, indent=4)
    print(f"Training history saved to {save_path}")


def plot_training_history(history: Dict, save_dir: str = RESULTS_DIR) -> None:
    os.makedirs(save_dir, exist_ok=True)

    epochs = range(1, len(history["accuracy"]) + 1)

    # Plot accuracy
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, history["accuracy"], 'b-', label="Training Accuracy", linewidth=2)
    plt.plot(epochs, history["val_accuracy"], 'r-', label="Validation Accuracy", linewidth=2)
    plt.title("Model Accuracy", fontsize=14, fontweight='bold')
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Accuracy", fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "accuracy_graph.png"), dpi=300, bbox_inches='tight')
    plt.close()

    # Plot loss
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, history["loss"], 'b-', label="Training Loss", linewidth=2)
    plt.plot(epochs, history["val_loss"], 'r-', label="Validation Loss", linewidth=2)
    plt.title("Model Loss", fontsize=14, fontweight='bold')
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Loss", fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "loss_graph.png"), dpi=300, bbox_inches='tight')
    plt.close()

    # Plot precision and recall if available
    if "precision" in history and history["precision"]:
        plt.figure(figsize=(10, 6))
        plt.plot(epochs, history["precision"], 'g-', label="Training Precision", linewidth=2)
        if "val_precision" in history and history["val_precision"]:
            plt.plot(epochs, history["val_precision"], 'orange', label="Validation Precision", linewidth=2)
        plt.title("Model Precision", fontsize=14, fontweight='bold')
        plt.xlabel("Epoch", fontsize=12)
        plt.ylabel("Precision", fontsize=12)
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "precision_graph.png"), dpi=300, bbox_inches='tight')
        plt.close()

    if "recall" in history and history["recall"]:
        plt.figure(figsize=(10, 6))
        plt.plot(epochs, history["recall"], 'm-', label="Training Recall", linewidth=2)
        if "val_recall" in history and history["val_recall"]:
            plt.plot(epochs, history["val_recall"], 'purple', label="Validation Recall", linewidth=2)
        plt.title("Model Recall", fontsize=14, fontweight='bold')
        plt.xlabel("Epoch", fontsize=12)
        plt.ylabel("Recall", fontsize=12)
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "recall_graph.png"), dpi=300, bbox_inches='tight')
        plt.close()

    print(f"Training plots saved to {save_dir}")


def load_training_history(load_path: str = "results/training_history.json") -> Dict:
    with open(load_path, 'r') as f:
        return json.load(f)


def evaluate_on_test_set(
    model: tf.keras.Model,
    test_dir: str = "dataset/test",
    batch_size: int = 32
) -> Dict:
    from src.data_preprocessing import create_data_generators

    _, _, test_gen = create_data_generators(
        train_dir="dataset/train",
        val_dir="dataset/val",
        test_dir=test_dir,
        batch_size=batch_size
    )

    print("Evaluating on test set...")
    results = model.evaluate(test_gen, verbose=1)

    metrics = {
        "test_loss": results[0],
        "test_accuracy": results[1],
        "test_precision": results[2] if len(results) > 2 else None,
        "test_recall": results[3] if len(results) > 3 else None,
        "test_auc": results[4] if len(results) > 4 else None
    }

    print(f"\nTest Results:")
    print(f"  Loss: {metrics['test_loss']:.4f}")
    print(f"  Accuracy: {metrics['test_accuracy']:.4f}")
    if metrics['test_precision']:
        print(f"  Precision: {metrics['test_precision']:.4f}")
    if metrics['test_recall']:
        print(f"  Recall: {metrics['test_recall']:.4f}")
    if metrics['test_auc']:
        print(f"  AUC: {metrics['test_auc']:.4f}")

    return metrics


if __name__ == "__main__":
    # Train the model
    print("Starting waste classification model training...")
    model, history = train_model()
    print("\nTraining completed!")

    # Evaluate on test set if available
    if os.path.exists("dataset/test"):
        evaluate_on_test_set(model)