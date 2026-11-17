import os
import json
from typing import Tuple, Optional

import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import (
    ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard
)

from config import (
    MODEL_CONFIG, TRAIN_CONFIG, CLASS_NAMES, NUM_CLASSES,
    MODELS_DIR
)


def build_model(
    input_shape: Tuple[int, int, int] = MODEL_CONFIG["input_shape"],
    num_classes: int = NUM_CLASSES,
    base_trainable: bool = MODEL_CONFIG["base_model_trainable"],
    dropout_rate: float = MODEL_CONFIG["dropout_rate"]
) -> tf.keras.Model:
    # Load MobileNetV2 without the top classification layer
    base_model = MobileNetV2(
        weights="imagenet",
        include_top=False,
        input_shape=input_shape
    )

    # Freeze base model layers initially
    base_model.trainable = base_trainable

    # Build the model
    inputs = tf.keras.Input(shape=input_shape)

    # Preprocessing is handled in data pipeline (rescale 1/255)
    x = base_model(inputs, training=False)

    # Global Average Pooling reduces spatial dimensions to 1x1
    x = layers.GlobalAveragePooling2D()(x)

    # Dropout for regularization
    x = layers.Dropout(dropout_rate)(x)

    # Classification head
    outputs = layers.Dense(num_classes, activation="softmax", name="predictions")(x)

    model = tf.keras.Model(inputs, outputs, name="WasteClassifier")

    return model


def compile_model(
    model: tf.keras.Model,
    learning_rate: float = MODEL_CONFIG["learning_rate"]
) -> tf.keras.Model:
    model.compile(
        optimizer=optimizers.Adam(learning_rate=learning_rate),
        loss="categorical_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
            tf.keras.metrics.AUC(name="auc")
        ]
    )
    return model


def get_callbacks(
    checkpoint_path: str = TRAIN_CONFIG["model_checkpoint_path"],
    early_stopping_patience: int = TRAIN_CONFIG["early_stopping_patience"],
    reduce_lr_patience: int = TRAIN_CONFIG["reduce_lr_patience"],
    reduce_lr_factor: float = TRAIN_CONFIG["reduce_lr_factor"]
) -> list:
    # Ensure directory exists
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)

    callbacks = [
        ModelCheckpoint(
            filepath=checkpoint_path,
            monitor="val_accuracy",
            mode="max",
            save_best_only=True,
            save_weights_only=False,
            verbose=1
        ),
        EarlyStopping(
            monitor="val_accuracy",
            mode="max",
            patience=early_stopping_patience,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            mode="min",
            patience=reduce_lr_patience,
            factor=reduce_lr_factor,
            min_lr=1e-7,
            verbose=1
        ),
        TensorBoard(
            log_dir=os.path.join(MODELS_DIR, "logs"),
            histogram_freq=1,
            write_graph=True,
            update_freq="epoch"
        )
    ]

    return callbacks


def unfreeze_base_model(
    model: tf.keras.Model,
    num_layers: int = MODEL_CONFIG["fine_tune_layers"]
) -> tf.keras.Model:
    # Find the base model (MobileNetV2)
    base_model = None
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model) and layer.name == "mobilenetv2_1.00_224":
            base_model = layer
            break

    if base_model is None:
        # Try alternative naming
        for layer in model.layers:
            if "mobilenet" in layer.name.lower():
                base_model = layer
                break

    if base_model is None:
        print("Warning: Could not find MobileNetV2 base model")
        return model

    # Unfreeze the base model
    base_model.trainable = True

    # Freeze all but the last N layers
    for layer in base_model.layers[:-num_layers]:
        layer.trainable = False

    print(f"Fine-tuning: Unfroze last {num_layers} layers of base model")
    print(f"Trainable layers in base: {sum(1 for l in base_model.layers if l.trainable)}")
    print(f"Frozen layers in base: {sum(1 for l in base_model.layers if not l.trainable)}")

    return model


def recompile_for_fine_tuning(
    model: tf.keras.Model,
    learning_rate: float = MODEL_CONFIG["fine_tune_learning_rate"]
) -> tf.keras.Model:
    model.compile(
        optimizer=optimizers.Adam(learning_rate=learning_rate),
        loss="categorical_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
            tf.keras.metrics.AUC(name="auc")
        ]
    )
    return model


def save_model_architecture(model: tf.keras.Model, save_path: str = "models/model_architecture.json") -> None:
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    model_json = model.to_json()
    with open(save_path, 'w') as f:
        f.write(model_json)
    print(f"Model architecture saved to {save_path}")


def load_model_with_custom_objects(model_path: str) -> tf.keras.Model:

    return tf.keras.models.load_model(model_path)


def get_model_summary(model: tf.keras.Model) -> str:
    from io import StringIO
    stream = StringIO()
    model.summary(print_fn=lambda x: stream.write(x + '\n'))
    return stream.getvalue()


def count_parameters(model: tf.keras.Model) -> dict:
    trainable = sum(tf.keras.backend.count_params(w) for w in model.trainable_weights)
    non_trainable = sum(tf.keras.backend.count_params(w) for w in model.non_trainable_weights)

    return {
        "trainable": trainable,
        "non_trainable": non_trainable,
        "total": trainable + non_trainable
    }


if __name__ == "__main__":
    # Test model building
    print("Building model...")
    model = build_model()
    model = compile_model(model)

    print("\nModel Summary:")
    print(get_model_summary(model))

    print("\nParameter Counts:")
    params = count_parameters(model)
    print(f"  Trainable: {params['trainable']:,}")
    print(f"  Non-trainable: {params['non_trainable']:,}")
    print(f"  Total: {params['total']:,}")