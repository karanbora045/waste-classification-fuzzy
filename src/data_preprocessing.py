import os
import json
import shutil
import random
from pathlib import Path
from typing import Tuple, List, Dict, Optional

import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split

from config import (
    DATASET_DIR, CLASS_NAMES, IMAGE_SIZE, BATCH_SIZE,
    TRAIN_SPLIT, VAL_SPLIT, TEST_SPLIT, AUGMENTATION_CONFIG
)


def check_dataset_exists() -> bool:
    if not os.path.exists(DATASET_DIR):
        return False

    for class_name in CLASS_NAMES:
        class_path = os.path.join(DATASET_DIR, class_name)
        if not os.path.exists(class_path) or not os.listdir(class_path):
            return False

    return True


def get_dataset_stats() -> Dict:
    stats = {"classes": {}, "total": 0}

    for class_name in CLASS_NAMES:
        class_path = os.path.join(DATASET_DIR, class_name)
        if os.path.exists(class_path):
            images = [f for f in os.listdir(class_path)
                     if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            count = len(images)
            stats["classes"][class_name] = count
            stats["total"] += count

    return stats


def split_dataset(
    source_dir: str = DATASET_DIR,
    train_dir: str = "dataset/train",
    val_dir: str = "dataset/val",
    test_dir: str = "dataset/test",
    train_ratio: float = TRAIN_SPLIT,
    val_ratio: float = VAL_SPLIT,
    test_ratio: float = TEST_SPLIT,
    seed: int = 42
) -> None:
    random.seed(seed)
    np.random.seed(seed)

    # Create output directories
    for split_dir in [train_dir, val_dir, test_dir]:
        for class_name in CLASS_NAMES:
            os.makedirs(os.path.join(split_dir, class_name), exist_ok=True)

    for class_name in CLASS_NAMES:
        class_path = os.path.join(source_dir, class_name)
        if not os.path.exists(class_path):
            print(f"Warning: Class directory {class_path} does not exist")
            continue

        # Get all image files
        images = [f for f in os.listdir(class_path)
                 if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

        if not images:
            print(f"Warning: No images found in {class_path}")
            continue

        # Shuffle images
        random.shuffle(images)

        # Calculate split indices
        n_total = len(images)
        n_train = int(n_total * train_ratio)
        n_val = int(n_total * val_ratio)

        # Split images
        train_images = images[:n_train]
        val_images = images[n_train:n_train + n_val]
        test_images = images[n_train + n_val:]

        # Copy files to respective directories
        for img in train_images:
            shutil.copy2(
                os.path.join(class_path, img),
                os.path.join(train_dir, class_name, img)
            )
        for img in val_images:
            shutil.copy2(
                os.path.join(class_path, img),
                os.path.join(val_dir, class_name, img)
            )
        for img in test_images:
            shutil.copy2(
                os.path.join(class_path, img),
                os.path.join(test_dir, class_name, img)
            )

        print(f"{class_name}: {len(train_images)} train, {len(val_images)} val, {len(test_images)} test")

    print("\nDataset split completed!")


def create_data_generators(
    train_dir: str = "dataset/train",
    val_dir: str = "dataset/val",
    test_dir: str = "dataset/test",
    target_size: Tuple[int, int] = IMAGE_SIZE,
    batch_size: int = BATCH_SIZE,
    augmentation_config: Dict = AUGMENTATION_CONFIG
) -> Tuple[ImageDataGenerator, ImageDataGenerator, ImageDataGenerator]:
    # Training data generator with augmentation
    train_datagen = ImageDataGenerator(
        rescale=1.0/255.0,
        rotation_range=augmentation_config.get("rotation_range", 20),
        width_shift_range=augmentation_config.get("width_shift_range", 0.2),
        height_shift_range=augmentation_config.get("height_shift_range", 0.2),
        shear_range=augmentation_config.get("shear_range", 0.2),
        zoom_range=augmentation_config.get("zoom_range", 0.2),
        horizontal_flip=augmentation_config.get("horizontal_flip", True),
        brightness_range=augmentation_config.get("brightness_range", [0.8, 1.2]),
        fill_mode=augmentation_config.get("fill_mode", "nearest")
    )

    # Validation and test data generators (only rescaling)
    val_test_datagen = ImageDataGenerator(rescale=1.0/255.0)

    # Create generators
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=target_size,
        batch_size=batch_size,
        class_mode="categorical",
        classes=CLASS_NAMES,
        shuffle=True,
        seed=42
    )

    val_generator = val_test_datagen.flow_from_directory(
        val_dir,
        target_size=target_size,
        batch_size=batch_size,
        class_mode="categorical",
        classes=CLASS_NAMES,
        shuffle=False,
        seed=42
    )

    test_generator = val_test_datagen.flow_from_directory(
        test_dir,
        target_size=target_size,
        batch_size=batch_size,
        class_mode="categorical",
        classes=CLASS_NAMES,
        shuffle=False,
        seed=42
    )

    return train_generator, val_generator, test_generator


def preprocess_image(image_path: str, target_size: Tuple[int, int] = IMAGE_SIZE) -> np.ndarray:
    from tensorflow.keras.preprocessing import image

    img = image.load_img(image_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0

    return img_array


def preprocess_image_array(image_array: np.ndarray, target_size: Tuple[int, int] = IMAGE_SIZE) -> np.ndarray:
    import cv2

    # Resize image
    resized = cv2.resize(image_array, target_size)

    # Normalize
    normalized = resized.astype(np.float32) / 255.0

    # Add batch dimension
    batched = np.expand_dims(normalized, axis=0)

    return batched


def save_class_indices(generator, save_path: str = "models/class_indices.json") -> None:
    class_indices = generator.class_indices
    # Invert to get index -> class name mapping
    index_to_class = {v: k for k, v in class_indices.items()}

    with open(save_path, 'w') as f:
        json.dump(index_to_class, f, indent=4)

    print(f"Class indices saved to {save_path}")


def load_class_indices(load_path: str = "models/class_indices.json") -> Dict[int, str]:
    with open(load_path, 'r') as f:
        data = json.load(f)
    # Convert string keys to integers (JSON saves keys as strings)
    return {int(k): v for k, v in data.items()}


def verify_no_data_leakage(train_dir: str, val_dir: str, test_dir: str) -> bool:
    def get_filenames(directory):
        filenames = set()
        for class_name in CLASS_NAMES:
            class_path = os.path.join(directory, class_name)
            if os.path.exists(class_path):
                for f in os.listdir(class_path):
                    if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                        filenames.add(f)
        return filenames

    train_files = get_filenames(train_dir)
    val_files = get_filenames(val_dir)
    test_files = get_filenames(test_dir)

    # Check intersections
    train_val = train_files & val_files
    train_test = train_files & test_files
    val_test = val_files & test_files

    if train_val or train_test or val_test:
        print("WARNING: Data leakage detected!")
        if train_val:
            print(f"  Train-Val overlap: {len(train_val)} files")
        if train_test:
            print(f"  Train-Test overlap: {len(train_test)} files")
        if val_test:
            print(f"  Val-Test overlap: {len(val_test)} files")
        return False

    print("No data leakage detected. Splits are clean.")
    return True


if __name__ == "__main__":
    # Test dataset check
    if check_dataset_exists():
        print("Dataset found!")
        stats = get_dataset_stats()
        print(f"Total images: {stats['total']}")
        for class_name, count in stats['classes'].items():
            print(f"  {class_name}: {count}")
    else:
        print("Dataset not found. Please download and organize the TrashNet dataset.")
        print("Expected structure:")
        for class_name in CLASS_NAMES:
            print(f"  dataset/{class_name}/")