import os
import json
import numpy as np
from typing import Dict, List, Tuple, Optional, Union

import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import cv2

from config import (
    CLASS_NAMES, NUM_CLASSES, IMAGE_SIZE,
    MODELS_DIR, CONFIDENCE_THRESHOLD
)
from src.data_preprocessing import load_class_indices, preprocess_image_array

class WasteClassifier:
    def __init__(self, model_path: str = None, class_indices_path: str = None):
        self.model_path = model_path or os.path.join(MODELS_DIR, "best_model.keras")
        self.class_indices_path = class_indices_path or os.path.join(MODELS_DIR, "class_indices.json")

        self.model = None
        self.class_names = CLASS_NAMES
        self.index_to_class = None
        self.loaded = False

    def load(self) -> bool:
        try:
            # Load model
            if not os.path.exists(self.model_path):
                print(f"Model not found at {self.model_path}")
                return False

            print(f"Loading model from {self.model_path}...")
            self.model = load_model(self.model_path)

            # Load class indices
            if os.path.exists(self.class_indices_path):
                self.index_to_class = load_class_indices(self.class_indices_path)
                # Ensure correct order
                self.class_names = [self.index_to_class[i] for i in range(NUM_CLASSES)]
            else:
                print("Class indices not found, using default order")
                self.index_to_class = {i: name for i, name in enumerate(CLASS_NAMES)}

            self.loaded = True
            print("Model loaded successfully!")
            return True

        except Exception as e:
            print(f"Error loading model: {e}")
            self.loaded = False
            return False

    def predict_image_path(self, image_path: str) -> Dict:
        if not self.loaded:
            if not self.load():
                return {"error": "Model not loaded"}

        if not os.path.exists(image_path):
            return {"error": f"Image not found: {image_path}"}

        try:
            # Preprocess image
            img_array = preprocess_image_array(
                cv2.imread(image_path)[:, :, ::-1],  # BGR to RGB
                target_size=IMAGE_SIZE
            )

            return self._predict_from_array(img_array)

        except Exception as e:
            return {"error": f"Prediction failed: {str(e)}"}

    def predict_image_array(self, image_array: np.ndarray) -> Dict:
        if not self.loaded:
            if not self.load():
                return {"error": "Model not loaded"}

        try:
            img_array = preprocess_image_array(image_array, target_size=IMAGE_SIZE)
            return self._predict_from_array(img_array)
        except Exception as e:
            return {"error": f"Prediction failed: {str(e)}"}

    def predict_pil_image(self, pil_image) -> Dict:
        if not self.loaded:
            if not self.load():
                return {"error": "Model not loaded"}

        try:
            # Convert PIL to numpy array
            img_array = np.array(pil_image.resize(IMAGE_SIZE))
            if img_array.shape[-1] == 4:  # RGBA
                img_array = img_array[:, :, :3]
            img_array = np.expand_dims(img_array.astype(np.float32) / 255.0, axis=0)
            return self._predict_from_array(img_array)
        except Exception as e:
            return {"error": f"Prediction failed: {str(e)}"}

    def _predict_from_array(self, img_array: np.ndarray) -> Dict:
        # Get predictions
        predictions = self.model.predict(img_array, verbose=0)
        probabilities = predictions[0]

        # Get predicted class
        predicted_idx = int(np.argmax(probabilities))
        predicted_class = self.class_names[predicted_idx]
        confidence = float(probabilities[predicted_idx])

        # Get all class probabilities
        all_probs = {
            self.class_names[i]: float(probabilities[i])
            for i in range(NUM_CLASSES)
        }

        # Sort by probability
        sorted_probs = dict(sorted(all_probs.items(), key=lambda x: x[1], reverse=True))

        # Check confidence
        low_confidence = confidence < CONFIDENCE_THRESHOLD

        result = {
            "predicted_class": predicted_class,
            "confidence": confidence,
            "confidence_percentage": f"{confidence * 100:.2f}%",
            "all_probabilities": sorted_probs,
            "low_confidence": low_confidence,
            "class_names": self.class_names
        }

        return result

    def predict_batch(self, image_paths: List[str]) -> List[Dict]:
        results = []
        for path in image_paths:
            results.append(self.predict_image_path(path))
        return results

    def get_model_info(self) -> Dict:
        if not self.loaded:
            self.load()

        if self.model is None:
            return {"error": "Model not loaded"}

        # Count parameters
        trainable = sum(tf.keras.backend.count_params(w) for w in self.model.trainable_weights)
        non_trainable = sum(tf.keras.backend.count_params(w) for w in self.model.non_trainable_weights)

        return {
            "model_path": self.model_path,
            "input_shape": self.model.input_shape,
            "output_shape": self.model.output_shape,
            "num_classes": NUM_CLASSES,
            "class_names": self.class_names,
            "trainable_params": trainable,
            "non_trainable_params": non_trainable,
            "total_params": trainable + non_trainable
        }


def predict_single_image(
    image_path: str,
    model_path: str = None
) -> Dict:
    classifier = WasteClassifier(model_path)
    return classifier.predict_image_path(image_path)


if __name__ == "__main__":
    # Test prediction
    import sys

    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        # Try to find a test image
        test_dir = "dataset/test"
        if os.path.exists(test_dir):
            for class_name in CLASS_NAMES:
                class_dir = os.path.join(test_dir, class_name)
                if os.path.exists(class_dir):
                    images = [f for f in os.listdir(class_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
                    if images:
                        image_path = os.path.join(class_dir, images[0])
                        break
            else:
                image_path = None
        else:
            image_path = None

    if image_path:
        print(f"Testing prediction on: {image_path}")
        result = predict_single_image(image_path)

        if "error" in result:
            print(f"Error: {result['error']}")
        else:
            print(f"\nPrediction: {result['predicted_class']}")
            print(f"Confidence: {result['confidence_percentage']}")
            print("\nAll Probabilities:")
            for cls, prob in result['all_probabilities'].items():
                print(f"  {cls:10s}: {prob*100:.2f}%")
            if result['low_confidence']:
                print(f"\n⚠️ Warning: Low confidence (< {CONFIDENCE_THRESHOLD*100}%)")
    else:
        print("No test image found. Please provide an image path as argument.")
        print("Usage: python predict.py <image_path>")