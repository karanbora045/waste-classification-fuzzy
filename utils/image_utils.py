import os
import base64
from io import BytesIO
from typing import IO, Tuple, Optional, Union
from pathlib import Path

import numpy as np
import cv2
from PIL import Image, ImageEnhance, ImageFilter
import matplotlib.pyplot as plt


def load_image(image_path: str) -> Optional[np.ndarray]:
    try:
        img = cv2.imread(image_path)
        if img is None:
            return None
        # Convert BGR to RGB
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    except Exception:
        return None


def load_image_pil(image_path: str) -> Optional[Image.Image]:
    try:
        return Image.open(image_path).convert('RGB')
    except Exception:
        return None


def resize_image(
    image: np.ndarray,
    target_size: Tuple[int, int],
    keep_aspect_ratio: bool = False
) -> np.ndarray:
    if keep_aspect_ratio:
        h, w = image.shape[:2]
        target_w, target_h = target_size

        # Calculate scaling factor
        scale = min(target_w / w, target_h / h)
        new_w, new_h = int(w * scale), int(h * scale)

        # Resize
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

        # Create padded image
        padded = np.zeros((target_h, target_w, 3), dtype=image.dtype)
        y_offset = (target_h - new_h) // 2
        x_offset = (target_w - new_w) // 2
        padded[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized

        return padded
    else:
        return cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)


def normalize_image(image: np.ndarray) -> np.ndarray:
    return image.astype(np.float32) / 255.0


def denormalize_image(image: np.ndarray) -> np.ndarray:
    return (np.clip(image, 0, 1) * 255).astype(np.uint8)


def image_to_base64(image: Union[np.ndarray, Image.Image], format: str = 'PNG') -> str:
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image.astype(np.uint8))

    buffered = BytesIO()
    image.save(buffered, format=format)
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return img_str


def base64_to_image(base64_str: str) -> Optional[Image.Image]:
    try:
        img_data = base64.b64decode(base64_str)
        return Image.open(BytesIO(img_data)).convert('RGB')
    except Exception:
        return None


def enhance_image(
    image: Union[np.ndarray, Image.Image],
    brightness: float = 1.0,
    contrast: float = 1.0,
    sharpness: float = 1.0
) -> Image.Image:
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image.astype(np.uint8))

    if brightness != 1.0:
        enhancer = ImageEnhance.Brightness(image)
        image = enhancer.enhance(brightness)

    if contrast != 1.0:
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(contrast)

    if sharpness != 1.0:
        enhancer = ImageEnhance.Sharpness(image)
        image = enhancer.enhance(sharpness)

    return image


def apply_augmentation(
    image: np.ndarray,
    rotation_range: int = 20,
    width_shift_range: float = 0.2,
    height_shift_range: float = 0.2,
    zoom_range: float = 0.2,
    horizontal_flip: bool = True,
    brightness_range: Tuple[float, float] = (0.8, 1.2)
) -> np.ndarray:
    h, w = image.shape[:2]

    # Random rotation
    angle = np.random.uniform(-rotation_range, rotation_range)
    M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
    image = cv2.warpAffine(image, M, (w, h), borderMode=cv2.BORDER_REFLECT)

    # Random shift
    tx = np.random.uniform(-width_shift_range, width_shift_range) * w
    ty = np.random.uniform(-height_shift_range, height_shift_range) * h
    M = np.float32([[1, 0, tx], [0, 1, ty]])
    image = cv2.warpAffine(image, M, (w, h), borderMode=cv2.BORDER_REFLECT)

    # Random zoom
    zoom = np.random.uniform(1 - zoom_range, 1 + zoom_range)
    if zoom != 1.0:
        new_w, new_h = int(w * zoom), int(h * zoom)
        image = cv2.resize(image, (new_w, new_h))
        # Crop or pad to original size
        if zoom > 1.0:
            # Crop center
            start_x = (new_w - w) // 2
            start_y = (new_h - h) // 2
            image = image[start_y:start_y+h, start_x:start_x+w]
        else:
            # Pad
            padded = np.zeros((h, w, 3), dtype=image.dtype)
            start_x = (w - new_w) // 2
            start_y = (h - new_h) // 2
            padded[start_y:start_y+new_h, start_x:start_x+new_w] = image
            image = padded

    # Random horizontal flip
    if horizontal_flip and np.random.random() > 0.5:
        image = cv2.flip(image, 1)

    # Random brightness
    brightness = np.random.uniform(*brightness_range)
    image = np.clip(image * brightness, 0, 255).astype(np.uint8)

    return image


def create_montage(
    images: list,
    grid_size: Tuple[int, int] = None,
    image_size: Tuple[int, int] = (224, 224)
) -> np.ndarray:
    n_images = len(images)
    if n_images == 0:
        return np.zeros((image_size[1], image_size[0], 3), dtype=np.uint8)

    if grid_size is None:
        cols = int(np.ceil(np.sqrt(n_images)))
        rows = int(np.ceil(n_images / cols))
    else:
        rows, cols = grid_size

    montage_h = rows * image_size[1]
    montage_w = cols * image_size[0]
    montage = np.zeros((montage_h, montage_w, 3), dtype=np.uint8)

    for idx, img in enumerate(images):
        if idx >= rows * cols:
            break

        row = idx // cols
        col = idx % cols

        # Resize image
        resized = cv2.resize(img, image_size)

        # Place in montage
        y_start = row * image_size[1]
        y_end = y_start + image_size[1]
        x_start = col * image_size[0]
        x_end = x_start + image_size[0]

        montage[y_start:y_end, x_start:x_end] = resized

    return montage


def plot_prediction_barchart(
    probabilities: dict,
    save_path: str = None,
    title: str = "Prediction Probabilities"
) -> plt.Figure:
    classes = list(probabilities.keys())
    probs = list(probabilities.values())

    # Sort by probability
    sorted_pairs = sorted(zip(classes, probs), key=lambda x: x[1], reverse=True)
    classes, probs = zip(*sorted_pairs)

    fig, ax = plt.subplots(figsize=(10, 6))

    # Color the top prediction differently
    colors = ['#e74c3c'] + ['#3498db'] * (len(classes) - 1)

    bars = ax.barh(range(len(classes)), probs, color=colors, alpha=0.8, edgecolor='white', height=0.6)

    ax.set_yticks(range(len(classes)))
    ax.set_yticklabels(classes, fontsize=11)
    ax.set_xlabel('Probability', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    ax.set_xlim(0, 1.0)

    # Add percentage labels
    for i, (bar, prob) in enumerate(zip(bars, probs)):
        ax.text(prob + 0.01, bar.get_y() + bar.get_height()/2,
                f'{prob*100:.1f}%', va='center', fontsize=10, fontweight='bold')

    ax.invert_yaxis()
    ax.grid(True, alpha=0.3, axis='x')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig


def validate_image_file(
    file_path: Union[str, os.PathLike, IO[bytes]],
    allowed_extensions: list = None
) -> Tuple[bool, str]:
    if allowed_extensions is None:
        allowed_extensions = ['.jpg', '.jpeg', '.png']

    is_file_upload = hasattr(file_path, "read") and hasattr(file_path, "name")
    file_name = file_path.name if is_file_upload else os.fspath(file_path)
    ext = os.path.splitext(file_name)[1].lower()
    if ext not in allowed_extensions:
        return False, f"Invalid file format. Allowed: {', '.join(allowed_extensions)}"

    # Try to open with PIL
    try:
        if not is_file_upload and not os.path.exists(file_path):
            return False, "File does not exist"

        img = Image.open(file_path)
        img.verify()
        if is_file_upload:
            file_path.seek(0)
        return True, ""
    except Exception as e:
        return False, f"Invalid image file: {str(e)}"


def get_image_info(image_path: str) -> dict:
    try:
        img = Image.open(image_path)
        return {
            "filename": os.path.basename(image_path),
            "format": img.format,
            "mode": img.mode,
            "size": img.size,
            "width": img.width,
            "height": img.height,
            "file_size": os.path.getsize(image_path)
        }
    except Exception as e:
        return {"error": str(e)}


if __name__ == "__main__":
    # Test utilities
    print("Image utilities module loaded successfully")

    # Test base64 conversion
    test_img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    b64 = image_to_base64(test_img)
    print(f"Base64 length: {len(b64)}")

    # Test validation
    valid, msg = validate_image_file("nonexistent.jpg")
    print(f"Validation test: {valid}, {msg}")