"""
GaussianImageDistribution - Image Processing Utilities

This module provides utility functions for image processing operations.
"""

from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
from PIL import Image


def load_image(image_path: Path) -> Optional[Image.Image]:
    """
    Load image from file path.

    Args:
        image_path: Path to image file

    Returns:
        PIL Image object or None if loading fails
    """
    if not image_path.exists():
        return None

    try:
        image = Image.open(image_path)
        # Convert to RGB if necessary
        if image.mode != "RGB":
            image = image.convert("RGB")
        return image
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return None


def load_images(
    image_paths: List[Path],
    skip_missing: bool = True
) -> List[Optional[Image.Image]]:
    """
    Load multiple images from file paths.

    Args:
        image_paths: List of image file paths
        skip_missing: If True, skip missing files (return None for those entries)

    Returns:
        List of PIL Image objects (or None for failed loads)
    """
    images = []
    for path in image_paths:
        if skip_missing and not path.exists():
            images.append(None)
            continue
        images.append(load_image(path))
    return images


def resize_image(
    image: Image.Image,
    size: Tuple[int, int],
    resample: Image.Resampling = Image.Resampling.LANCZOS
) -> Image.Image:
    """
    Resize image to specified size.

    Args:
        image: PIL Image
        size: Target size as (width, height)
        resample: Resampling filter

    Returns:
        Resized PIL Image
    """
    return image.resize(size, resample)


def center_crop(
    image: Image.Image,
    size: Tuple[int, int]
) -> Image.Image:
    """
    Center crop image to specified size.

    Args:
        image: PIL Image
        size: Target size as (width, height)

    Returns:
        Center-cropped PIL Image
    """
    width, height = image.size
    target_width, target_height = size

    # Calculate crop coordinates
    left = (width - target_width) // 2
    top = (height - target_height) // 2
    right = left + target_width
    bottom = top + target_height

    return image.crop((left, top, right, bottom))


def get_image_stats(image_path: Path) -> Optional[dict]:
    """
    Get image statistics.

    Args:
        image_path: Path to image file

    Returns:
        Dictionary with image stats or None if loading fails
    """
    image = load_image(image_path)
    if image is None:
        return None

    return {
        "path": str(image_path),
        "size": image.size,  # (width, height)
        "mode": image.mode,
        "format": image.format,
    }


def validate_image_format(image_path: Path, allowed_formats: List[str]) -> bool:
    """
    Check if image has allowed format.

    Args:
        image_path: Path to image file
        allowed_formats: List of allowed formats (e.g., ["JPEG", "PNG"])

    Returns:
        True if format is allowed, False otherwise
    """
    image = load_image(image_path)
    if image is None:
        return False

    return image.format in allowed_formats


def image_to_numpy(image: Image.Image) -> np.ndarray:
    """
    Convert PIL Image to numpy array.

    Args:
        image: PIL Image

    Returns:
        Numpy array of shape (H, W, C) with values in [0, 255]
    """
    return np.array(image)


def numpy_to_image(array: np.ndarray) -> Image.Image:
    """
    Convert numpy array to PIL Image.

    Args:
        array: Numpy array of shape (H, W, C) with values in [0, 255]

    Returns:
        PIL Image
    """
    return Image.fromarray(array.astype(np.uint8))


if __name__ == "__main__":
    # Test image utilities
    from pathlib import Path

    # Create a test image
    test_image_path = Path("test_image.jpg")
    test_image = Image.new("RGB", (100, 100), color="red")
    test_image.save(test_image_path)

    # Test loading
    loaded = load_image(test_image_path)
    print(f"Loaded image: {loaded.size if loaded else 'Failed'}")

    # Test stats
    stats = get_image_stats(test_image_path)
    print(f"Image stats: {stats}")

    # Test validation
    is_valid = validate_image_format(test_image_path, ["JPEG", "PNG"])
    print(f"Is valid format: {is_valid}")

    # Cleanup
    test_image_path.unlink()
