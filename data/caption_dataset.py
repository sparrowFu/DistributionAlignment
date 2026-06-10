"""
GaussianImageDistribution - Image Caption Dataset

This module provides a PyTorch Dataset class for loading image-caption pairs
from a parquet file and image directory.
"""

import warnings
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

import config
from utils.logger import get_logger


# Get logger
logger = get_logger("caption_dataset")


class ImageCaptionDataset(Dataset):
    """
    Dataset for image-caption pairs.

    Each sample contains:
        - image: PIL.Image
        - image_path: str
        - image_name: str
        - captions: List[str]

    The dataset reads from a parquet file with columns:
        - url: Original image URL
        - caption: List of text descriptions (List[str])
        - image_file_name: Image filename (e.g., "000000000009.jpg")

    Images are loaded from IMAGES_DIR / image_file_name.
    """

    def __init__(
        self,
        captions_path: Optional[Path] = None,
        images_dir: Optional[Path] = None,
        num_captions: int = 5,
        transform=None
    ):
        """
        Initialize the dataset.

        Args:
            captions_path: Path to parquet file (uses config default if None)
            images_dir: Directory containing images (uses config default if None)
            num_captions: Number of captions per image (will pad if necessary)
            transform: Optional transform to apply to images
        """
        self.captions_path = captions_path or config.CAPTIONS_PATH
        self.images_dir = images_dir or config.IMAGES_DIR
        self.num_captions = num_captions
        self.transform = transform

        # Load parquet file
        logger.info(f"Loading captions from: {self.captions_path}")
        self.data = pd.read_parquet(self.captions_path)

        # Filter out samples without image_file_name
        original_count = len(self.data)
        self.data = self.data[self.data["image_file_name"].notna()].reset_index(drop=True)

        if len(self.data) < original_count:
            logger.warning(f"Filtered {original_count - len(self.data)} samples without image_file_name")

        logger.info(f"Loaded {len(self.data)} samples")

        # Check for missing images
        self._validate_images()

    def _validate_images(self) -> None:
        """
        Check which images exist and log warnings for missing ones.
        This is a lightweight check - samples are still included even if image is missing.
        """
        logger.info("Checking sample image availability (quick check of first 100 samples)...")
        missing_count = 0
        check_samples = min(len(self.data), 100)  # Reduced from 1000 for faster initialization

        for i in range(check_samples):
            image_file_name = self.data.loc[i, "image_file_name"]
            image_path = self.images_dir / image_file_name

            if not image_path.exists():
                missing_count += 1

        if missing_count > 0:
            missing_ratio = missing_count / check_samples
            logger.warning(
                f"Found {missing_count} missing images in first {check_samples} samples "
                f"({missing_ratio:.1%} missing rate)"
            )
        else:
            logger.info("All checked images found")

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.data)

    def _get_captions(self, idx: int) -> List[str]:
        """
        Get captions for a sample, padding or truncating as needed.

        Args:
            idx: Sample index

        Returns:
            List of exactly num_captions strings
        """
        # Get caption column (should be List[str])
        # Use iloc for faster positional access
        captions = self.data.iloc[idx]["caption"]

        # Convert to list - handle numpy arrays, lists, and strings
        if isinstance(captions, list):
            captions = list(captions)
        elif isinstance(captions, str):
            captions = [captions]
        else:
            # Handle numpy array or other iterable types
            try:
                captions = list(captions)
            except (TypeError, ValueError):
                # If conversion fails, return empty list
                captions = []

        # Filter out non-string items and convert to strings
        captions = [str(c) for c in captions if c is not None]

        # If no valid captions, return a placeholder
        if not captions:
            captions = ["no caption available"]

        # Pad or truncate to num_captions
        if len(captions) < self.num_captions:
            # Repeat captions to fill up to num_captions
            # Use a safe loop that won't infinite loop
            if captions:  # Only if we have at least one caption
                while len(captions) < self.num_captions:
                    captions.append(captions[0])  # Just repeat first caption
        elif len(captions) > self.num_captions:
            captions = captions[:self.num_captions]

        return captions

    def __getitem__(self, idx: int) -> Optional[Dict[str, any]]:
        """
        Get a single sample.

        Args:
            idx: Sample index

        Returns:
            Dictionary with keys:
                - image: PIL.Image or None if image missing
                - image_path: str
                - image_name: str
                - captions: List[str]
            Returns None if image cannot be loaded
        """
        # Get image file name
        image_file_name = self.data.loc[idx, "image_file_name"]
        image_path = self.images_dir / image_file_name

        # Check if file exists before attempting to open
        if not image_path.exists():
            logger.warning(f"Image not found: {image_path}")
            return None

        # Load image
        try:
            image = Image.open(image_path)
            if image.mode != "RGB":
                image = image.convert("RGB")
        except Exception as e:
            warnings.warn(f"Failed to load image {image_path}: {e}")
            return None

        # Apply transform if provided
        if self.transform is not None:
            image = self.transform(image)

        # Get captions
        captions = self._get_captions(idx)

        return {
            "image": image,
            "image_path": str(image_path),
            "image_name": image_file_name,
            "captions": captions
        }


def filter_none_collate(batch):
    """
    Collate function that filters out None values and converts to batch format.

    Used by all training and evaluation scripts for image-caption data.

    Args:
        batch: List of dataset samples (may contain None for failed loads)

    Returns:
        Dictionary with batched tensors, or None if all samples failed
    """
    filtered = [item for item in batch if item is not None]

    if not filtered:
        return None

    return {
        "image": [item["image"] for item in filtered],
        "captions": [item["captions"] for item in filtered],
        "image_path": [item["image_path"] for item in filtered],
        "image_name": [item["image_name"] for item in filtered],
    }


if __name__ == "__main__":
    # Test dataset
    import config
    from utils.logger import setup_logger

    # Setup logger with console output
    setup_logger("caption_dataset", log_file=config.LOG_DIR / "dataset_test.log")

    # Create dataset
    dataset = ImageCaptionDataset(
        num_captions=config.NUM_CAPTIONS
    )

    print(f"Dataset size: {len(dataset)}")

    # Get first sample
    if len(dataset) > 0:
        sample = dataset[0]
        print(f"\nSample 0:")
        print(f"  Image path: {sample['image_path']}")
        print(f"  Image name: {sample['image_name']}")
        print(f"  Image size: {sample['image'].size}")
        print(f"  Captions ({len(sample['captions'])}):")
        for i, caption in enumerate(sample['captions']):
            print(f"    [{i+1}] {caption}")
