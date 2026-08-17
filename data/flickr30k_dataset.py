"""
Flickr30K Dataset

Dataset loader for Flickr30K image-caption pairs.
Supports the standard Flickr30K split for cross-dataset evaluation (Exp1).

Expected directory structure:
    TrainDatasets/flickr30k/
        images/           # All Flickr30K images
        captions.txt      # Format: image_name\tcaption per line
                         # Each image has K (typically 5) captions
"""

import warnings
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, Subset

import config
from utils.logger import get_logger


logger = get_logger("flickr30k_dataset")


def _detect_separator(sample_line: str) -> Optional[str]:
    """Infer the CSV/TSV delimiter from one sample line ('\t' beats ',' beats None)."""
    if "\t" in sample_line:
        return "\t"
    if "," in sample_line:
        return ","
    return None


def _looks_like_filename(token: str) -> bool:
    """True if a token looks like an image filename (e.g. ends in .jpg)."""
    token = token.strip().lower()
    return any(token.endswith(ext) for ext in (".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"))


def _pick_column(columns, keywords, default):
    """Return the column name matching a keyword.

    Exact header match is preferred over substring containment, so e.g. with
    Flickr30K columns [image_name, comment_number, comment] the caption keyword
    'comment' matches the 'comment' column, not 'comment_number'.
    """
    lowered = [str(c).strip().lower() for c in columns]
    for kw in keywords:
        for i, c in enumerate(lowered):
            if c == kw:
                return columns[i]
    for kw in keywords:
        for i, c in enumerate(lowered):
            if kw in c:
                return columns[i]
    return columns[default]


class Flickr30KDataset(Dataset):
    """
    Dataset for Flickr30K image-caption pairs.

    Each sample contains:
        - image: PIL.Image
        - image_name: str
        - captions: List[str] (K captions per image)

    The dataset groups captions by image, so each sample corresponds to
    one unique image with all its associated captions.
    """

    def __init__(
        self,
        root_dir: Optional[Path] = None,
        images_dir: Optional[Path] = None,
        captions_path: Optional[Path] = None,
        num_captions: int = 5,
        split: Optional[str] = None,
    ):
        """
        Initialize Flickr30K dataset.

        Args:
            root_dir: Root directory of Flickr30K (contains images/ and captions.txt)
            images_dir: Directory containing images (overrides root_dir/images)
            captions_path: Path to captions file (overrides root_dir/captions.txt)
            num_captions: Number of captions per image (default 5)
            split: Optional split ("train", "val", "test") for standard Flickr30K splits
        """
        self.root_dir = root_dir or config.FLICKR30K_ROOT
        self.images_dir = images_dir or config.FLICKR30K_IMAGES_DIR
        self.captions_path = captions_path or config.FLICKR30K_CAPTIONS_PATH
        self.num_captions = num_captions

        # Load and group captions by image
        self._load_captions()

        # Apply split if specified
        if split is not None:
            self._apply_split(split)

        logger.info(f"Flickr30K: {len(self)} images loaded")

    def _load_captions(self):
        """Load captions and group them by image name.

        Handles the standard Flickr30K CSV (header: image_name,comment_number,
        comment), a plain image,caption CSV/TSV, or a headerless
        image<TAB/COMMA>caption file. The delimiter, whether a header row is
        present, and the image/caption columns are all detected automatically.
        """
        if not self.captions_path.exists():
            logger.warning(
                f"Captions file not found: {self.captions_path}\n"
                f"Expected Flickr30K CSV (image_name,comment_number,comment) or "
                f"image,caption / image<TAB>caption."
            )
            self.image_names = []
            self.captions_dict = {}
            return

        logger.info(f"Loading Flickr30K captions from: {self.captions_path}")

        # Peek at the first non-empty line to detect the delimiter and whether
        # the file has a header row (a header field is not an image filename).
        with open(self.captions_path, "r", encoding="utf-8") as f:
            sample = ""
            for line in f:
                if line.strip():
                    sample = line.rstrip("\n")
                    break
        sep = _detect_separator(sample)
        has_header = sep is not None and not _looks_like_filename(sample.split(sep)[0])

        if has_header:
            df = pd.read_csv(self.captions_path, sep=sep, engine="python")
            if df.shape[1] < 2:
                raise ValueError(
                    f"Could not parse image/caption pairs from {self.captions_path} "
                    f"(found {df.shape[1]} column(s)).")
            img_col = _pick_column(
                df.columns, ("image_name", "image", "filename", "file"),
                default=df.columns[0])
            # Caption column: prefer a named caption/comment column, else the
            # last column (skips a comment_number/index column in the middle).
            cap_col = _pick_column(
                df.columns, ("caption", "comment", "text", "sentence"),
                default=df.columns[-1])
        else:
            # Headerless: column 0 is the image, last column is the caption.
            df = pd.read_csv(self.captions_path, sep=sep, header=None, engine="python")
            if df.shape[1] < 2:
                raise ValueError(
                    f"Could not parse image/caption pairs from {self.captions_path} "
                    f"(found {df.shape[1]} column(s)).")
            img_col = df.columns[0]
            cap_col = df.columns[-1]

        captions_dict = {}
        for _, row in df.iterrows():
            img_name = str(row[img_col]).strip()
            caption = str(row[cap_col]).strip()
            if not img_name or img_name.lower() in ("nan", "none"):
                continue
            captions_dict.setdefault(img_name, []).append(caption)

        # Keep only images with at least one caption.
        self.image_names = []
        self.captions_dict = {}
        for img_name, caps in captions_dict.items():
            if caps:
                self.image_names.append(img_name)
                self.captions_dict[img_name] = caps

        logger.info(f"Found {len(self.image_names)} unique images")

    def _apply_split(self, split: str):
        """
        Apply a percentage-based split.

        - train: first 90%
        - val:   next 5%
        - test:  last 5%
        """
        n = len(self.image_names)

        # Standard split ratios
        if split == "train":
            # First 90% for training
            split_end = int(n * 0.9)
            self.image_names = self.image_names[:split_end]
        elif split == "val":
            # Next 5% for validation
            val_start = int(n * 0.9)
            val_end = int(n * 0.95)
            self.image_names = self.image_names[val_start:val_end]
        elif split == "test":
            test_start = int(n * 0.95)
            self.image_names = self.image_names[test_start:]
        else:
            raise ValueError(f"Unknown split: {split}. Use 'train', 'val', or 'test'.")

        logger.info(f"Applied '{split}' split: {len(self.image_names)} images")

    def __len__(self) -> int:
        return len(self.image_names)

    def _get_captions(self, idx: int) -> List[str]:
        """Get captions for image at idx, pad/truncate to num_captions."""
        img_name = self.image_names[idx]
        captions = self.captions_dict.get(img_name, [])

        if not captions:
            captions = ["no caption available"]

        # Pad or truncate
        if len(captions) < self.num_captions:
            while len(captions) < self.num_captions:
                captions.append(captions[0])
        elif len(captions) > self.num_captions:
            captions = captions[:self.num_captions]

        return captions

    def __getitem__(self, idx: int) -> Optional[Dict[str, any]]:
        """
        Get a single sample.

        Returns:
            Dictionary with:
                - image: PIL.Image
                - image_name: str
                - captions: List[str]
            Returns None if image cannot be loaded.
        """
        img_name = self.image_names[idx]
        image_path = self.images_dir / img_name

        if not image_path.exists():
            # Try common extensions
            for ext in [".jpg", ".png", ".jpeg"]:
                alt_path = self.images_dir / (img_name + ext)
                if alt_path.exists():
                    image_path = alt_path
                    break
            else:
                logger.warning(f"Image not found: {image_path}")
                return None

        try:
            image = Image.open(image_path)
            if image.mode != "RGB":
                image = image.convert("RGB")
        except Exception as e:
            warnings.warn(f"Failed to load image {image_path}: {e}")
            return None

        captions = self._get_captions(idx)

        return {
            "image": image,
            "image_path": str(image_path),
            "image_name": img_name,
            "captions": captions,
        }


def get_flickr30k_test_loader(
    batch_size: int = 32,
    num_workers: int = 0,
    num_captions: int = 5,
) -> Optional[Dict]:
    """
    Get Flickr30K test dataloader for Exp1 evaluation.

    Returns None if Flickr30K data is not available.

    Returns:
        Dictionary with 'dataloader' and 'dataset' keys, or None
    """
    from torch.utils.data import DataLoader
    from data.caption_dataset import filter_none_collate

    if not config.FLICKR30K_ROOT.exists():
        logger.warning(
            f"Flickr30K not found at {config.FLICKR30K_ROOT}. "
            f"Skipping Flickr30K evaluation."
        )
        return None

    dataset = Flickr30KDataset(
        split="test",
        num_captions=num_captions,
    )

    if len(dataset) == 0:
        logger.warning("Flickr30K test set is empty")
        return None

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=filter_none_collate,
    )

    return {"dataloader": dataloader, "dataset": dataset}
