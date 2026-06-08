"""
GaussianImageDistribution - VQA Dataset

This module provides a PyTorch Dataset class for loading VQA (Visual Question Answering)
data from text files. Each sample contains an image, a question, an answer label,
and a question type.

Data files format (one line per sample):
    - questions.txt: question text
    - img_filenames.txt: image filename (e.g., "000000397899.jpg")
    - types.txt: question type index (0, 1, 2, 3)
    - answers.txt: answer text (one of 430 classes)
"""

import warnings
from pathlib import Path
from typing import Dict, List, Optional

from PIL import Image
from torch.utils.data import Dataset

import config
from utils.logger import get_logger


logger = get_logger("vqa_dataset")


class VQADataset(Dataset):
    """
    Dataset for VQA (Visual Question Answering) downstream task.

    Reads 4 text files (questions, img_filenames, types, answers) and builds
    an answer-to-index mapping from the training set's unique answers.

    Each sample returns:
        - image: PIL.Image
        - question: str
        - answer_index: int (0~429)
        - question_type: int (0~3)
    """

    def __init__(
        self,
        questions_path: Path,
        img_filenames_path: Path,
        types_path: Path,
        answers_path: Path,
        images_dir: Path,
        answer_vocab: Optional[Dict[str, int]] = None,
    ):
        """
        Initialize the VQA dataset.

        Args:
            questions_path: Path to questions text file
            img_filenames_path: Path to image filenames text file
            types_path: Path to question types text file
            answers_path: Path to answers text file
            images_dir: Directory containing COCO images
            answer_vocab: Optional pre-built answer → index mapping.
                         If None, builds from this dataset's answers.
        """
        self.images_dir = images_dir

        # Read text files
        logger.info(f"Loading VQA data...")
        logger.info(f"  Questions: {questions_path}")
        logger.info(f"  Filenames: {img_filenames_path}")
        logger.info(f"  Types:     {types_path}")
        logger.info(f"  Answers:   {answers_path}")
        logger.info(f"  Images dir: {images_dir}")

        with open(questions_path, "r", encoding="utf-8") as f:
            self.questions = [line.strip() for line in f]

        with open(img_filenames_path, "r", encoding="utf-8") as f:
            self.img_filenames = [line.strip() for line in f]

        with open(types_path, "r", encoding="utf-8") as f:
            self.types = [int(line.strip()) for line in f]

        with open(answers_path, "r", encoding="utf-8") as f:
            self.answers = [line.strip() for line in f]

        # Validate consistency
        n = len(self.questions)
        assert len(self.img_filenames) == n, f"Mismatch: questions={n}, filenames={len(self.img_filenames)}"
        assert len(self.types) == n, f"Mismatch: questions={n}, types={len(self.types)}"
        assert len(self.answers) == n, f"Mismatch: questions={n}, answers={len(self.answers)}"

        # Build or use answer vocabulary
        if answer_vocab is not None:
            self.answer_vocab = answer_vocab
        else:
            unique_answers = sorted(set(self.answers))
            self.answer_vocab = {ans: idx for idx, ans in enumerate(unique_answers)}
            logger.info(f"Built answer vocabulary with {len(self.answer_vocab)} classes")

        self.num_classes = len(self.answer_vocab)
        logger.info(f"Loaded {n} VQA samples, {self.num_classes} answer classes")

    def get_answer_vocab(self) -> Dict[str, int]:
        """Return the answer → index mapping."""
        return self.answer_vocab.copy()

    def __len__(self) -> int:
        return len(self.questions)

    def __getitem__(self, idx: int) -> Optional[Dict[str, any]]:
        """
        Get a single sample.

        Returns:
            Dictionary with keys:
                - image: PIL.Image (RGB)
                - question: str
                - answer_index: int
                - question_type: int
            Returns None if image cannot be loaded.
        """
        image_path = self.images_dir / self.img_filenames[idx]

        # Load image
        try:
            if not image_path.exists():
                logger.warning(f"Image not found: {image_path}")
                return None
            image = Image.open(image_path)
            if image.mode != "RGB":
                image = image.convert("RGB")
        except Exception as e:
            warnings.warn(f"Failed to load image {image_path}: {e}")
            return None

        answer_text = self.answers[idx]
        answer_index = self.answer_vocab.get(answer_text, -1)
        if answer_index == -1:
            # Answer not in vocabulary (can happen for test set if vocab comes from train)
            # Map to a default class; this sample won't contribute meaningfully
            answer_index = 0

        return {
            "image": image,
            "question": self.questions[idx],
            "answer_index": answer_index,
            "question_type": self.types[idx],
        }


def vqa_collate_fn(batch: List[Optional[Dict]]) -> Optional[Dict[str, any]]:
    """
    Custom collate function that filters out None values (failed image loads).

    Returns:
        Dictionary with batched data:
            - images: List[PIL.Image]
            - questions: List[str]
            - answer_indices: List[int]
            - question_types: List[int]
        Returns None if all samples failed.
    """
    filtered = [item for item in batch if item is not None]
    if not filtered:
        return None

    return {
        "images": [item["image"] for item in filtered],
        "questions": [item["question"] for item in filtered],
        "answer_indices": [item["answer_index"] for item in filtered],
        "question_types": [item["question_type"] for item in filtered],
    }


if __name__ == "__main__":
    from utils.logger import setup_logger

    setup_logger("vqa_dataset", log_file=config.LOG_DIR / "vqa_dataset_test.log")

    # Test with training data
    dataset = VQADataset(
        questions_path=config.VQA_TRAIN_QUESTIONS,
        img_filenames_path=config.VQA_TRAIN_IMG_FILENAMES,
        types_path=config.VQA_TRAIN_TYPES,
        answers_path=config.VQA_TRAIN_ANSWERS,
        images_dir=config.VQA_IMAGES_DIR,
    )

    print(f"Dataset size: {len(dataset)}")
    print(f"Number of answer classes: {dataset.num_classes}")

    if len(dataset) > 0:
        sample = dataset[0]
        print(f"\nSample 0:")
        print(f"  Image size: {sample['image'].size}")
        print(f"  Question: {sample['question']}")
        print(f"  Answer index: {sample['answer_index']}")
        print(f"  Question type: {sample['question_type']}")
