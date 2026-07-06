"""
GaussianImageDistribution - CLIP Zero-Shot VQA Model

This module implements a CLIP Zero-Shot baseline for VQA classification.
It uses frozen CLIP to compute cosine similarity between image features
and precomputed answer text features. No training required.

Approach:
    1. Precompute text features for all 430 answer candidates
    2. At inference, encode image and question with frozen CLIP
    3. Compute combined similarity: 0.5 * (img@ans.T) + 0.5 * (q@ans.T)
    4. Return similarity scores as logits

This baseline tests how much pre-trained CLIP already knows about
visual concepts and their textual descriptions.
"""

from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPModel, CLIPProcessor

import config
from utils.logger import get_logger
from utils.image_preprocess import preprocess_images_on_gpu


logger = get_logger("clip_zero_shot")


class CLIPZeroShotVQA(nn.Module):
    """
    CLIP Zero-Shot VQA baseline.

    Uses frozen CLIP to compute cosine similarity between combined
    image+question features and precomputed answer text features.
    No training required - purely inference-based.
    """

    def __init__(
        self,
        answer_list: List[str],
        model_path: Optional[str] = None,
        img_weight: float = 0.5,
        text_weight: float = 0.5,
    ):
        """
        Initialize CLIP Zero-Shot VQA model.

        Args:
            answer_list: List of answer strings (430 answers)
            model_path: Path to local CLIP model directory
            img_weight: Weight for image-answer similarity
            text_weight: Weight for question-answer similarity
        """
        super().__init__()

        self.model_path = model_path or str(config.CLIP_VIT_L_14_PATH)
        self.img_weight = img_weight
        self.text_weight = text_weight

        # Load CLIP model from local files
        logger.info(f"Loading CLIP model from: {self.model_path}")
        self.clip_model = CLIPModel.from_pretrained(
            self.model_path,
            local_files_only=True,
        )

        # Load CLIP processor from local files
        logger.info(f"Loading CLIP processor from: {self.model_path}")
        self.processor = CLIPProcessor.from_pretrained(
            self.model_path,
            local_files_only=True,
        )

        # Freeze all CLIP parameters
        for param in self.clip_model.parameters():
            param.requires_grad = False

        # Precompute answer text features
        self._precompute_answer_features(answer_list)

        logger.info(
            f"CLIP Zero-Shot VQA initialized: {len(answer_list)} answers, "
            f"img_weight={img_weight}, text_weight={text_weight}"
        )

    def _precompute_answer_features(self, answer_list: List[str]) -> None:
        """
        Precompute and cache normalized text features for all answer candidates.

        Args:
            answer_list: List of answer strings
        """
        logger.info(f"Precomputing features for {len(answer_list)} answers...")

        # Use simple template for answer encoding
        templates = [f"{answer}" for answer in answer_list]

        with torch.no_grad():
            text_inputs = self.processor(
                text=templates,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=77,
            )
            text_outputs = self.clip_model.text_model(
                input_ids=text_inputs["input_ids"],
                attention_mask=text_inputs["attention_mask"],
            )
            text_feats = text_outputs.pooler_output  # (num_answers, 768)
            text_feats = self.clip_model.text_projection(text_feats)  # (num_answers, 768)
            text_feats = F.normalize(text_feats, dim=-1)

        # Register as buffer (saved with model state but not a parameter)
        self.register_buffer("answer_features", text_feats)
        logger.info(f"Answer features precomputed: {text_feats.shape}")

    def encode_image(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Encode image with CLIP and normalize.

        Args:
            pixel_values: Image tensor (B, 3, 224, 224)

        Returns:
            Normalized image features (B, 768)
        """
        vision_outputs = self.clip_model.vision_model(pixel_values=pixel_values)
        img_feat = vision_outputs.pooler_output
        img_feat = self.clip_model.visual_projection(img_feat)
        img_feat = F.normalize(img_feat, dim=-1)
        return img_feat

    def encode_text(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Encode text with CLIP and normalize.

        Args:
            input_ids: Token IDs (B, seq_len)
            attention_mask: Attention mask (B, seq_len)

        Returns:
            Normalized text features (B, 768)
        """
        text_outputs = self.clip_model.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        text_feat = text_outputs.pooler_output
        text_feat = self.clip_model.text_projection(text_feat)
        text_feat = F.normalize(text_feat, dim=-1)
        return text_feat

    def forward(
        self,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass - compute zero-shot VQA logits.

        Combined similarity:
            logits = img_weight * (img_feat @ ans.T) + text_weight * (q_feat @ ans.T)

        Args:
            pixel_values: Image tensor (B, 3, 224, 224)
            input_ids: Token IDs (B, seq_len)
            attention_mask: Attention mask (B, seq_len)

        Returns:
            Logits tensor (B, num_answers)
        """
        img_feat = self.encode_image(pixel_values)  # (B, 768)
        q_feat = self.encode_text(input_ids, attention_mask)  # (B, 768)

        # Cosine similarity with answer features
        img_score = img_feat @ self.answer_features.T  # (B, num_answers)
        q_score = q_feat @ self.answer_features.T  # (B, num_answers)

        # Weighted combination
        logits = self.img_weight * img_score + self.text_weight * q_score

        return logits

    def process_images(self, images: List) -> torch.Tensor:
        """Process PIL images to tensors using CLIP processor."""
        device = next(self.parameters()).device
        return preprocess_images_on_gpu(images, device)

    def process_text(self, texts: List[str]) -> Dict[str, torch.Tensor]:
        """Process text strings to token IDs using CLIP processor."""
        return self.processor(
            text=texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=77,
        )

    def trainable_parameters(self) -> List[nn.Parameter]:
        """Get list of trainable parameters (none for zero-shot)."""
        return [p for p in self.parameters() if p.requires_grad]

    def num_trainable_parameters(self) -> int:
        """Count trainable parameters (should be 0)."""
        return sum(p.numel() for p in self.trainable_parameters())

    def save(self, path: str) -> None:
        """Save model state (answer features + config)."""
        state = {
            "model_state_dict": self.state_dict(),
            "img_weight": self.img_weight,
            "text_weight": self.text_weight,
        }
        torch.save(state, path)
        logger.info(f"Model saved to: {path}")

    def load(self, path: str, strict: bool = True) -> None:
        """Load model state."""
        state = torch.load(path, map_location="cpu", weights_only=False)
        self.load_state_dict(state["model_state_dict"], strict=strict)
        logger.info(f"Model loaded from: {path}")
        if "img_weight" in state:
            self.img_weight = state["img_weight"]
        if "text_weight" in state:
            self.text_weight = state["text_weight"]


if __name__ == "__main__":
    from utils.logger import setup_logger
    from utils.seed import set_seed

    setup_logger("clip_zero_shot", config.LOG_DIR / "clip_zero_shot_test.log")
    set_seed(config.SEED)

    # Test with dummy answer list
    dummy_answers = [f"answer_{i}" for i in range(430)]

    model = CLIPZeroShotVQA(
        answer_list=dummy_answers,
        img_weight=0.5,
        text_weight=0.5,
    )

    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Trainable parameters: {model.num_trainable_parameters():,}")
    print(f"Answer features shape: {model.answer_features.shape}")

    batch_size = 2
    dummy_images = torch.randn(batch_size, 3, 224, 224)
    dummy_input_ids = torch.randint(0, 49408, (batch_size, 77))
    dummy_attention_mask = torch.ones(batch_size, 77, dtype=torch.long)

    with torch.no_grad():
        logits = model(dummy_images, dummy_input_ids, dummy_attention_mask)

    print(f"\nOutput logits shape: {logits.shape}")
    assert logits.shape == (batch_size, 430), f"Expected (2, 430), got {logits.shape}"
    print("Forward pass test passed!")
