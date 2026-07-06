"""
GaussianImageDistribution - CLIP Baseline Model

This module implements a CLIP fine-tuning model with support for
freezing image/text encoders.
"""

from typing import Dict, List, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPModel, CLIPProcessor

import config
from utils.logger import get_logger
from utils.image_preprocess import preprocess_images_on_gpu


logger = get_logger("clip_baseline")


class CLIPFineTuneBaseline(nn.Module):
    """
    CLIP Fine-tuning Baseline Model.

    This model wraps the pre-trained CLIP model and provides:
    - Image and text encoding
    - Optional freezing of image/text encoders
    - Normalized feature extraction

    The model loads from a local directory without internet access.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        freeze_image: bool = False,
        freeze_text: bool = False
    ):
        """
        Initialize CLIP fine-tuning model.

        Args:
            model_path: Path to local CLIP model directory
                      (uses config.CLIP_VIT_L_14_PATH if None)
            freeze_image: Whether to freeze image encoder
            freeze_text: Whether to freeze text encoder
        """
        super().__init__()

        self.model_path = model_path or str(config.CLIP_VIT_L_14_PATH)
        self.freeze_image = freeze_image
        self.freeze_text = freeze_text

        # Load CLIP model from local files
        logger.info(f"Loading CLIP model from: {self.model_path}")
        self.clip_model = CLIPModel.from_pretrained(
            self.model_path,
            local_files_only=True
        )

        # Load processor from local files
        logger.info(f"Loading CLIP processor from: {self.model_path}")
        self.processor = CLIPProcessor.from_pretrained(
            self.model_path,
            local_files_only=True
        )

        # Freeze encoders if requested
        if freeze_image:
            self._freeze_image_encoder()
            logger.info("Image encoder frozen")

        if freeze_text:
            self._freeze_text_encoder()
            logger.info("Text encoder frozen")

        logger.info(f"CLIP model loaded: freeze_image={freeze_image}, freeze_text={freeze_text}")

    def _freeze_image_encoder(self) -> None:
        """Freeze the image encoder parameters."""
        for param in self.clip_model.vision_model.parameters():
            param.requires_grad = False
        # Also freeze projection if exists
        if hasattr(self.clip_model, "visual_projection"):
            for param in self.clip_model.visual_projection.parameters():
                param.requires_grad = False

    def _freeze_text_encoder(self) -> None:
        """Freeze the text encoder parameters."""
        for param in self.clip_model.text_model.parameters():
            param.requires_grad = False
        # Also freeze projection if exists
        if hasattr(self.clip_model, "text_projection"):
            for param in self.clip_model.text_projection.parameters():
                param.requires_grad = False

    def encode_image(
        self,
        images: torch.Tensor,
        normalize: bool = True
    ) -> torch.Tensor:
        """
        Encode images to feature vectors.

        Args:
            images: Image tensor of shape (B, C, H, W)
            normalize: Whether to L2-normalize features

        Returns:
            Image features of shape (B, projection_dim)
        """
        # Get image features from CLIP
        # get_image_features returns a BaseModelOutputWithPooling object
        outputs = self.clip_model.vision_model(pixel_values=images)
        image_features = outputs.pooler_output  # Use pooled output

        # Project to common dimension if needed
        if hasattr(self.clip_model, 'visual_projection'):
            image_features = self.clip_model.visual_projection(image_features)

        # Normalize if requested
        if normalize:
            image_features = F.normalize(image_features, dim=-1)

        return image_features

    def encode_text(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        normalize: bool = True
    ) -> torch.Tensor:
        """
        Encode text to feature vectors.

        Args:
            input_ids: Input token IDs of shape (B, L)
            attention_mask: Attention mask of shape (B, L)
            normalize: Whether to L2-normalize features

        Returns:
            Text features of shape (B, projection_dim)
        """
        # Get text features from CLIP
        # get_text_features returns a BaseModelOutputWithPooling object
        outputs = self.clip_model.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        text_features = outputs.pooler_output  # Use pooled output

        # Project to common dimension if needed
        if hasattr(self.clip_model, 'text_projection'):
            text_features = self.clip_model.text_projection(text_features)

        # Normalize if requested
        if normalize:
            text_features = F.normalize(text_features, dim=-1)

        return text_features

    def forward(
        self,
        images: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        normalize: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass - encode both images and text.

        Args:
            images: Image tensor of shape (B, C, H, W)
            input_ids: Input token IDs of shape (B, L)
            attention_mask: Attention mask of shape (B, L)
            normalize: Whether to L2-normalize features

        Returns:
            Tuple of (image_features, text_features), each of shape (B, projection_dim)
        """
        image_features = self.encode_image(images, normalize=normalize)
        text_features = self.encode_text(input_ids, attention_mask, normalize=normalize)

        return image_features, text_features

    def process_images(
        self,
        images: List
    ) -> torch.Tensor:
        """
        Process a list of PIL images to tensors.

        Args:
            images: List of PIL Images

        Returns:
            Processed image tensor
        """
        device = next(self.parameters()).device
        return preprocess_images_on_gpu(images, device)

    def process_text(
        self,
        texts: List[str]
    ) -> Dict[str, torch.Tensor]:
        """
        Process a list of text strings to model inputs.

        Args:
            texts: List of text strings

        Returns:
            Dictionary with input_ids and attention_mask
        """
        return self.processor(text=texts, return_tensors="pt", padding=True, truncation=True, max_length=77)

    def get_similarity(
        self,
        image_features: torch.Tensor,
        text_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute similarity matrix between image and text features.

        Args:
            image_features: Image features of shape (N_images, D)
            text_features: Text features of shape (N_texts, D)

        Returns:
            Similarity matrix of shape (N_images, N_texts)
        """
        # Assume features are already normalized
        return image_features @ text_features.T

    def trainable_parameters(self) -> List[nn.Parameter]:
        """
        Get list of trainable parameters.

        Returns:
            List of parameters with requires_grad=True
        """
        return [p for p in self.parameters() if p.requires_grad]

    def num_trainable_parameters(self) -> int:
        """
        Count trainable parameters.

        Returns:
            Number of trainable parameters
        """
        return sum(p.numel() for p in self.trainable_parameters())

    def save(self, path: str) -> None:
        """
        Save model state.

        Args:
            path: Path to save checkpoint
        """
        state = {
            "model_state_dict": self.state_dict(),
            "freeze_image": self.freeze_image,
            "freeze_text": self.freeze_text,
        }
        torch.save(state, path)
        logger.info(f"Model saved to: {path}")

    def load(self, path: str, strict: bool = True) -> None:
        """
        Load model state.

        Args:
            path: Path to checkpoint
            strict: Whether to strictly enforce state dict matching
        """
        state = torch.load(path, map_location="cpu", weights_only=False)
        self.load_state_dict(state["model_state_dict"], strict=strict)
        logger.info(f"Model loaded from: {path}")


if __name__ == "__main__":
    # Test model
    import config
    from utils.logger import setup_logger
    from utils.seed import set_seed

    # Setup
    setup_logger("clip_baseline", config.LOG_DIR / "model_test.log")
    set_seed(config.SEED)

    # Create model
    model = CLIPFineTuneBaseline(
        freeze_image=config.CLIP_BASELINE_FREEZE_IMAGE,
        freeze_text=config.CLIP_BASELINE_FREEZE_TEXT
    )

    print(f"Model created successfully")
    print(f"Trainable parameters: {model.num_trainable_parameters():,}")

    # Test forward pass with dummy data
    batch_size = 2
    seq_length = 77  # CLIP default max length

    # Create dummy inputs
    dummy_images = [torch.rand(3, 224, 224) for _ in range(batch_size)]
    dummy_texts = ["A photo of a cat", "A photo of a dog"]

    # Process inputs
    image_tensor = model.process_images(dummy_images)
    text_inputs = model.process_text(dummy_texts)

    print(f"\nProcessed image tensor shape: {image_tensor.shape}")
    print(f"Processed text input_ids shape: {text_inputs['input_ids'].shape}")

    # Forward pass
    with torch.no_grad():
        image_features, text_features = model(
            images=image_tensor,
            input_ids=text_inputs["input_ids"],
            attention_mask=text_inputs["attention_mask"]
        )

    print(f"\nImage features shape: {image_features.shape}")
    print(f"Text features shape: {text_features.shape}")

    # Test similarity
    similarity = model.get_similarity(image_features, text_features)
    print(f"\nSimilarity matrix shape: {similarity.shape}")
    print(f"Similarity matrix:\n{similarity}")
