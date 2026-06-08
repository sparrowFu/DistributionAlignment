"""
GaussianImageDistribution - Freeze-Align Model

This module implements the Freeze-Align method adapted for this project.
Freeze-Align connects frozen unimodal encoders with trainable projectors
and preserves neighborhood structure via STRUCTURE regularization.

Reference:
    Freeze-Align (CVPR 2025)
    https://github.com/mayug/freeze-align

Architecture:
    Frozen CLIP ViT-L/14
        |                    |
    Image Encoder (frozen)  Text Encoder (frozen)
        |                    |
    img_feat (768)       text_feat (768)
        |                    |
    img_projector        text_projector
        |                    |
    proj_img (768)       proj_text (768)

    STRUCTURE loss preserves multiscale neighborhood structure
    between original CLIP features and projected features.
"""

from typing import Dict, List, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPModel, CLIPProcessor

import config
from utils.logger import get_logger


logger = get_logger("freeze_align_model")


class FreezeAlignModel(nn.Module):
    """
    Freeze-Align Model: frozen CLIP encoders + trainable projectors.

    The projectors map CLIP image/text features to an aligned space,
    with STRUCTURE regularization preserving neighborhood structure
    from the original CLIP feature space.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        proj_dim: int = 256,
        dropout_rate: float = 0.1,
    ):
        """
        Initialize Freeze-Align model.

        Args:
            model_path: Path to local CLIP model directory
            proj_dim: Bottleneck dimension for projectors
            dropout_rate: Dropout rate for projectors
        """
        super().__init__()

        self.model_path = model_path or str(config.CLIP_VIT_L_14_PATH)
        self.proj_dim = proj_dim
        self.dropout_rate = dropout_rate
        self.feature_dim = 768  # CLIP projection_dim

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
        self._freeze_clip()
        logger.info("CLIP parameters frozen")

        # Image projector: 768 → proj_dim → 768
        self.img_projector = nn.Sequential(
            nn.Linear(self.feature_dim, proj_dim),
            nn.Dropout(dropout_rate),
            nn.ReLU(),
            nn.Linear(proj_dim, self.feature_dim),
        )

        # Text projector: 768 → proj_dim → 768
        self.text_projector = nn.Sequential(
            nn.Linear(self.feature_dim, proj_dim),
            nn.Dropout(dropout_rate),
            nn.ReLU(),
            nn.Linear(proj_dim, self.feature_dim),
        )

        # Initialize projectors
        self._init_projectors()

        # Storage for STRUCTURE loss (computed during forward)
        self._last_structure_loss = None

        logger.info(
            f"Freeze-Align model initialized: proj_dim={proj_dim}, "
            f"trainable params={self.num_trainable_parameters():,}"
        )

    def _freeze_clip(self) -> None:
        """Freeze all CLIP model parameters."""
        for param in self.clip_model.parameters():
            param.requires_grad = False

    def _init_projectors(self) -> None:
        """Initialize projector weights with Xavier initialization."""
        for projector in [self.img_projector, self.text_projector]:
            for layer in projector:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
                    if layer.bias is not None:
                        nn.init.zeros_(layer.bias)

    def _get_raw_image_features(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Get CLIP image features (before projector).

        Args:
            pixel_values: Image tensor (B, 3, 224, 224)

        Returns:
            Image features (B, 768)
        """
        vision_outputs = self.clip_model.vision_model(pixel_values=pixel_values)
        img_feat = vision_outputs.pooler_output  # (B, 1024)
        img_feat = self.clip_model.visual_projection(img_feat)  # (B, 768)
        return img_feat

    def _get_raw_text_features(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Get CLIP text features (before projector).

        Args:
            input_ids: Token IDs (B, seq_len)
            attention_mask: Attention mask (B, seq_len)

        Returns:
            Text features (B, 768)
        """
        text_outputs = self.clip_model.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        text_feat = text_outputs.pooler_output  # (B, 768)
        text_feat = self.clip_model.text_projection(text_feat)  # (B, 768)
        return text_feat

    def encode_image(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Encode image through CLIP + projector.

        Args:
            pixel_values: Image tensor (B, 3, 224, 224)

        Returns:
            Projected image features (B, 768)
        """
        raw_feat = self._get_raw_image_features(pixel_values)
        projected = self.img_projector(raw_feat)
        return projected

    def encode_text(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Encode text through CLIP + projector.

        Args:
            input_ids: Token IDs (B, seq_len)
            attention_mask: Attention mask (B, seq_len)

        Returns:
            Projected text features (B, 768)
        """
        raw_feat = self._get_raw_text_features(input_ids, attention_mask)
        projected = self.text_projector(raw_feat)
        return projected

    def compute_structure_loss(
        self,
        original_feats: torch.Tensor,
        projected_feats: torch.Tensor,
        k: int = 5,
        temperature: float = 0.1,
    ) -> torch.Tensor:
        """
        STRUCTURE regularization loss.

        Preserves neighborhood structure between original and projected
        feature spaces by minimizing KL divergence of soft k-NN distributions.

        Args:
            original_feats: Features before projection (B, D)
            projected_feats: Features after projection (B, D)
            k: Number of nearest neighbors
            temperature: Temperature for soft distribution

        Returns:
            STRUCTURE loss scalar
        """
        B = original_feats.size(0)
        if B < k + 1:
            return torch.tensor(0.0, device=original_feats.device)

        # Pairwise distance in original space
        orig_dist = torch.cdist(original_feats, original_feats)  # (B, B)
        orig_prob = F.softmax(-orig_dist / temperature, dim=1)  # (B, B)

        # Pairwise distance in projected space
        proj_dist = torch.cdist(projected_feats, projected_feats)  # (B, B)
        proj_log_prob = F.log_softmax(-proj_dist / temperature, dim=1)  # (B, B)

        # KL divergence: KL(orig || proj)
        structure_loss = F.kl_div(proj_log_prob, orig_prob, reduction="batchmean")

        return structure_loss

    def forward(
        self,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.

        Args:
            pixel_values: Image tensor (B, 3, 224, 224)
            input_ids: Token IDs (B, seq_len)
            attention_mask: Attention mask (B, seq_len)

        Returns:
            Dictionary containing:
                - raw_img_features: CLIP image features (B, 768)
                - raw_text_features: CLIP text features (B, 768)
                - proj_img_features: Projected image features (B, 768)
                - proj_text_features: Projected text features (B, 768)
        """
        raw_img = self._get_raw_image_features(pixel_values)
        raw_text = self._get_raw_text_features(input_ids, attention_mask)

        proj_img = self.img_projector(raw_img)
        proj_text = self.text_projector(raw_text)

        # Compute STRUCTURE loss and store for training
        self._last_structure_loss = (
            self.compute_structure_loss(raw_img, proj_img)
            + self.compute_structure_loss(raw_text, proj_text)
        ) / 2.0

        return {
            "raw_img_features": raw_img,
            "raw_text_features": raw_text,
            "proj_img_features": proj_img,
            "proj_text_features": proj_text,
        }

    @property
    def last_extra_loss(self) -> Optional[torch.Tensor]:
        """Get the last computed STRUCTURE loss."""
        return self._last_structure_loss

    def process_images(self, images: List) -> torch.Tensor:
        """Process PIL images to tensors using CLIP processor."""
        return self.processor(images=images, return_tensors="pt")["pixel_values"]

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
        """Get list of trainable parameters."""
        return [p for p in self.parameters() if p.requires_grad]

    def num_trainable_parameters(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.trainable_parameters())

    def save(self, path: str) -> None:
        """Save model state."""
        state = {
            "model_state_dict": self.state_dict(),
            "proj_dim": self.proj_dim,
            "dropout_rate": self.dropout_rate,
        }
        torch.save(state, path)
        logger.info(f"Model saved to: {path}")

    def load(self, path: str, strict: bool = True) -> None:
        """Load model state."""
        state = torch.load(path, map_location="cpu", weights_only=False)
        self.load_state_dict(state["model_state_dict"], strict=strict)
        logger.info(f"Model loaded from: {path}")
        if "proj_dim" in state:
            self.proj_dim = state["proj_dim"]
        if "dropout_rate" in state:
            self.dropout_rate = state["dropout_rate"]


if __name__ == "__main__":
    from utils.logger import setup_logger
    from utils.seed import set_seed

    setup_logger("freeze_align_model", config.LOG_DIR / "freeze_align_model_test.log")
    set_seed(config.SEED)

    model = FreezeAlignModel(
        proj_dim=256,
        dropout_rate=0.1,
    )

    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Trainable parameters: {model.num_trainable_parameters():,}")

    batch_size = 2
    dummy_images = torch.randn(batch_size, 3, 224, 224)
    dummy_input_ids = torch.randint(0, 49408, (batch_size, 77))
    dummy_attention_mask = torch.ones(batch_size, 77, dtype=torch.long)

    with torch.no_grad():
        outputs = model(dummy_images, dummy_input_ids, dummy_attention_mask)

    print(f"\nOutput shapes:")
    for key, value in outputs.items():
        print(f"  {key}: {value.shape}")
    print(f"  structure_loss: {model.last_extra_loss}")
