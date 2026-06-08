"""
GaussianImageDistribution - FATE Model

This module implements the FATE (Feature-Adapted Parameter-Efficient Tuning)
method adapted for this project. FATE extracts vision features from the last
layer of the vision encoder, projects them via a small 2-layer FC network,
and injects them into text features as a perturbation.

Reference:
    FATE (AAAI 2025)
    https://ojs.aaai.org/index.php/AAAI/article/view/32975

Architecture:
    Frozen CLIP ViT-L/14
        |
    Image Encoder (frozen)  Text Encoder (frozen)
        |                        |
    img_feat (768)          text_feat (768)
        |                        |
    F(FV) = f1(ReLU(f2(FV)))    |
        |                        |
        +--- α * F(FV) ----------+  (vision perturbation)
                    |
            adapted_text_feat (768)

    Only the projector (f1, f2) is trainable.
    α = 0.001 (scaling factor, from original paper).
    ~15K extra parameters with bottleneck_dim=64.
"""

from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPModel, CLIPProcessor

import config
from utils.logger import get_logger


logger = get_logger("fate_model")


class FATEModel(nn.Module):
    """
    FATE Model: Feature-Adapted Parameter-Efficient Tuning.

    Projects vision features into text feature space via a small
    2-layer network and adds them as a perturbation to text features.

    Key equations (from paper):
        F(FV) = f1(ReLU(f2(FV)))      (Eq. 9)
        adapted_text = text_feat + α * F(FV)
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        bottleneck_dim: int = 64,
        alpha: float = 0.001,
    ):
        """
        Initialize FATE model.

        Args:
            model_path: Path to local CLIP model directory
            bottleneck_dim: Bottleneck dimension for projector
            alpha: Scaling factor for vision perturbation
        """
        super().__init__()

        self.model_path = model_path or str(config.CLIP_VIT_L_14_PATH)
        self.bottleneck_dim = bottleneck_dim
        self.alpha = alpha
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

        # Vision-to-text projector: F(FV) = f1(ReLU(f2(FV)))
        # f2: Linear(768, bottleneck_dim)
        # f1: Linear(bottleneck_dim, 768)
        self.projector = nn.Sequential(
            nn.Linear(self.feature_dim, bottleneck_dim),   # f2
            nn.ReLU(),
            nn.Linear(bottleneck_dim, self.feature_dim),   # f1
        )

        # Initialize projector
        self._init_projector()

        logger.info(
            f"FATE model initialized: bottleneck_dim={bottleneck_dim}, "
            f"alpha={alpha}, trainable params={self.num_trainable_parameters():,}"
        )

    def _freeze_clip(self) -> None:
        """Freeze all CLIP model parameters."""
        for param in self.clip_model.parameters():
            param.requires_grad = False

    def _init_projector(self) -> None:
        """Initialize projector weights with Xavier initialization."""
        for layer in self.projector:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)

    def encode_image(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Get CLIP image features.

        Args:
            pixel_values: Image tensor (B, 3, 224, 224)

        Returns:
            Image features (B, 768)
        """
        vision_outputs = self.clip_model.vision_model(pixel_values=pixel_values)
        img_feat = vision_outputs.pooler_output  # (B, 1024)
        img_feat = self.clip_model.visual_projection(img_feat)  # (B, 768)
        return img_feat

    def encode_text(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Get CLIP text features (without vision perturbation).

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

    def compute_vision_perturbation(self, img_feat: torch.Tensor) -> torch.Tensor:
        """
        Compute vision-based perturbation for text features.

        F(FV) = f1(ReLU(f2(FV)))

        Args:
            img_feat: Image features (B, 768)

        Returns:
            Vision perturbation (B, 768)
        """
        return self.projector(img_feat)

    def encode_text_adapted(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        img_feat: torch.Tensor,
    ) -> torch.Tensor:
        """
        Get adapted text features with vision perturbation.

        adapted_text = text_feat + α * F(FV)

        Args:
            input_ids: Token IDs (B, seq_len)
            attention_mask: Attention mask (B, seq_len)
            img_feat: Image features for computing perturbation (B, 768)

        Returns:
            Adapted text features (B, 768)
        """
        text_feat = self.encode_text(input_ids, attention_mask)
        vision_pert = self.compute_vision_perturbation(img_feat)
        adapted_text = text_feat + self.alpha * vision_pert
        return adapted_text

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
                - img_features: CLIP image features (B, 768)
                - text_features: CLIP text features, raw (B, 768)
                - text_features_adapted: Adapted text features (B, 768)
                - vision_perturbation: Vision perturbation (B, 768)
        """
        img_feat = self.encode_image(pixel_values)
        text_feat = self.encode_text(input_ids, attention_mask)
        vision_pert = self.compute_vision_perturbation(img_feat)
        text_adapted = text_feat + self.alpha * vision_pert

        return {
            "img_features": img_feat,
            "text_features": text_feat,
            "text_features_adapted": text_adapted,
            "vision_perturbation": vision_pert,
        }

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
            "bottleneck_dim": self.bottleneck_dim,
            "alpha": self.alpha,
        }
        torch.save(state, path)
        logger.info(f"Model saved to: {path}")

    def load(self, path: str, strict: bool = True) -> None:
        """Load model state."""
        state = torch.load(path, map_location="cpu", weights_only=False)
        self.load_state_dict(state["model_state_dict"], strict=strict)
        logger.info(f"Model loaded from: {path}")
        if "bottleneck_dim" in state:
            self.bottleneck_dim = state["bottleneck_dim"]
        if "alpha" in state:
            self.alpha = state["alpha"]


if __name__ == "__main__":
    from utils.logger import setup_logger
    from utils.seed import set_seed

    setup_logger("fate_model", config.LOG_DIR / "fate_model_test.log")
    set_seed(config.SEED)

    model = FATEModel(
        bottleneck_dim=64,
        alpha=0.001,
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

    # Verify perturbation effect
    diff = (outputs["text_features_adapted"] - outputs["text_features"]).abs().mean()
    print(f"\nMean perturbation magnitude: {diff:.6f}")
    print(f"Expected ~{model.alpha * outputs['vision_perturbation'].abs().mean():.6f}")
