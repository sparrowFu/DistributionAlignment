"""
GaussianImageDistribution - CLIP-AST Model

This module implements the CLIP-AST (Adaptive Selective Tuning) method
adapted for this project. CLIP-AST automatically selects critical parameters
in CLIP for fine-tuning based on gradient magnitude, achieving parameter
efficiency without extra parameter overhead.

Reference:
    CLIP-AST (CVPR 2025), pages 4280-4290
    https://cvpr.thecvf.com/virtual/2025/poster/34663

Architecture:
    CLIP ViT-L/14 (selectively unfrozen)
        |
    After 1 epoch warmup:
        1. Compute gradient magnitudes for all CLIP parameters
        2. Select top select_ratio parameters to keep trainable
        3. Freeze remaining parameters
        4. Continue training with selected params + classifier head

    No extra parameter overhead beyond the base CLIP model.
"""

from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPModel, CLIPProcessor

import config
from utils.logger import get_logger


logger = get_logger("clip_ast_model")


class CLIPASTModel(nn.Module):
    """
    CLIP-AST Model: Adaptive Selective Tuning for CLIP.

    Selectively fine-tunes a small fraction of CLIP parameters
    based on gradient magnitude, without extra parameter overhead.

    Training procedure:
        1. Warmup: all CLIP params trainable with small LR
        2. After warmup: call select_parameters() to select top-K params
        3. Continue training with only selected params
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        select_ratio: float = 0.05,
    ):
        """
        Initialize CLIP-AST model.

        Args:
            model_path: Path to local CLIP model directory
            select_ratio: Fraction of CLIP parameters to select for fine-tuning
        """
        super().__init__()

        self.model_path = model_path or str(config.CLIP_VIT_L_14_PATH)
        self.select_ratio = select_ratio
        self.feature_dim = 768  # CLIP projection_dim
        self._param_selected = False

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

        # Initially freeze all CLIP params; will be selectively unfrozen during training
        self._freeze_clip()
        logger.info("CLIP parameters initially frozen (will be selectively unfrozen)")

        logger.info(
            f"CLIP-AST model initialized: select_ratio={select_ratio}"
        )

    def _freeze_clip(self) -> None:
        """Freeze all CLIP model parameters."""
        for param in self.clip_model.parameters():
            param.requires_grad = False

    def unfreeze_all_clip(self) -> None:
        """Unfreeze all CLIP parameters for warmup phase."""
        for param in self.clip_model.parameters():
            param.requires_grad = True
        logger.info("All CLIP parameters unfrozen for warmup")

    def select_parameters(self) -> None:
        """
        Select top-K CLIP parameters to keep trainable based on gradient magnitude.

        Should be called after a warmup forward-backward pass so gradients
        are available. Parameters with the largest gradient magnitudes are
        considered most important for the downstream task.

        After selection, non-selected parameters are frozen.
        """
        # Collect gradient magnitudes for all CLIP parameters
        param_scores = []
        for name, param in self.clip_model.named_parameters():
            if param.grad is not None:
                score = param.grad.data.abs().mean().item()
                param_scores.append((name, param, score))
            else:
                param_scores.append((name, param, 0.0))

        if not param_scores:
            logger.warning("No gradients found for parameter selection")
            return

        # Sort by score descending
        param_scores.sort(key=lambda x: x[2], reverse=True)

        # Select top-K parameter tensors
        k = max(1, int(len(param_scores) * self.select_ratio))

        # Freeze non-selected params
        selected_names = set(name for name, _, _ in param_scores[:k])
        for name, param in self.clip_model.named_parameters():
            if name not in selected_names:
                param.requires_grad = False

        self._param_selected = True

        # Log selection statistics
        selected_count = sum(
            p.numel() for n, p, _ in param_scores[:k]
        )
        total_count = sum(
            p.numel() for _, p, _ in param_scores
        )
        logger.info(
            f"CLIP-AST: Selected {k}/{len(param_scores)} parameter tensors "
            f"({selected_count:,}/{total_count:,} parameters, "
            f"{selected_count / max(total_count, 1) * 100:.2f}%)"
        )

    def encode_image(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Encode image through CLIP vision encoder.

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
        Encode text through CLIP text encoder.

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
                - img_features: Image features (B, 768)
                - text_features: Text features (B, 768)
        """
        img_feat = self.encode_image(pixel_values)
        text_feat = self.encode_text(input_ids, attention_mask)

        return {
            "img_features": img_feat,
            "text_features": text_feat,
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
            "select_ratio": self.select_ratio,
            "param_selected": self._param_selected,
        }
        torch.save(state, path)
        logger.info(f"Model saved to: {path}")

    def load(self, path: str, strict: bool = True) -> None:
        """Load model state."""
        state = torch.load(path, map_location="cpu", weights_only=False)
        self.load_state_dict(state["model_state_dict"], strict=strict)
        logger.info(f"Model loaded from: {path}")
        if "select_ratio" in state:
            self.select_ratio = state["select_ratio"]
        if "param_selected" in state:
            self._param_selected = state["param_selected"]


if __name__ == "__main__":
    from utils.logger import setup_logger
    from utils.seed import set_seed

    setup_logger("clip_ast_model", config.LOG_DIR / "clip_ast_model_test.log")
    set_seed(config.SEED)

    model = CLIPASTModel(select_ratio=0.05)

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
