"""
GaussianImageDistribution - VQA Model

This module implements a VQA (Visual Question Answering) model that wraps
an existing base model as a feature extractor and adds a trainable
classification head on top.

Supported base model types:
    - "dist_align": DistributionAlignmentModel (frozen CLIP + MLP distribution heads)
    - "clip_baseline": CLIPFineTuneBaseline (frozen CLIP)
    - "freeze_align": FreezeAlignModel (frozen CLIP + trainable projectors)
    - "fate": FATEModel (frozen CLIP + vision→text projector)
    - "clip_ast": CLIPASTModel (selectively fine-tuned CLIP params)
    - "clip_zero_shot": CLIPZeroShotVQA (no training, similarity-based)

Architecture:
    Input: PIL Image + Question Text
        |                    |
    Base Model (frozen)  CLIP Text Encoder (frozen)
        |                    |
    img_feat (768)       question_feat (768)
        |                    |
        +------ concat ------+
                 |
          FC (1536 -> hidden_dim)
                 |
            ReLU + Dropout
                 |
          FC (hidden_dim -> num_classes)
                 |
            CrossEntropy Loss
"""

from typing import Dict, List, Optional

import torch
import torch.nn as nn

import config
from utils.logger import get_logger


logger = get_logger("vqa_model")

# All supported model types for VQA training (excludes clip_zero_shot)
TRAINABLE_MODEL_TYPES = ["dist_align", "clip_baseline", "freeze_align", "fate", "clip_ast"]

# All supported model types (includes clip_zero_shot)
ALL_MODEL_TYPES = TRAINABLE_MODEL_TYPES + ["clip_zero_shot"]


class VQAModel(nn.Module):
    """
    VQA Model with frozen backbone and trainable classification head.

    Supports multiple base model types, each providing 768-dim image and
    text features that are concatenated and fed to a classification head.

    For "fate" model type, text features are adapted with vision perturbation
    before concatenation.
    """

    def __init__(
        self,
        model_type: str = "dist_align",
        num_classes: int = 430,
        hidden_dim: int = 512,
        dropout: float = 0.1,
        answer_vocab: Optional[Dict[str, int]] = None,
        base_ckpt_path: Optional[str] = None,
        device: str = "cpu",
    ):
        """
        Initialize VQA model.

        Args:
            model_type: Base model type (see ALL_MODEL_TYPES)
            num_classes: Number of answer classes
            hidden_dim: Hidden dimension for the classification head
            dropout: Dropout rate for classification head
            answer_vocab: Answer → index mapping (saved with checkpoint)
            base_ckpt_path: Path to pre-trained base model checkpoint
            device: Device to load base model onto
        """
        super().__init__()

        if model_type not in ALL_MODEL_TYPES:
            raise ValueError(
                f"Unknown model_type: {model_type}. "
                f"Use one of: {ALL_MODEL_TYPES}"
            )

        self.model_type = model_type
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        self.dropout_rate = dropout
        self.answer_vocab = answer_vocab or {}
        self.feature_dim = 768  # CLIP ViT-Large projection dimension

        # Load base model
        self._load_base_model(model_type, base_ckpt_path, device)

        # Freeze base model (except clip_ast which is selectively unfrozen later)
        if model_type != "clip_ast":
            self._freeze_base()

        # Classification head: concat(img_feat, text_feat) -> logits
        self.classifier = nn.Sequential(
            nn.Linear(self.feature_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

        # Initialize classifier weights
        self._init_classifier()

        logger.info(
            f"VQAModel initialized: type={model_type}, "
            f"num_classes={num_classes}, hidden_dim={hidden_dim}"
        )

    def _load_base_model(
        self,
        model_type: str,
        ckpt_path: Optional[str],
        device: str,
    ):
        """Load the base model and optionally restore from checkpoint."""
        if model_type == "dist_align":
            from models.dist_align_model import DistributionAlignmentModel
            self.base_model = DistributionAlignmentModel(
                freeze_clip=True,
                distribution_merging=config.DIST_ALIGN_DISTRIBUTION_MERGING,
                dropout_rate=config.DIST_ALIGN_DROPOUT_RATE,
            )
            if ckpt_path:
                logger.info(f"Loading dist_align checkpoint: {ckpt_path}")
                self.base_model.load(ckpt_path)

        elif model_type == "clip_baseline":
            from models.clip_baseline import CLIPFineTuneBaseline
            self.base_model = CLIPFineTuneBaseline(
                freeze_image=True,
                freeze_text=True,
            )
            if ckpt_path:
                logger.info(f"Loading clip_baseline checkpoint: {ckpt_path}")
                self.base_model.load(ckpt_path)

        elif model_type == "freeze_align":
            from models.freeze_align_model import FreezeAlignModel
            self.base_model = FreezeAlignModel(
                proj_dim=config.FREEZE_ALIGN_PROJ_DIM,
                dropout_rate=config.FREEZE_ALIGN_DROPOUT_RATE,
            )
            if ckpt_path:
                logger.info(f"Loading freeze_align checkpoint: {ckpt_path}")
                self.base_model.load(ckpt_path)

        elif model_type == "fate":
            from models.fate_model import FATEModel
            self.base_model = FATEModel(
                bottleneck_dim=config.FATE_BOTTLENECK_DIM,
                alpha=config.FATE_ALPHA,
            )
            if ckpt_path:
                logger.info(f"Loading fate checkpoint: {ckpt_path}")
                self.base_model.load(ckpt_path)

        elif model_type == "clip_ast":
            from models.clip_ast_model import CLIPASTModel
            self.base_model = CLIPASTModel(
                select_ratio=config.CLIP_AST_SELECT_RATIO,
            )
            if ckpt_path:
                logger.info(f"Loading clip_ast checkpoint: {ckpt_path}")
                self.base_model.load(ckpt_path)

        self.base_model = self.base_model.to(device)

    def _freeze_base(self):
        """Freeze all parameters in the base model."""
        for param in self.base_model.parameters():
            param.requires_grad = False
        logger.info("Base model parameters frozen")

    def _init_classifier(self):
        """Initialize classifier weights with Xavier initialization."""
        for layer in self.classifier:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)

    def extract_image_features(
        self,
        pixel_values: torch.Tensor,
    ) -> torch.Tensor:
        """
        Extract image features from the base model.

        Args:
            pixel_values: Image tensor (B, 3, 224, 224)

        Returns:
            Image features (B, 768)
        """
        if self.model_type in ("dist_align", "clip_baseline"):
            # Use CLIP vision model directly
            vision_outputs = self.base_model.clip_model.vision_model(
                pixel_values=pixel_values
            )
            img_features = vision_outputs.pooler_output
            img_features = self.base_model.clip_model.visual_projection(img_features)

        elif self.model_type == "freeze_align":
            # Freeze-Align: use projected features
            img_features = self.base_model.encode_image(pixel_values)

        elif self.model_type == "fate":
            # FATE: use raw CLIP image features
            img_features = self.base_model.encode_image(pixel_values)

        elif self.model_type == "clip_ast":
            # CLIP-AST: use fine-tuned CLIP features
            img_features = self.base_model.encode_image(pixel_values)

        else:
            raise ValueError(f"extract_image_features not supported for {self.model_type}")

        return img_features

    def extract_text_features(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        img_feat: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Extract text features using the base model.

        Args:
            input_ids: Token IDs (B, seq_len)
            attention_mask: Attention mask (B, seq_len)
            img_feat: Image features (needed for FATE model type)

        Returns:
            Text features (B, 768)
        """
        if self.model_type in ("dist_align", "clip_baseline"):
            text_outputs = self.base_model.clip_model.text_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
            text_features = text_outputs.pooler_output
            text_features = self.base_model.clip_model.text_projection(text_features)

        elif self.model_type == "freeze_align":
            # Freeze-Align: use projected features
            text_features = self.base_model.encode_text(input_ids, attention_mask)

        elif self.model_type == "fate":
            # FATE: use adapted text features with vision perturbation
            if img_feat is None:
                raise ValueError("FATE model requires img_feat for text feature extraction")
            text_features = self.base_model.encode_text_adapted(
                input_ids, attention_mask, img_feat
            )

        elif self.model_type == "clip_ast":
            # CLIP-AST: use fine-tuned CLIP features
            text_features = self.base_model.encode_text(input_ids, attention_mask)

        else:
            raise ValueError(f"extract_text_features not supported for {self.model_type}")

        return text_features

    @property
    def extra_loss(self) -> Optional[torch.Tensor]:
        """Get extra loss from base model (e.g., STRUCTURE regularization for Freeze-Align)."""
        if hasattr(self.base_model, "last_extra_loss"):
            return self.base_model.last_extra_loss
        return None

    def process_images(self, images: List) -> torch.Tensor:
        """Process PIL images to tensors using CLIP processor."""
        return self.base_model.processor(images=images, return_tensors="pt")["pixel_values"]

    def process_text(self, texts: List[str]) -> Dict[str, torch.Tensor]:
        """Process text strings to token IDs using CLIP processor."""
        return self.base_model.processor(
            text=texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=77,
        )

    def forward(
        self,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            pixel_values: Image tensor (B, 3, 224, 224)
            input_ids: Token IDs (B, seq_len)
            attention_mask: Attention mask (B, seq_len)

        Returns:
            Logits tensor (B, num_classes)
        """
        img_feat = self.extract_image_features(pixel_values)  # (B, 768)
        text_feat = self.extract_text_features(
            input_ids, attention_mask, img_feat=img_feat
        )  # (B, 768)

        combined = torch.cat([img_feat, text_feat], dim=1)  # (B, 1536)
        logits = self.classifier(combined)  # (B, num_classes)

        return logits

    def trainable_parameters(self) -> List[nn.Parameter]:
        """Get list of trainable parameters."""
        return [p for p in self.parameters() if p.requires_grad]

    def num_trainable_parameters(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.trainable_parameters())

    def save(self, path: str) -> None:
        """
        Save model state (classifier + base model adapters).

        For clip_ast, also saves the fine-tuned CLIP parameters.
        For other types, saves only the classification head and adapter weights.
        """
        state = {
            "classifier_state_dict": self.classifier.state_dict(),
            "model_type": self.model_type,
            "num_classes": self.num_classes,
            "hidden_dim": self.hidden_dim,
            "dropout_rate": self.dropout_rate,
            "answer_vocab": self.answer_vocab,
        }

        # For clip_ast, also save the fine-tuned CLIP parameters
        if self.model_type == "clip_ast":
            state["clip_state_dict"] = self.base_model.clip_model.state_dict()

        # For freeze_align and fate, save adapter weights
        if self.model_type in ("freeze_align", "fate"):
            adapter_state = {
                k: v for k, v in self.base_model.state_dict().items()
                if not k.startswith("clip_model.")
            }
            state["adapter_state_dict"] = adapter_state

        torch.save(state, path)
        logger.info(f"VQA model saved to: {path}")

    def load_classifier(self, path: str) -> None:
        """
        Load model state (classifier + base model adapters).

        Args:
            path: Path to VQA checkpoint
        """
        state = torch.load(path, map_location="cpu", weights_only=False)
        self.classifier.load_state_dict(state["classifier_state_dict"])

        # Restore adapter weights if present
        if "adapter_state_dict" in state:
            self.base_model.load_state_dict(state["adapter_state_dict"], strict=False)
            logger.info("Adapter weights restored")

        # Restore CLIP weights for clip_ast
        if "clip_state_dict" in state:
            self.base_model.clip_model.load_state_dict(state["clip_state_dict"])
            logger.info("CLIP-AST fine-tuned CLIP weights restored")

        if "answer_vocab" in state:
            self.answer_vocab = state["answer_vocab"]
        if "num_classes" in state:
            self.num_classes = state["num_classes"]

        logger.info(f"VQA model loaded from: {path}")


if __name__ == "__main__":
    from utils.logger import setup_logger
    from utils.seed import set_seed

    setup_logger("vqa_model", config.LOG_DIR / "vqa_model_test.log")
    set_seed(config.SEED)

    # Test with dist_align base
    print("Testing VQAModel with dist_align backbone...")
    model = VQAModel(
        model_type="dist_align",
        num_classes=430,
        hidden_dim=config.VQA_HIDDEN_DIM,
        dropout=config.VQA_DROPOUT,
    )

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = model.num_trainable_parameters()
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Frozen parameters: {total_params - trainable_params:,}")

    # Test forward pass with dummy data
    batch_size = 2
    dummy_images = torch.randn(batch_size, 3, 224, 224)
    dummy_input_ids = torch.randint(0, 49408, (batch_size, 77))
    dummy_attention_mask = torch.ones(batch_size, 77, dtype=torch.long)

    with torch.no_grad():
        logits = model(dummy_images, dummy_input_ids, dummy_attention_mask)

    print(f"Output logits shape: {logits.shape}")
    assert logits.shape == (batch_size, 430), f"Expected (2, 430), got {logits.shape}"
    print("Forward pass test passed!")
