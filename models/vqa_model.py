"""
GaussianImageDistribution - VQA Model

This module implements a VQA (Visual Question Answering) model that wraps
an existing base model as a feature extractor and adds a trainable
classification head on top.

Supported base model types (from experiment plan):
    B1 "clip_zero_shot": CLIPZeroShotVQA (no training, similarity-based)
    B2 "clip_baseline": CLIPFineTuneBaseline (frozen CLIP)
    B3 "prolip": ProLIPModel (probabilistic, implicit σ)
    B4 "grove": GroVEModel (GP-based posterior variance)
    Ours "dist_align": DistributionAlignmentModel (MSDA, σ²=caption spread)

Architecture (all models, unified 768-dim features):
    Input: PIL Image + Question Text
        |                    |
    Feature Extractor    Feature Extractor
        |                    |
    img_feat (768)       text_feat (768)
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

For dist_align, the 768-dim feature is obtained by sampling from the
learned Gaussian distribution (during training) or using the distribution
mean (during evaluation):

    Training:  z = mu + eps * sigma    (stochastic, acts as regularization)
    Eval:      feat = mu               (deterministic)
    Eval+MC:   average multiple z samples (uncertainty-aware prediction)
"""

from typing import Dict, List, Optional

import torch
import torch.nn as nn

import config
from utils.logger import get_logger
from utils.image_preprocess import preprocess_images_on_gpu


logger = get_logger("vqa_model")

# All supported model types for VQA training (excludes clip_zero_shot)
TRAINABLE_MODEL_TYPES = [
    "dist_align", "clip_baseline",
    "prolip", "grove",
]

# clip_zero_shot is handled separately in train_vqa.py (no classifier head)
ALL_MODEL_TYPES = TRAINABLE_MODEL_TYPES + ["clip_zero_shot"]


class VQAModel(nn.Module):
    """
    VQA Model with frozen backbone and trainable classification head.

    All model types produce 768-dim features per modality, concatenated
    to 1536-dim for the classification head. This ensures identical model
    architecture across all methods for fair comparison.

    For dist_align, features are sampled from learned Gaussian distributions
    during training, providing stochastic regularization. During evaluation,
    the deterministic distribution mean (mu) is used by default, with an
    optional MC sampling mode for uncertainty-aware predictions.
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
        num_mc_samples: int = 0,
    ):
        super().__init__()

        if model_type not in TRAINABLE_MODEL_TYPES:
            raise ValueError(
                f"VQAModel does not support model_type: {model_type}. "
                f"Use one of: {TRAINABLE_MODEL_TYPES}. "
                f"clip_zero_shot is handled separately."
            )

        self.model_type = model_type
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        self.dropout_rate = dropout
        self.answer_vocab = answer_vocab or {}
        self.num_mc_samples = num_mc_samples
        self.feature_dim = 768  # CLIP ViT-Large projection dimension (unified)

        # Load base model
        self._load_base_model(model_type, base_ckpt_path, device)

        # Freeze base model
        self._freeze_base()

        # Classification head: concat(img_feat, text_feat) -> logits
        self.classifier = nn.Sequential(
            nn.Linear(self.feature_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

        self._init_classifier()

        logger.info(
            f"VQAModel initialized: type={model_type}, "
            f"num_classes={num_classes}, hidden_dim={hidden_dim}, "
            f"feature_dim={self.feature_dim}"
        )

    def _load_base_model(self, model_type: str, ckpt_path: Optional[str], device: str):
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

        elif model_type == "prolip":
            from models.prolip_model import ProLIPModel
            self.base_model = ProLIPModel(
                freeze_clip=True,
                dropout_rate=config.DIST_ALIGN_DROPOUT_RATE,
            )
            if ckpt_path:
                logger.info(f"Loading prolip checkpoint: {ckpt_path}")
                self.base_model.load(ckpt_path)

        elif model_type == "grove":
            from models.grove_model import GroVEModel
            self.base_model = GroVEModel(
                num_inducing=config.GROVE_NUM_INDUCING,
                freeze_clip=True,
            )
            if ckpt_path:
                logger.info(f"Loading grove checkpoint: {ckpt_path}")
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

    def extract_image_features(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Extract image features from the base model.

        For dist_align, returns sampled z during training, mu during eval.
        For all other models, delegates to base_model.encode_image().

        Args:
            pixel_values: Image tensor (B, 3, 224, 224)

        Returns:
            Image features (B, 768)
        """
        if self.model_type == "dist_align":
            clip_feat = self.base_model.clip_model.get_image_features(pixel_values)
            clip_feat = clip_feat.pooler_output
            img_mu = self.base_model.img_mu_head(clip_feat)

            if self.training:
                img_logvar = self.base_model.img_logvar_head(clip_feat)
                eps = torch.randn_like(img_mu)
                return img_mu + eps * torch.exp(0.5 * img_logvar)
            else:
                return img_mu

        elif self.model_type == "clip_baseline":
            return self.base_model.encode_image(pixel_values, normalize=False)

        else:
            # prolip, grove: all have encode_image(pixel_values)
            return self.base_model.encode_image(pixel_values)

    def extract_text_features(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Extract text features using the base model.

        For dist_align, returns sampled z during training, mu during eval.
        For all other models, delegates to base_model.encode_text().

        Args:
            input_ids: Token IDs (B, seq_len)
            attention_mask: Attention mask (B, seq_len)

        Returns:
            Text features (B, 768)
        """
        if self.model_type == "dist_align":
            clip_feat = self.base_model.clip_model.get_text_features(
                input_ids=input_ids, attention_mask=attention_mask,
            )
            clip_feat = clip_feat.pooler_output
            text_mu = self.base_model.text_mu_head(clip_feat)

            if self.training:
                text_logvar = self.base_model.text_logvar_head(clip_feat)
                eps = torch.randn_like(text_mu)
                return text_mu + eps * torch.exp(0.5 * text_logvar)
            else:
                return text_mu

        elif self.model_type == "clip_baseline":
            return self.base_model.encode_text(input_ids, attention_mask, normalize=False)

        else:
            # prolip, grove: all have encode_text(input_ids, attention_mask)
            return self.base_model.encode_text(input_ids, attention_mask)

    def _sample_dist_image_features(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Sample image features from the learned distribution (dist_align only)."""
        clip_feat = self.base_model.clip_model.get_image_features(pixel_values)
        clip_feat = clip_feat.pooler_output
        img_mu = self.base_model.img_mu_head(clip_feat)
        img_logvar = self.base_model.img_logvar_head(clip_feat)
        eps = torch.randn_like(img_mu)
        return img_mu + eps * torch.exp(0.5 * img_logvar)

    def _sample_dist_text_features(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Sample text features from the learned distribution (dist_align only)."""
        clip_feat = self.base_model.clip_model.get_text_features(
            input_ids=input_ids, attention_mask=attention_mask,
        )
        clip_feat = clip_feat.pooler_output
        text_mu = self.base_model.text_mu_head(clip_feat)
        text_logvar = self.base_model.text_logvar_head(clip_feat)
        eps = torch.randn_like(text_mu)
        return text_mu + eps * torch.exp(0.5 * text_logvar)

    def process_images(self, images: List) -> torch.Tensor:
        """Process PIL images to tensors using CLIP processor."""
        device = next(self.parameters()).device
        return preprocess_images_on_gpu(images, device)

    def process_text(self, texts: List[str]) -> Dict[str, torch.Tensor]:
        """Process text strings to token IDs using CLIP processor."""
        return self.base_model.processor(
            text=texts, return_tensors="pt",
            padding=True, truncation=True, max_length=77,
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
        # MC sampling for dist_align during evaluation
        if (self.model_type == "dist_align"
                and self.num_mc_samples > 0
                and not self.training):
            all_logits = []
            for _ in range(self.num_mc_samples):
                img_feat = self._sample_dist_image_features(pixel_values)
                text_feat = self._sample_dist_text_features(input_ids, attention_mask)
                combined = torch.cat([img_feat, text_feat], dim=1)
                all_logits.append(self.classifier(combined))
            return torch.stack(all_logits).mean(dim=0)

        # Standard forward
        img_feat = self.extract_image_features(pixel_values)
        text_feat = self.extract_text_features(input_ids, attention_mask)
        combined = torch.cat([img_feat, text_feat], dim=1)
        return self.classifier(combined)

    def trainable_parameters(self) -> List[nn.Parameter]:
        return [p for p in self.parameters() if p.requires_grad]

    def num_trainable_parameters(self) -> int:
        return sum(p.numel() for p in self.trainable_parameters())

    def save(self, path: str) -> None:
        """Save model state (classifier head only, base is frozen)."""
        state = {
            "classifier_state_dict": self.classifier.state_dict(),
            "model_type": self.model_type,
            "num_classes": self.num_classes,
            "hidden_dim": self.hidden_dim,
            "dropout_rate": self.dropout_rate,
            "answer_vocab": self.answer_vocab,
            "num_mc_samples": self.num_mc_samples,
        }
        torch.save(state, path)
        logger.info(f"VQA model saved to: {path}")

    def load_classifier(self, path: str) -> None:
        """Load model state (classifier + base model adapters)."""
        state = torch.load(path, map_location="cpu", weights_only=False)
        self.classifier.load_state_dict(state["classifier_state_dict"])

        # Restore adapter weights if present (backward compat)
        if "adapter_state_dict" in state:
            self.base_model.load_state_dict(state["adapter_state_dict"], strict=False)
            logger.info("Adapter weights restored")

        if "answer_vocab" in state:
            self.answer_vocab = state["answer_vocab"]
        if "num_classes" in state:
            self.num_classes = state["num_classes"]
        if "num_mc_samples" in state:
            self.num_mc_samples = state["num_mc_samples"]

        logger.info(f"VQA model loaded from: {path}")

    def load_classifier_from_state(self, state: dict) -> None:
        """Load classifier state from a resume checkpoint dict."""
        if "classifier_state_dict" in state:
            self.classifier.load_state_dict(state["classifier_state_dict"])
        elif "model_state_dict" in state:
            # Resume from a full-model checkpoint (model.state_dict())
            full_state = state["model_state_dict"]
            classifier_state = {
                k.replace("classifier.", ""): v
                for k, v in full_state.items()
                if k.startswith("classifier.")
            }
            self.classifier.load_state_dict(classifier_state)
        if "answer_vocab" in state:
            self.answer_vocab = state["answer_vocab"]
        logger.info("VQA classifier restored from resume checkpoint")
