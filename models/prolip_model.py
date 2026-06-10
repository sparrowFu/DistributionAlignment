"""
GaussianImageDistribution - ProLIP Baseline (B3)

Probabilistic representation learning via inclusion-based embeddings.
ProLIP models each sample as a Gaussian distribution in the embedding space,
with σ learned implicitly from the inclusion loss during pretraining.

Reference: https://arxiv.org/abs/2410.02337 (ProLIP)

This wrapper loads ProLIP pretrained weights and provides the same interface
as DistributionAlignmentModel for fair comparison.
"""

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from transformers import CLIPModel, CLIPProcessor

import config
from utils.logger import get_logger
from models.baseline_utils import merge_distributions_moment_matching, encode_clip_features, init_heads_xavier


logger = get_logger("prolip_model")


class ProLIPModel(nn.Module):
    """
    ProLIP Baseline wrapper.

    ProLIP extends CLIP by learning probabilistic embeddings where each sample
    is represented as a Gaussian N(μ, σ²I). Unlike our UC-CL, ProLIP's σ is
    learned implicitly from an inclusion-based loss on 1B image-text pairs.

    This wrapper:
    1. Loads a pretrained CLIP ViT-L/14 as the base encoder
    2. Adds μ and σ heads (same architecture as our method for fairness)
    3. Loads ProLIP pretrained weights if available, otherwise uses random init

    Key difference from UC-CL: σ has no explicit semantic constraint.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        hidden_dim: int = 768,
        freeze_clip: bool = True,
        dropout_rate: float = 0.1,
        prolip_weights_path: Optional[str] = None,
    ):
        """
        Initialize ProLIP model.

        Args:
            model_path: Path to local CLIP model directory
            hidden_dim: Hidden dimension for embeddings
            freeze_clip: Whether to freeze CLIP parameters
            dropout_rate: Dropout rate for heads
            prolip_weights_path: Path to ProLIP pretrained weights (optional)
        """
        super().__init__()

        self.model_path = model_path or str(config.CLIP_VIT_L_14_PATH)
        self.hidden_dim = hidden_dim
        self.freeze_clip = freeze_clip
        self.dropout_rate = dropout_rate

        # Load CLIP backbone (same as our method)
        logger.info(f"Loading CLIP model from: {self.model_path}")
        self.clip_model = CLIPModel.from_pretrained(
            self.model_path, local_files_only=True
        )
        self.processor = CLIPProcessor.from_pretrained(
            self.model_path, local_files_only=True
        )

        if freeze_clip:
            self._freeze_clip()

        # ProLIP: μ and σ heads (same architecture as UC-CL for fair comparison)
        # σ is learned via inclusion loss, NOT constrained to caption variance
        self.img_mu_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Dropout(dropout_rate),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.img_logvar_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Dropout(dropout_rate),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.text_mu_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Dropout(dropout_rate),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.text_logvar_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Dropout(dropout_rate),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Initialize heads
        self._init_heads()

        # Load ProLIP pretrained weights if available
        if prolip_weights_path:
            self._load_prolip_weights(prolip_weights_path)

        logger.info("ProLIP model initialized")

    def _freeze_clip(self):
        """Freeze CLIP parameters."""
        for param in self.clip_model.parameters():
            param.requires_grad = False

    def _init_heads(self):
        """Initialize head weights with Xavier initialization."""
        init_heads_xavier([self.img_mu_head, self.img_logvar_head,
                           self.text_mu_head, self.text_logvar_head])

    def _load_prolip_weights(self, path: str):
        """Load ProLIP-specific weights (μ and σ heads only)."""
        try:
            state = torch.load(path, map_location="cpu", weights_only=False)
            if "model_state_dict" in state:
                self.load_state_dict(state["model_state_dict"], strict=False)
            else:
                self.load_state_dict(state, strict=False)
            logger.info(f"ProLIP weights loaded from: {path}")
        except Exception as e:
            logger.warning(f"Failed to load ProLIP weights: {e}. Using random init.")

    def merge_distributions(
        self, mus: torch.Tensor, logvars: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Merge K caption distributions via moment matching."""
        return merge_distributions_moment_matching(mus, logvars)

    def forward(
        self,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.

        Args:
            pixel_values: (B, C, H, W)
            input_ids: (B, K, max_len)
            attention_mask: (B, K, max_len)

        Returns:
            Dictionary with img/text features and distribution parameters.
        """
        # CLIP encoding
        img_features, text_features, B, K = encode_clip_features(
            self.clip_model, pixel_values, input_ids, attention_mask
        )
        text_features_avg = text_features.mean(dim=1)

        # Distribution parameters
        img_mu = self.img_mu_head(img_features)
        img_logvar = self.img_logvar_head(img_features)

        text_mus, text_logvars = [], []
        for k in range(K):
            mu_k = self.text_mu_head(text_features[:, k, :])
            logvar_k = self.text_logvar_head(text_features[:, k, :])
            text_mus.append(mu_k)
            text_logvars.append(logvar_k)

        text_mus = torch.stack(text_mus, dim=1)
        text_logvars = torch.stack(text_logvars, dim=1)

        text_mu, text_logvar = self.merge_distributions(text_mus, text_logvars)

        img_sigma = torch.exp(0.5 * img_logvar)
        text_sigma = torch.exp(0.5 * text_logvar)

        return {
            "img_features": img_features,
            "text_features": text_features_avg,
            "img_mu": img_mu,
            "img_logvar": img_logvar,
            "img_sigma": img_sigma,
            "text_mu": text_mu,
            "text_logvar": text_logvar,
            "text_sigma": text_sigma,
            "text_mus": text_mus,
        }

    def encode_image(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Extract deterministic image features (μ)."""
        clip_feat = self.clip_model.get_image_features(pixel_values)
        clip_feat = clip_feat.pooler_output
        return self.img_mu_head(clip_feat)

    def encode_text(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """Extract deterministic text features (μ)."""
        clip_feat = self.clip_model.get_text_features(
            input_ids=input_ids, attention_mask=attention_mask
        )
        clip_feat = clip_feat.pooler_output
        return self.text_mu_head(clip_feat)

    def process_images(self, images: List) -> torch.Tensor:
        """Process PIL images to tensors."""
        return self.processor(images=images, return_tensors="pt")["pixel_values"]

    def process_text(self, texts: List[str]) -> Dict[str, torch.Tensor]:
        """Process text strings to model inputs."""
        return self.processor(
            text=texts, return_tensors="pt",
            padding=True, truncation=True, max_length=77,
        )

    def trainable_parameters(self) -> List[nn.Parameter]:
        return [p for p in self.parameters() if p.requires_grad]

    def num_trainable_parameters(self) -> int:
        return sum(p.numel() for p in self.trainable_parameters())

    def save(self, path: str) -> None:
        state = {
            "model_state_dict": self.state_dict(),
            "hidden_dim": self.hidden_dim,
            "freeze_clip": self.freeze_clip,
            "dropout_rate": self.dropout_rate,
        }
        torch.save(state, path)
        logger.info(f"ProLIP model saved to: {path}")

    def load(self, path: str, strict: bool = True) -> None:
        state = torch.load(path, map_location="cpu", weights_only=False)
        self.load_state_dict(state["model_state_dict"], strict=strict)
        logger.info(f"ProLIP model loaded from: {path}")
