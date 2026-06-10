"""
GaussianImageDistribution - ICPE Baseline (B5)

ICPE: Intra-Class Probabilistic Embeddings.
Training-free method that computes intra-class covariance on CLIP features
as the distributional representation.

Reference: ICPE (training-free probabilistic embeddings)

Key difference from UC-CL: σ is a statistical quantity (intra-class covariance),
not learned via gradient descent. No training required.
"""

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPModel, CLIPProcessor

import config
from utils.logger import get_logger
from models.baseline_utils import encode_clip_features


logger = get_logger("icpe_model")


class ICPEModel(nn.Module):
    """
    ICPE Baseline wrapper (Training-Free).

    ICPE computes probabilistic embeddings without any training:
    1. Uses frozen CLIP to extract features
    2. Computes intra-class covariance from k-NN in the feature space
    3. Represents each sample as N(μ, σ²) where σ² is the k-NN covariance

    Since this is training-free, σ is purely a statistical quantity and
    may not capture the semantic diversity of descriptions.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        hidden_dim: int = 768,
        num_neighbors: int = 10,
        regularization: float = 1e-6,
    ):
        """
        Initialize ICPE model.

        Args:
            model_path: Path to local CLIP model directory
            hidden_dim: Hidden dimension for CLIP embeddings
            num_neighbors: Number of nearest neighbors for covariance estimation
            regularization: Regularization for covariance matrix
        """
        super().__init__()

        self.model_path = model_path or str(config.CLIP_VIT_L_14_PATH)
        self.num_neighbors = num_neighbors
        self.regularization = regularization

        # Load frozen CLIP
        logger.info(f"Loading CLIP model from: {self.model_path}")
        self.clip_model = CLIPModel.from_pretrained(
            self.model_path, local_files_only=True
        )
        self.processor = CLIPProcessor.from_pretrained(
            self.model_path, local_files_only=True
        )

        # Freeze all parameters (ICPE is training-free)
        for param in self.clip_model.parameters():
            param.requires_grad = False

        logger.info(f"ICPE model initialized (training-free, k={num_neighbors})")

    @torch.no_grad()
    def compute_icpe_covariance(
        self,
        features: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute ICPE variance from k-NN covariance.

        For each sample, find its k nearest neighbors in the feature space,
        then compute the per-dimension variance of those neighbors.

        Args:
            features: (N, D) feature matrix

        Returns:
            Per-sample variance (N, D)
        """
        features = F.normalize(features, dim=-1)
        N, D = features.shape
        k = min(self.num_neighbors, N)

        # Compute pairwise similarity
        sim = torch.matmul(features, features.T)  # (N, N)

        # For each sample, find top-k neighbors (excluding self)
        sim.fill_diagonal_(float("-inf"))
        _, topk_indices = sim.topk(k, dim=-1)  # (N, k)

        # Compute variance of neighbor features per dimension
        neighbor_features = features[topk_indices]  # (N, k, D)
        variance = neighbor_features.var(dim=1)  # (N, D)

        # Add regularization to prevent zero variance
        variance = variance + self.regularization

        return variance

    @torch.no_grad()
    def forward(
        self,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass (inference only, no gradients).

        Args:
            pixel_values: (B, C, H, W)
            input_ids: (B, K, max_len)
            attention_mask: (B, K, max_len)

        Returns:
            Dictionary with features and ICPE distribution parameters.
        """
        # CLIP encoding
        img_features, text_features, B, K = encode_clip_features(
            self.clip_model, pixel_values, input_ids, attention_mask
        )
        text_features_avg = text_features.mean(dim=1)

        # ICPE: μ = CLIP feature, σ² = k-NN covariance
        # Note: ICPE variance is computed after collecting all features
        # For forward pass, we use CLIP features directly as μ
        img_mu = img_features
        text_mu = text_features_avg

        # Placeholder variance (will be replaced after batch processing)
        # In practice, ICPE variance requires the full dataset
        img_logvar = torch.zeros_like(img_features)
        text_logvar = torch.zeros_like(text_features_avg)

        # Compute per-caption text mus
        text_mus = text_features  # (B, K, D) - use raw CLIP features

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

    @torch.no_grad()
    def compute_distributions_batch(
        self,
        all_img_features: torch.Tensor,
        all_text_features: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute ICPE distributions for a full dataset.

        Args:
            all_img_features: (N, D) all image features
            all_text_features: (N, D) all text features

        Returns:
            (img_logvar, text_logvar) each (N, D)
        """
        img_var = self.compute_icpe_covariance(all_img_features)
        text_var = self.compute_icpe_covariance(all_text_features)

        img_logvar = torch.log(img_var)
        text_logvar = torch.log(text_var)

        return img_logvar, text_logvar

    def encode_image(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Extract image features (CLIP, deterministic)."""
        clip_feat = self.clip_model.get_image_features(pixel_values)
        return clip_feat.pooler_output

    def encode_text(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """Extract text features (CLIP, deterministic)."""
        clip_feat = self.clip_model.get_text_features(
            input_ids=input_ids, attention_mask=attention_mask
        )
        return clip_feat.pooler_output

    def process_images(self, images: List) -> torch.Tensor:
        return self.processor(images=images, return_tensors="pt")["pixel_values"]

    def process_text(self, texts: List[str]) -> Dict[str, torch.Tensor]:
        return self.processor(
            text=texts, return_tensors="pt",
            padding=True, truncation=True, max_length=77,
        )

    def trainable_parameters(self) -> List[nn.Parameter]:
        return []

    def num_trainable_parameters(self) -> int:
        return 0

    def save(self, path: str) -> None:
        """ICPE has no trainable parameters, save config only."""
        state = {
            "model_type": "icpe",
            "num_neighbors": self.num_neighbors,
            "regularization": self.regularization,
        }
        torch.save(state, path)
        logger.info(f"ICPE config saved to: {path}")

    def load(self, path: str, strict: bool = True) -> None:
        """Load ICPE config (no model weights needed)."""
        state = torch.load(path, map_location="cpu", weights_only=False)
        if "num_neighbors" in state:
            self.num_neighbors = state["num_neighbors"]
        logger.info(f"ICPE config loaded from: {path}")
