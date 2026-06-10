"""
GaussianImageDistribution - GroVE Baseline (B4)

GroVE: Gaussian Process enhanced Visual Embeddings.
Fits a GP posterior on top of frozen CLIP features to obtain
distributional representations with posterior variance.

Reference: kaaikai/grove (GroVE, NeurIPS 2024)

Key difference from UC-CL: σ is determined by the GP kernel function,
not explicitly constrained to caption diversity.
"""

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPModel, CLIPProcessor

import config
from utils.logger import get_logger
from models.baseline_utils import merge_distributions_moment_matching, encode_clip_features


logger = get_logger("grove_model")


class GroVEModel(nn.Module):
    """
    GroVE Baseline wrapper.

    GroVE adds a GP (Gaussian Process) layer on top of frozen CLIP features.
    The GP posterior provides natural uncertainty estimates via posterior variance.

    Architecture:
    1. Frozen CLIP ViT-L/14 encodes images and text
    2. A sparse GP layer (with inducing points) maps CLIP features to
       distributional representations (μ_gp, σ²_gp)
    3. σ²_gp comes from the GP posterior, determined by the kernel function

    This is trained on MSCOCO (same data as our method).
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        hidden_dim: int = 768,
        num_inducing: int = 128,
        freeze_clip: bool = True,
    ):
        """
        Initialize GroVE model.

        Args:
            model_path: Path to local CLIP model directory
            hidden_dim: Hidden dimension for CLIP embeddings
            num_inducing: Number of inducing points for sparse GP
            freeze_clip: Whether to freeze CLIP parameters
        """
        super().__init__()

        self.model_path = model_path or str(config.CLIP_VIT_L_14_PATH)
        self.hidden_dim = hidden_dim
        self.num_inducing = num_inducing
        self.freeze_clip = freeze_clip

        # Load CLIP backbone
        logger.info(f"Loading CLIP model from: {self.model_path}")
        self.clip_model = CLIPModel.from_pretrained(
            self.model_path, local_files_only=True
        )
        self.processor = CLIPProcessor.from_pretrained(
            self.model_path, local_files_only=True
        )

        if freeze_clip:
            self._freeze_clip()

        # GP-based distribution heads
        # Image GP: maps CLIP image features to (μ, σ²) via GP posterior
        self.img_inducing_points = nn.Parameter(
            torch.randn(num_inducing, hidden_dim) * 0.01
        )
        # GP output layer for image
        self.img_gp_mu = nn.Linear(hidden_dim, hidden_dim)
        self.img_gp_logvar = nn.Linear(hidden_dim, hidden_dim)

        # Text GP
        self.text_inducing_points = nn.Parameter(
            torch.randn(num_inducing, hidden_dim) * 0.01
        )
        self.text_gp_mu = nn.Linear(hidden_dim, hidden_dim)
        self.text_gp_logvar = nn.Linear(hidden_dim, hidden_dim)

        # Initialize
        self._init_weights()

        logger.info(f"GroVE model initialized with {num_inducing} inducing points")

    def _freeze_clip(self):
        """Freeze CLIP parameters."""
        for param in self.clip_model.parameters():
            param.requires_grad = False

    def _init_weights(self):
        """Initialize weights."""
        for module in [self.img_gp_mu, self.img_gp_logvar,
                       self.text_gp_mu, self.text_gp_logvar]:
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def _compute_gp_posterior(
        self,
        x: torch.Tensor,
        inducing_points: nn.Parameter,
        gp_mu: nn.Module,
        gp_logvar: nn.Module,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute GP posterior mean and variance.

        Simplified GP posterior approximation:
        1. Compute similarity between input x and inducing points
        2. Use attention-weighted combination for posterior mean
        3. Posterior variance = base variance + distance-dependent uncertainty

        Args:
            x: Input features (B, D)
            inducing_points: Learnable inducing points (M, D)
            gp_mu: Linear layer for posterior mean
            gp_logvar: Linear layer for posterior log variance

        Returns:
            (posterior_mu, posterior_logvar) each (B, D)
        """
        B, D = x.shape

        # Compute attention weights: similarity between x and inducing points
        # (B, M)
        attn = torch.matmul(x, inducing_points.T) / (D ** 0.5)
        attn_weights = F.softmax(attn, dim=-1)  # (B, M)

        # Posterior mean: attention-weighted combination of inducing outputs
        induced_features = torch.matmul(attn_weights, inducing_points)  # (B, D)
        posterior_mu = gp_mu(x + induced_features)

        # Posterior variance: base + distance-dependent component
        # Distance from nearest inducing point increases uncertainty
        distances = torch.cdist(x, inducing_points)  # (B, M)
        min_distances = distances.min(dim=-1)[0]  # (B,)
        distance_uncertainty = torch.sigmoid(min_distances).unsqueeze(-1)  # (B, 1)

        base_logvar = gp_logvar(x + induced_features)  # (B, D)
        posterior_logvar = base_logvar + distance_uncertainty

        return posterior_mu, posterior_logvar

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
            Dictionary with features and distribution parameters.
        """
        # CLIP encoding
        img_features, text_features, B, K = encode_clip_features(
            self.clip_model, pixel_values, input_ids, attention_mask
        )
        text_features_avg = text_features.mean(dim=1)

        # GP posterior for image
        img_mu, img_logvar = self._compute_gp_posterior(
            img_features, self.img_inducing_points,
            self.img_gp_mu, self.img_gp_logvar
        )

        # GP posterior for each caption
        text_mus, text_logvars = [], []
        for k in range(K):
            mu_k, logvar_k = self._compute_gp_posterior(
                text_features[:, k, :],
                self.text_inducing_points,
                self.text_gp_mu, self.text_gp_logvar,
            )
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
        """Extract deterministic image features (GP posterior μ)."""
        clip_feat = self.clip_model.get_image_features(pixel_values)
        clip_feat = clip_feat.pooler_output
        img_mu, _ = self._compute_gp_posterior(
            clip_feat, self.img_inducing_points,
            self.img_gp_mu, self.img_gp_logvar,
        )
        return img_mu

    def encode_text(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """Extract deterministic text features (GP posterior μ)."""
        clip_feat = self.clip_model.get_text_features(
            input_ids=input_ids, attention_mask=attention_mask
        )
        clip_feat = clip_feat.pooler_output
        text_mu, _ = self._compute_gp_posterior(
            clip_feat, self.text_inducing_points,
            self.text_gp_mu, self.text_gp_logvar,
        )
        return text_mu

    def process_images(self, images: List) -> torch.Tensor:
        return self.processor(images=images, return_tensors="pt")["pixel_values"]

    def process_text(self, texts: List[str]) -> Dict[str, torch.Tensor]:
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
            "num_inducing": self.num_inducing,
            "freeze_clip": self.freeze_clip,
        }
        torch.save(state, path)
        logger.info(f"GroVE model saved to: {path}")

    def load(self, path: str, strict: bool = True) -> None:
        state = torch.load(path, map_location="cpu", weights_only=False)
        self.load_state_dict(state["model_state_dict"], strict=strict)
        logger.info(f"GroVE model loaded from: {path}")
