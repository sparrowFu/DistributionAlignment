"""
GaussianImageDistribution - D2P Baseline (B6)

D2P: Distribution to Point matching.
Learns to match a distribution of embeddings (from multiple captions)
to a single point embedding (from image), using a distribution matching loss.

Reference: D2P (Distribution to Point)

Key difference from UC-CL: D2P does not model σ for the image side;
it focuses on matching distributions via sampling-based losses.
"""

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPModel, CLIPProcessor

import config
from utils.logger import get_logger
from models.baseline_utils import merge_distributions_moment_matching, encode_clip_features


logger = get_logger("d2p_model")


class D2PModel(nn.Module):
    """
    D2P Baseline wrapper.

    D2P learns distributional text representations and matches them to
    point-level image representations:

    1. Image: single point embedding via CLIP + projection
    2. Text: distribution embedding via CLIP + MLP head, sampled from
       a learned Gaussian
    3. Matching: distribution-to-point loss using sampled features

    Unlike UC-CL, D2P does not model image uncertainty (σ_img).
    The image side is a deterministic point embedding.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        hidden_dim: int = 768,
        freeze_clip: bool = True,
        num_samples: int = 10,
        dropout_rate: float = 0.1,
    ):
        """
        Initialize D2P model.

        Args:
            model_path: Path to local CLIP model directory
            hidden_dim: Hidden dimension for CLIP embeddings
            freeze_clip: Whether to freeze CLIP parameters
            num_samples: Number of samples for distribution-to-point matching
            dropout_rate: Dropout rate
        """
        super().__init__()

        self.model_path = model_path or str(config.CLIP_VIT_L_14_PATH)
        self.hidden_dim = hidden_dim
        self.freeze_clip = freeze_clip
        self.num_samples = num_samples
        self.dropout_rate = dropout_rate

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

        # Image: point embedding (no σ)
        self.img_projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Dropout(dropout_rate),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Text: distribution embedding (μ + σ)
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

        self._init_weights()

        logger.info(f"D2P model initialized (S={num_samples} samples)")

    def _freeze_clip(self):
        """Freeze CLIP parameters."""
        for param in self.clip_model.parameters():
            param.requires_grad = False

    def _init_weights(self):
        """Initialize weights."""
        for module in [self.img_projection, self.text_mu_head, self.text_logvar_head]:
            for layer in module:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
                    if layer.bias is not None:
                        nn.init.zeros_(layer.bias)

    def merge_distributions(
        self, mus: torch.Tensor, logvars: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Merge K caption distributions via moment matching."""
        return merge_distributions_moment_matching(mus, logvars)

    def d2p_loss(
        self,
        img_point: torch.Tensor,
        text_mu: torch.Tensor,
        text_logvar: torch.Tensor,
        temperature: float = 0.07,
    ) -> torch.Tensor:
        """
        Distribution-to-Point contrastive loss.

        Samples S text embeddings from N(text_mu, text_logvar) and computes
        the expected similarity with the image point embedding.

        Args:
            img_point: Image point embedding (B, D)
            text_mu: Text distribution mean (B, D)
            text_logvar: Text distribution log variance (B, D)
            temperature: Temperature for softmax

        Returns:
            D2P contrastive loss scalar
        """
        B = img_point.shape[0]

        # Sample S text features from each distribution
        img_point = F.normalize(img_point, dim=-1)

        # Expected similarity via Monte Carlo sampling
        total_loss = 0.0
        for _ in range(self.num_samples):
            eps = torch.randn_like(text_mu)
            text_sample = text_mu + eps * torch.exp(0.5 * text_logvar)
            text_sample = F.normalize(text_sample, dim=-1)

            # Compute similarity matrix
            logits = torch.matmul(img_point, text_sample.T) / temperature  # (B, B)
            labels = torch.arange(B, device=img_point.device)

            loss_i2t = F.cross_entropy(logits, labels)
            loss_t2i = F.cross_entropy(logits.T, labels)
            total_loss += (loss_i2t + loss_t2i) / 2

        return total_loss / self.num_samples

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

        # Image: point embedding (no σ for image)
        img_mu = self.img_projection(img_features)
        img_logvar = torch.zeros_like(img_mu)  # No variance for image

        # Text: distribution embedding
        text_mus, text_logvars = [], []
        for k in range(K):
            mu_k = self.text_mu_head(text_features[:, k, :])
            logvar_k = self.text_logvar_head(text_features[:, k, :])
            text_mus.append(mu_k)
            text_logvars.append(logvar_k)

        text_mus = torch.stack(text_mus, dim=1)
        text_logvars = torch.stack(text_logvars, dim=1)

        text_mu, text_logvar = self.merge_distributions(text_mus, text_logvars)

        img_sigma = torch.exp(0.5 * img_logvar)  # All ones (no variance)
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
        """Extract image point embedding."""
        clip_feat = self.clip_model.get_image_features(pixel_values)
        clip_feat = clip_feat.pooler_output
        return self.img_projection(clip_feat)

    def encode_text(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """Extract text distribution mean."""
        clip_feat = self.clip_model.get_text_features(
            input_ids=input_ids, attention_mask=attention_mask
        )
        clip_feat = clip_feat.pooler_output
        return self.text_mu_head(clip_feat)

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
            "freeze_clip": self.freeze_clip,
            "num_samples": self.num_samples,
            "dropout_rate": self.dropout_rate,
        }
        torch.save(state, path)
        logger.info(f"D2P model saved to: {path}")

    def load(self, path: str, strict: bool = True) -> None:
        state = torch.load(path, map_location="cpu", weights_only=False)
        self.load_state_dict(state["model_state_dict"], strict=strict)
        logger.info(f"D2P model loaded from: {path}")
