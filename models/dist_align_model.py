"""
GaussianImageDistribution - Distribution Alignment Model

This module implements a distribution-based image-text alignment model
that models image and text embeddings as Gaussian distributions.
"""

from typing import Dict, List, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPModel, CLIPProcessor

import config
from utils.logger import get_logger


logger = get_logger("dist_align_model")


class DistributionAlignmentModel(nn.Module):
    """
    Distribution Alignment Model for Multi-Modal Semantic Matching.

    This model extends CLIP by learning to model image and text embeddings
    as Gaussian distributions, addressing:
    - Modality gap between image and text embeddings
    - One-to-many relationships between images and descriptions

    Architecture:
    - CLIP encoder (frozen or fine-tuned)
    - Distribution modeling MLP heads for image and text
    - Distribution merging for multiple text descriptions
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        hidden_dim: int = 768,
        freeze_clip: bool = True,
        distribution_merging: str = "moment_matching",
        dropout_rate: float = 0.1
    ):
        """
        Initialize distribution alignment model.

        Args:
            model_path: Path to local CLIP model directory
                      (uses config.CLIP_VIT_L_14_PATH if None)
            hidden_dim: Hidden dimension for CLIP embeddings
            freeze_clip: Whether to freeze CLIP parameters
            distribution_merging: Method to merge multiple text distributions
                                 ("moment_matching", "poe", "simple")
            dropout_rate: Dropout rate for MLP heads
        """
        super().__init__()

        self.model_path = model_path or str(config.CLIP_VIT_L_14_PATH)
        self.hidden_dim = hidden_dim
        self.freeze_clip = freeze_clip
        self.distribution_merging = distribution_merging
        self.dropout_rate = dropout_rate

        # Load CLIP model from local files
        logger.info(f"Loading CLIP model from: {self.model_path}")
        self.clip_model = CLIPModel.from_pretrained(
            self.model_path,
            local_files_only=True
        )

        # Load CLIP processor from local files
        logger.info(f"Loading CLIP processor from: {self.model_path}")
        self.processor = CLIPProcessor.from_pretrained(
            self.model_path,
            local_files_only=True
        )

        # Freeze CLIP parameters if requested
        if freeze_clip:
            self._freeze_clip()
            logger.info("CLIP parameters frozen")
        else:
            logger.info("CLIP parameters will be fine-tuned")

        # Image distribution modeling heads
        self.img_mu_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Dropout(dropout_rate),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        self.img_logvar_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Dropout(dropout_rate),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # Text distribution modeling heads
        self.text_mu_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Dropout(dropout_rate),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        self.text_logvar_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Dropout(dropout_rate),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # Initialize distribution heads
        self._init_distribution_heads()

        logger.info("Distribution alignment model initialized")

    def _freeze_clip(self) -> None:
        """Freeze CLIP model parameters."""
        for param in self.clip_model.parameters():
            param.requires_grad = False

    def _init_distribution_heads(self) -> None:
        """Initialize distribution modeling heads with Xavier initialization."""
        for head in [self.img_mu_head, self.img_logvar_head,
                     self.text_mu_head, self.text_logvar_head]:
            for layer in head:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
                    if layer.bias is not None:
                        nn.init.zeros_(layer.bias)

    def merge_distributions(
        self,
        mus: torch.Tensor,
        logvars: torch.Tensor,
        method: str = "moment_matching"
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Merge multiple Gaussian distributions into one.

        Args:
            mus: Distribution means of shape (B, K, D)
            logvars: Distribution log variances of shape (B, K, D)
            method: Merging method ("moment_matching", "poe", "simple")

        Returns:
            Tuple of (combined_mu, combined_logvar), each of shape (B, D)
        """
        B, K, D = mus.shape

        if method == "moment_matching":
            return self._moment_matching(mus, logvars)
        elif method == "poe":
            return self._product_of_experts(mus, logvars)
        elif method == "simple":
            return mus.mean(dim=1), logvars.mean(dim=1)
        else:
            raise ValueError(f"Unknown merging method: {method}")

    def _moment_matching(
        self,
        mus: torch.Tensor,
        logvars: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Merge distributions using moment matching.

        Minimizes KL divergence between merged and original distributions.

        Args:
            mus: Distribution means of shape (B, K, D)
            logvars: Distribution log variances of shape (B, K, D)

        Returns:
            Tuple of (combined_mu, combined_logvar), each of shape (B, D)
        """
        B, K, D = mus.shape
        device = mus.device

        # Uniform weights
        weights = torch.ones(K, device=device) / K  # (K,)

        # Combine means: μ = Σ wᵢμᵢ
        combined_mu = (weights.view(1, K, 1) * mus).sum(dim=1)  # (B, D)

        # Combine variances: σ² = Σ wᵢ(σᵢ² + μᵢ²) - μ²
        vars = torch.exp(logvars)  # (B, K, D)
        combined_var = (weights.view(1, K, 1) * (vars + mus ** 2)).sum(dim=1) - combined_mu ** 2
        combined_logvar = torch.log(combined_var + 1e-6)  # (B, D)

        return combined_mu, combined_logvar

    def _product_of_experts(
        self,
        mus: torch.Tensor,
        logvars: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Merge distributions using Product of Experts (PoE).

        The product of multiple Gaussian distributions is still Gaussian.

        Args:
            mus: Distribution means of shape (B, K, D)
            logvars: Distribution log variances of shape (B, K, D)

        Returns:
            Tuple of (combined_mu, combined_logvar), each of shape (B, D)
        """
        vars = torch.exp(logvars)  # (B, K, D)

        # Precisions (inverse of variance)
        precisions = 1.0 / (vars + 1e-6)  # (B, K, D)

        # Combine precisions: τ = Σ τᵢ
        combined_precision = precisions.sum(dim=1)  # (B, D)

        # Combine means: μ = (Σ τᵢμᵢ) / τ
        combined_mu = (precisions * mus).sum(dim=1) / (combined_precision + 1e-6)  # (B, D)

        # Combine variances: σ² = 1/τ
        combined_var = 1.0 / (combined_precision + 1e-6)  # (B, D)
        combined_logvar = torch.log(combined_var + 1e-6)  # (B, D)

        return combined_mu, combined_logvar

    def forward(
        self,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass - encode images and captions to distributions.

        Args:
            pixel_values: Image tensor of shape (B, C, H, W) from processor
            input_ids: Text input_ids of shape (B, K, max_len)
            attention_mask: Attention mask of shape (B, K, max_len)

        Returns:
            Dictionary containing:
                - img_features: CLIP image features (B, hidden_dim)
                - text_features: CLIP text features (B, hidden_dim)
                - img_mu: Image distribution mean (B, hidden_dim)
                - img_logvar: Image distribution log variance (B, hidden_dim)
                - text_mu: Merged text distribution mean (B, hidden_dim)
                - text_logvar: Merged text distribution log variance (B, hidden_dim)
                - text_mus: Per-caption distribution means (B, K, hidden_dim)
                            Used by distributional consistency loss.
        """
        # CLIP encoding (keep features for contrastive loss)
        img_features = self.clip_model.get_image_features(pixel_values)
        img_features = img_features.pooler_output  # Extract actual tensor

        B, K, max_len = input_ids.shape
        input_ids_flat = input_ids.view(B * K, max_len)
        attention_mask_flat = attention_mask.view(B * K, max_len)

        text_features = self.clip_model.get_text_features(
            input_ids=input_ids_flat,
            attention_mask=attention_mask_flat
        )
        text_features = text_features.pooler_output  # Extract actual tensor
        text_features = text_features.view(B, K, -1)  # (B, K, hidden_dim)

        # Average text features for contrastive loss
        text_features_avg = text_features.mean(dim=1)  # (B, hidden_dim)

        # Image distribution
        img_mu = self.img_mu_head(img_features)  # (B, hidden_dim)
        img_logvar = self.img_logvar_head(img_features)  # (B, hidden_dim)

        # Text distributions (K captions)
        text_mus, text_logvars = [], []
        for k in range(K):
            mu_k = self.text_mu_head(text_features[:, k, :])
            logvar_k = self.text_logvar_head(text_features[:, k, :])
            text_mus.append(mu_k)
            text_logvars.append(logvar_k)

        text_mus = torch.stack(text_mus, dim=1)  # (B, K, hidden_dim)
        text_logvars = torch.stack(text_logvars, dim=1)  # (B, K, hidden_dim)

        # Merge distributions
        text_mu, text_logvar = self.merge_distributions(
            text_mus, text_logvars, method=self.distribution_merging
        )

        # Compute sigma for downstream use
        img_sigma = torch.exp(0.5 * img_logvar)
        text_sigma = torch.exp(0.5 * text_logvar)

        return {
            'img_features': img_features,
            'text_features': text_features_avg,
            'img_mu': img_mu,
            'img_logvar': img_logvar,
            'img_sigma': img_sigma,
            'text_mu': text_mu,
            'text_logvar': text_logvar,
            'text_sigma': text_sigma,
            'text_mus': text_mus,  # Per-caption means for distributional consistency loss
        }

    def process_images(
        self,
        images: List
    ) -> torch.Tensor:
        """
        Process a list of PIL images to tensors.

        Args:
            images: List of PIL Images

        Returns:
            Processed image tensor of shape (B, C, H, W)
        """
        return self.processor(images=images, return_tensors="pt")["pixel_values"]

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
        return self.processor(
            text=texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=77  # CLIP's max sequence length
        )

    def encode_image(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Extract deterministic image features (distribution mean).

        Args:
            pixel_values: Image tensor (B, 3, 224, 224)

        Returns:
            Image mu features (B, hidden_dim)
        """
        clip_feat = self.clip_model.get_image_features(pixel_values)
        clip_feat = clip_feat.pooler_output
        return self.img_mu_head(clip_feat)

    def encode_text(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Extract deterministic text features (distribution mean).

        Args:
            input_ids: Token IDs (B, seq_len)
            attention_mask: Attention mask (B, seq_len)

        Returns:
            Text mu features (B, hidden_dim)
        """
        clip_feat = self.clip_model.get_text_features(
            input_ids=input_ids, attention_mask=attention_mask,
        )
        clip_feat = clip_feat.pooler_output
        return self.text_mu_head(clip_feat)

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
            "hidden_dim": self.hidden_dim,
            "freeze_clip": self.freeze_clip,
            "distribution_merging": self.distribution_merging,
            "dropout_rate": self.dropout_rate,
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

        # Load configuration
        if "hidden_dim" in state:
            self.hidden_dim = state["hidden_dim"]
        if "freeze_clip" in state:
            self.freeze_clip = state["freeze_clip"]
        if "distribution_merging" in state:
            self.distribution_merging = state["distribution_merging"]
        if "dropout_rate" in state:
            self.dropout_rate = state["dropout_rate"]


if __name__ == "__main__":
    # Test model
    import config
    from utils.logger import setup_logger
    from utils.seed import set_seed

    # Setup
    setup_logger("dist_align_model", config.LOG_DIR / "dist_align_model_test.log")
    set_seed(config.SEED)

    # Create model
    model = DistributionAlignmentModel(
        freeze_clip=config.DIST_ALIGN_FREEZE_CLIP,
        distribution_merging=config.DIST_ALIGN_DISTRIBUTION_MERGING,
        dropout_rate=config.DIST_ALIGN_DROPOUT_RATE
    )

    print(f"Model created successfully")
    print(f"Trainable parameters: {model.num_trainable_parameters():,}")

    # Test forward pass with dummy data
    batch_size = 2
    num_captions = 5
    max_seq_len = 77

    # Create dummy inputs
    dummy_images = torch.randn(batch_size, 3, 224, 224)
    dummy_input_ids = torch.randint(0, 49408, (batch_size, num_captions, max_seq_len))
    dummy_attention_mask = torch.ones(batch_size, num_captions, max_seq_len, dtype=torch.long)

    print(f"\nInput shapes:")
    print(f"  Images: {dummy_images.shape}")
    print(f"  Input IDs: {dummy_input_ids.shape}")

    # Forward pass
    with torch.no_grad():
        outputs = model(dummy_images, dummy_input_ids, dummy_attention_mask)

    print(f"\nOutput shapes:")
    for key, value in outputs.items():
        print(f"  {key}: {value.shape}")
