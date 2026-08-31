"""
MCDisp_Align Model

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
from utils.image_preprocess import preprocess_images_on_gpu


logger = get_logger("mcdisp_align_model")


class MCDispAlignModel(nn.Module):
    """
    MCDisp_Align model for multi-modal semantic matching.

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
        dropout_rate: float = 0.1,
        cov_rank: Optional[int] = None,
    ):
        """
        Initialize distribution alignment model.

        Args:
            model_path: Path to local CLIP model directory
                      (uses config.CLIP_VIT_L_14_PATH if None)
            hidden_dim: Hidden dimension for CLIP embeddings
            freeze_clip: Whether to freeze CLIP parameters
            dropout_rate: Dropout rate for MLP heads
            cov_rank: Low-rank covariance rank r for the IMAGE Gaussian
                      Sigma_v = diag(sigma_v^2) + U_v U_v^T (image-side only;
                      text distributions are diagonal, as in the paper).
                      0 = diagonal only. Defaults to config.
        """
        super().__init__()

        self.model_path = model_path or str(config.CLIP_VIT_L_14_PATH)
        self.hidden_dim = hidden_dim
        self.freeze_clip = freeze_clip
        self.dropout_rate = dropout_rate
        # Low-rank covariance rank r (0 = diagonal-only). Defaults to config.
        self.cov_rank = cov_rank if cov_rank is not None else config.MCDISP_ALIGN_COV_RANK

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

        # Low-rank covariance factor heads U in R^{D x r} (image and text).
        # Built via _build_cov_heads() so load() can rebuild them to match a
        # checkpoint's cov_rank before load_state_dict.
        self._build_cov_heads()

        # Initialize distribution heads
        self._init_distribution_heads()

        logger.info("Distribution alignment model initialized")

    def _freeze_clip(self) -> None:
        """Freeze CLIP model parameters."""
        for param in self.clip_model.parameters():
            param.requires_grad = False

    def _init_distribution_heads(self) -> None:
        """Initialize mu/logvar distribution heads with Xavier init (cov heads are initialized separately in _build_cov_heads)."""
        for head in [self.img_mu_head, self.img_logvar_head,
                     self.text_mu_head, self.text_logvar_head]:
            for layer in head:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
                    if layer.bias is not None:
                        nn.init.zeros_(layer.bias)

    def _cov_factor(
        self, head: nn.Module, features: torch.Tensor
    ) -> Optional[torch.Tensor]:
        """Predict the low-rank covariance factor U.

        Args:
            head: A cov head (img_cov_head or text_cov_head).
            features: Backbone features of shape (..., D).

        Returns:
            U of shape (..., D, r), or None when cov_rank == 0.
        """
        if self.cov_rank == 0:
            return None
        lead = features.shape[:-1]
        u = head(features)                                  # (..., D*r)
        u = u.view(*lead, self.hidden_dim, self.cov_rank)   # (..., D, r)
        return u

    def _build_cov_heads(self) -> None:
        """(Re)create the low-rank covariance factor head for ``self.cov_rank``.

        Covariance is image-side only (text is diagonal in
        v1), so only ``img_cov_head`` is built. Small (non-zero) init keeps
        Sigma near-diagonal at the start while still letting the L_cov
        subspace-alignment gradient bootstrap U (zero init would make that loss
        have zero gradient w.r.t. U). Called from ``__init__`` and from
        ``load()`` (to match a checkpoint's cov_rank).
        """
        # Drop any existing head so a cov_rank change during load() is clean.
        if hasattr(self, "img_cov_head"):
            del self.img_cov_head

        if self.cov_rank > 0:
            self.img_cov_head = nn.Linear(self.hidden_dim, self.hidden_dim * self.cov_rank)
            nn.init.normal_(self.img_cov_head.weight, std=1e-2)
            nn.init.zeros_(self.img_cov_head.bias)

    def _floor_logvar(self, raw_logvar: torch.Tensor) -> torch.Tensor:
        """Map a raw head output to log-variance with a small numerical floor.

        sigma^2 = softplus(x) + VAR_FLOOR, so sigma^2 > VAR_FLOOR (~1e-4) and is
        smooth and strictly positive everywhere. This is a *numerical* floor only
        (prevents exp / division blow-up), NOT a semantic floor: the variance
        range is learned through training, driven by L_var (data-driven caption
        spread) and L_reg (pull toward sigma_0^2). The old hard 0.1 floor is
        removed so sigma^2 can track the true caption spread even when it is
        below 0.1.
        """
        return torch.log(F.softplus(raw_logvar) + config.MCDISP_ALIGN_VAR_FLOOR)

    def merge_distributions(
        self,
        mus: torch.Tensor,
        logvars: torch.Tensor,
        us: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Form the text distribution from the K per-caption Gaussians
        N(mu_k, Sigma_k) by moment matching (paper §3.2, eq:caption_set_moments:
        Sigma_bar_t = S_t + (1/K) sum_k Sigma_k^t).

        The text variance diag(Sigma_bar_t) includes both the averaged
        per-caption diagonal sigma_k^2 and the between-caption dispersion
        s_t^2, so it is the diagonal of the moment-matched covariance.

        Args:
            mus: Per-caption means of shape (B, K, D)
            logvars: Per-caption log variances of shape (B, K, D)
            us: Per-caption low-rank factors of shape (B, K, D, r) or None

        Returns:
            Tuple of (combined_mu, combined_logvar), each of shape (B, D),
            where combined_logvar is log(diag(Sigma_bar_t) + eps) -- the text
            variance that L_var regresses the image variance to.
        """
        return self._moment_matching(mus, logvars, us)

    def _moment_matching(
        self,
        mus: torch.Tensor,
        logvars: torch.Tensor,
        us: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Moment-match the K per-caption Gaussians into the text distribution.

        diag(Sigma_bar_t) = (1/K) sum_k [ sigma_k^2 + diag(U_k U_k^T) ]
                          + (1/K) sum_k (mu_k - mu_bar)^2
                          = s_t^2 + (1/K) sum_k sigma_k^2   (text variance)

        Args:
            mus: Per-caption means of shape (B, K, D)
            logvars: Per-caption log variances of shape (B, K, D)
            us: Per-caption low-rank factors of shape (B, K, D, r) or None

        Returns:
            Tuple of (combined_mu, combined_logvar), each of shape (B, D)
        """
        # Combine means: mu_bar = (1/K) sum_k mu_k
        combined_mu = mus.mean(dim=1)  # (B, D)

        # A07: centered second moment. The old form
        # mean(var + mu^2) - mean(mu)^2 cancels effective digits when the
        # shared center is large and the caption spread is small; the
        # centered form is exact and matches the statistics the loss and
        # diagnostics already compute from deviations.
        diag_cov = torch.exp(logvars)  # sigma_k^2, (B, K, D)
        if us is not None:
            diag_cov = diag_cov + (us ** 2).sum(dim=-1)  # + diag(U U^T)
        dev = mus - combined_mu.unsqueeze(1)                     # (B, K, D)
        avg_caption_var = diag_cov.mean(dim=1)                   # a_t (B, D)
        combined_var = avg_caption_var + (dev ** 2).mean(dim=1)  # a_t + s_t^2
        # clamp (not +eps) so K=1 returns the caption's own variance unshifted;
        # the floor is dead in practice since sigma_k^2 >= MCDISP_ALIGN_VAR_FLOOR.
        combined_logvar = torch.log(combined_var.clamp_min(config.MCDISP_ALIGN_COV_EPS))  # (B, D)

        return combined_mu, combined_logvar

    def forward(
        self,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass - encode images and K captions into distributions.

        The image is modeled as a general Gaussian N(mu_v, Sigma_v) with
        Sigma_v = diag(sigma_v^2) + U_v U_v^T (U is None when cov_rank == 0);
        each caption text distribution is diagonal Gaussian
        N(mu_{i,k}^t, diag(sigma_{i,k}^{t 2})).

        Args:
            pixel_values: Image tensor of shape (B, C, H, W) from processor
            input_ids: Text input_ids of shape (B, K, max_len)
            attention_mask: Attention mask of shape (B, K, max_len)

        Returns:
            Dictionary containing (all standard keys are kept for backward
            compatibility with eval scripts):
                - img_features / text_features: CLIP features (B, D)
                - img_mu / img_logvar / img_sigma: image distribution (B, D)
                - text_mu / text_logvar / text_sigma: text distribution formed
                  from the K per-caption distributions by moment matching
                  (text mean mu_bar_t and text variance diag(Sigma_bar_t)) (B, D)
                - text_mus: per-caption means (B, K, D)
                - text_logvars: per-caption log variances (B, K, D)
                - img_U: image covariance factor (B, D, r) or None
                - text_Us: always None (caption distributions are diagonal)
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

        # Image distribution: mu, sigma^2, U
        img_mu = self.img_mu_head(img_features)  # (B, hidden_dim)
        img_logvar = self._floor_logvar(self.img_logvar_head(img_features))  # (B, hidden_dim)
        img_U = self._cov_factor(self.img_cov_head, img_features) if self.cov_rank > 0 else None

        # A distribution for each caption: mu, sigma^2. Caption distributions
        # are diagonal (the low-rank covariance is image-side), so no text_U.
        text_mus, text_logvars = [], []
        for k in range(K):
            fk = text_features[:, k, :]
            text_mus.append(self.text_mu_head(fk))
            text_logvars.append(self._floor_logvar(self.text_logvar_head(fk)))

        text_mus = torch.stack(text_mus, dim=1)  # (B, K, hidden_dim)
        text_logvars = torch.stack(text_logvars, dim=1)  # (B, K, hidden_dim)
        text_Us = None  # caption distributions are diagonal

        # Form the text distribution from the K per-caption distributions
        # (moment matching; text variance = dispersion + averaged caption var)
        text_mu, text_logvar = self.merge_distributions(
            text_mus, text_logvars, us=text_Us
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
            'text_mus': text_mus,            # Per-caption means
            'text_logvars': text_logvars,    # Per-caption log variances
            'img_U': img_U,                  # Image covariance factor (or None)
            'text_Us': text_Us,              # Per-caption covariance factors (or None)
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
        device = next(self.parameters()).device
        return preprocess_images_on_gpu(images, device)

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

    def save(self, path: str, epoch: Optional[int] = None,
             objective_version: Optional[str] = None) -> None:
        """
        Save model state.

        Args:
            path: Path to save checkpoint
            epoch: Optional training epoch this checkpoint was taken at (1-based;
                recorded for traceability of checkpoint selection; load() ignores
                it when absent, so older checkpoints stay loadable)
            objective_version: Optional objective identifier (A13,
                config.MCDISP_ALIGN_OBJECTIVE_VERSION). Stored as a top-level
                checkpoint key when not None; load() reads top-level keys
                conditionally, so old checkpoints (without it) stay loadable.
        """
        state = {
            "model_state_dict": self.state_dict(),
            "hidden_dim": self.hidden_dim,
            "freeze_clip": self.freeze_clip,
            "dropout_rate": self.dropout_rate,
            "cov_rank": self.cov_rank,
            "epoch": epoch,
        }
        if objective_version is not None:
            state["objective_version"] = objective_version
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

        # Match the checkpoint's covariance rank BEFORE loading weights, so a
        # cov_rank=0 (diagonal) checkpoint can load into a model constructed
        # with a different default cov_rank (and vice versa) without missing/
        # unexpected key errors.
        if "cov_rank" in state and state["cov_rank"] != self.cov_rank:
            logger.info(f"cov_rank {self.cov_rank} -> {state['cov_rank']} (from checkpoint); "
                        f"rebuilding covariance heads")
            self.cov_rank = state["cov_rank"]
            self._build_cov_heads()

        self.load_state_dict(state["model_state_dict"], strict=strict)
        logger.info(f"Model loaded from: {path}")

        # Load configuration
        if "hidden_dim" in state:
            self.hidden_dim = state["hidden_dim"]
        if "freeze_clip" in state:
            self.freeze_clip = state["freeze_clip"]
        if "dropout_rate" in state:
            self.dropout_rate = state["dropout_rate"]
        if "cov_rank" in state:
            self.cov_rank = state["cov_rank"]


if __name__ == "__main__":
    # Test model
    import config
    from utils.logger import setup_logger
    from utils.seed import set_seed

    # Setup
    setup_logger("mcdisp_align_model", config.LOG_DIR / "mcdisp_align_model_test.log")
    set_seed(config.SEED)

    model = MCDispAlignModel(
        freeze_clip=config.MCDISP_ALIGN_FREEZE_CLIP,
        dropout_rate=config.MCDISP_ALIGN_DROPOUT_RATE
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

    with torch.no_grad():
        outputs = model(dummy_images, dummy_input_ids, dummy_attention_mask)

    print(f"\nOutput shapes:")
    for key, value in outputs.items():
        shape = value.shape if value is not None else None
        print(f"  {key}: {shape}")

    # Verify covariance factor shapes (default model uses cov_rank > 0).
    # Text is diagonal-only, so text_Us is always None.
    assert outputs['img_U'] is not None, "img_U should exist for cov_rank>0"
    assert outputs['img_U'].shape == (batch_size, 768, config.MCDISP_ALIGN_COV_RANK)
    assert outputs['text_Us'] is None, "text_Us should be None (text diagonal-only)"
    assert outputs['text_logvars'].shape == (batch_size, num_captions, 768)
    print("\nCovariance factor shapes verified.")

    # Verify diagonal-only mode (cov_rank=0) returns None for U
    diag_model = MCDispAlignModel(
        freeze_clip=config.MCDISP_ALIGN_FREEZE_CLIP,
        dropout_rate=config.MCDISP_ALIGN_DROPOUT_RATE,
        cov_rank=0,
    )
    with torch.no_grad():
        diag_out = diag_model(dummy_images, dummy_input_ids, dummy_attention_mask)
    assert diag_out['img_U'] is None and diag_out['text_Us'] is None
    print("Diagonal-only mode (cov_rank=0): img_U/text_Us are None. Verified.")
