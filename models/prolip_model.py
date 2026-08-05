"""
GaussianImageDistribution - ProLIP Baseline (B3)

Real ProLIP ViT-H/14 (SanghyukChun/ProLIP-ViT-H-14-FT-DC-1B-1_28M) loaded via
the `prolip` library. Each image/text is modeled as a Gaussian N(mu, sigma^2 I)
where mu and log(sigma^2) come from ProLIP's pretrained uncertainty heads.
sigma is learned implicitly via ProLIP's inclusion objective during
pretraining; unlike our MCDisp_Align, it carries no explicit semantic constraint.

Two usage modes (mirror the CLIP baseline):
  - zero_shot : ProLIPModel(freeze=True) -- frozen pretrained weights, no checkpoint
  - fine_tune : ProLIPModel() then .load(config.PROLIP_BEST_CKPT), or train via
                scripts/train_prolip.py with the ProLIP inclusion loss.

Three local artifacts (no network): config.PROLIP_{MODEL,PROCESSOR,TOKENIZER}_PATH.

Reference: https://arxiv.org/abs/2410.18857 (ProLIP)
"""

import os
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

import config
from utils.logger import get_logger


# All ProLIP artifacts are local (see config.PROLIP_*_PATH); never hit the hub.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from prolip.model import ProLIPHF                # noqa: E402
from prolip.tokenizer import HFTokenizer          # noqa: E402
from transformers import CLIPProcessor            # noqa: E402


logger = get_logger("prolip_model")


def merge_distributions_moment_matching(
    mus: torch.Tensor, logvars: torch.Tensor
) -> tuple:
    """Merge K caption Gaussians N(mu_k, sigma_k^2 I) via moment matching.

    Args:
        mus: (B, K, D) per-caption means (raw, un-normalized)
        logvars: (B, K, D) per-caption log variances

    Returns:
        (combined_mu, combined_logvar) each (B, D)
    """
    combined_mu = mus.mean(dim=1)
    var = torch.exp(logvars)
    combined_var = (var + mus ** 2).mean(dim=1) - combined_mu ** 2
    combined_logvar = torch.log(combined_var.clamp(min=1e-6))
    return combined_mu, combined_logvar


class ProLIPModel(nn.Module):
    """Wrapper around the real ProLIP ViT-H/14 model.

    Exposes a CLIP-baseline-compatible interface (process_images / process_text /
    forward / save / load / trainable_parameters) so the retrieval and training
    scripts mirror the CLIP trio. ``forward`` returns the ProLIP-native feature
    dicts (normalized means, for the inclusion loss) plus flat repo-style aliases
    (img_mu / text_mu / img_logvar / text_logvar) consumed by the downstream
    experiment scripts.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        processor_path: Optional[str] = None,
        tokenizer_path: Optional[str] = None,
        freeze: bool = False,
        context_length: Optional[int] = None,
    ):
        """
        Args:
            model_path: ProLIPHF weights dir (config.PROLIP_MODEL_PATH if None)
            processor_path: CLIP image processor dir (config.PROLIP_PROCESSOR_PATH)
            tokenizer_path: HFTokenizer dir (config.PROLIP_TOKENIZER_PATH)
            freeze: freeze all backbone params (zero-shot mode)
            context_length: tokenizer context length (config.PROLIP_CONTEXT_LENGTH)
        """
        super().__init__()

        self.model_path = str(model_path or config.PROLIP_MODEL_PATH)
        self.processor_path = str(processor_path or config.PROLIP_PROCESSOR_PATH)
        self.tokenizer_path = str(tokenizer_path or config.PROLIP_TOKENIZER_PATH)
        self.context_length = context_length or config.PROLIP_CONTEXT_LENGTH
        self.freeze = freeze

        logger.info(f"Loading ProLIPHF from: {self.model_path}")
        self.prolip = ProLIPHF.from_pretrained(self.model_path)

        logger.info(f"Loading processor from: {self.processor_path}")
        self.processor = CLIPProcessor.from_pretrained(
            self.processor_path, local_files_only=True
        )

        logger.info(f"Loading tokenizer from: {self.tokenizer_path}")
        self.tokenizer = HFTokenizer(self.tokenizer_path, context_length=self.context_length)

        # Compat: transformers>=5 removed PreTrainedTokenizer.batch_encode_plus,
        # which prolip's HFTokenizer.__call__ relies on. Alias it to the modern
        # __call__ (same kwargs: return_tensors/max_length/padding/truncation).
        _tok = self.tokenizer.tokenizer
        if not hasattr(_tok, "batch_encode_plus"):
            _tok.batch_encode_plus = _tok.__call__

        if freeze:
            self.freeze_all()
            logger.info("All ProLIP parameters frozen (zero-shot mode)")

        logger.info("ProLIP model initialized")

    # ------------------------------------------------------------------ freeze
    def freeze_all(self) -> None:
        """Freeze every ProLIP parameter."""
        for param in self.prolip.parameters():
            param.requires_grad = False

    def unfreeze_all(self) -> None:
        """Unfreeze every ProLIP parameter (full fine-tuning)."""
        for param in self.prolip.parameters():
            param.requires_grad = True

    # ----------------------------------------------------------- preprocessing
    def process_images(self, images: List) -> torch.Tensor:
        """Process a list of PIL images into a pixel_values tensor on the model device."""
        device = next(self.parameters()).device
        pixel_values = self.processor(images=images, return_tensors="pt")["pixel_values"]
        return pixel_values.to(device)

    def process_text(self, texts: List[str]) -> Dict[str, torch.Tensor]:
        """Tokenize a list of captions.

        Returns a CLIP-style dict {"input_ids", "attention_mask"} for consistency
        with the rest of the repo. ProLIP's text tower handles padding internally
        (pad_id=0); ``forward`` ignores ``attention_mask`` -- it is informational.
        """
        input_ids = self.tokenizer(texts)                       # (B, context_length)
        attention_mask = (input_ids != 0).long()
        return {"input_ids": input_ids, "attention_mask": attention_mask}

    # --------------------------------------------------------------- encoders
    def encode_images(self, pixel_values: torch.Tensor, normalize: bool = False) -> Dict[str, torch.Tensor]:
        """Encode images -> {"mean", "std"} (std == log sigma^2)."""
        return self.prolip.encode_image(pixel_values, normalize=normalize)

    def encode_texts(self, input_ids: torch.Tensor, normalize: bool = False) -> Dict[str, torch.Tensor]:
        """Encode text -> {"mean", "std"} (std == log sigma^2)."""
        return self.prolip.encode_text(input_ids, normalize=normalize)

    # ----------------------------------------------------------------- forward
    def forward(
        self,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Encode image and text into probabilistic features.

        Args:
            pixel_values: (B, C, H, W)
            input_ids: (B, L) single caption, or (B, K, L) multiple captions
                       (merged via moment matching into one text Gaussian).
            attention_mask: unused (ProLIP handles padding); kept for signature
                            compatibility with the CLIP / mcdisp_align wrappers.

        Returns:
            Dict with ProLIP-native ``image_features`` / ``text_features`` dicts
            (L2-normalized means, for the inclusion loss) plus flat repo-style
            aliases (un-normalized means): img_mu / text_mu / img_logvar /
            text_logvar / img_sigma / text_sigma, and logit_scale / logit_bias.
        """
        img = self.prolip.encode_image(pixel_values, normalize=False)
        img_mu_raw, img_logvar = img["mean"], img["std"]        # std == log sigma^2

        if input_ids.dim() == 3:
            B, K, L = input_ids.shape
            text = self.prolip.encode_text(input_ids.reshape(B * K, L), normalize=False)
            cap_mu = text["mean"].view(B, K, -1)
            cap_logvar = text["std"].view(B, K, -1)
            text_mu_raw, text_logvar = merge_distributions_moment_matching(cap_mu, cap_logvar)
        else:
            text = self.prolip.encode_text(input_ids, normalize=False)
            text_mu_raw, text_logvar = text["mean"], text["std"]

        logit_scale = self.prolip.logit_scale.exp() if self.prolip.logit_scale is not None else None
        logit_bias = self.prolip.logit_bias

        return {
            # ProLIP-native (normalized means, matches upstream model.forward -> ProLIPLoss)
            "image_features": {"mean": F.normalize(img_mu_raw, dim=-1), "std": img_logvar},
            "text_features": {"mean": F.normalize(text_mu_raw, dim=-1), "std": text_logvar},
            "logit_scale": logit_scale,
            "logit_bias": logit_bias,
            # Flat repo-style aliases (un-normalized means; downstream scripts normalize)
            "img_features": img_mu_raw,
            "text_features_avg": text_mu_raw,
            "img_mu": img_mu_raw,
            "text_mu": text_mu_raw,
            "img_logvar": img_logvar,
            "text_logvar": text_logvar,
            "img_sigma": torch.exp(0.5 * img_logvar),
            "text_sigma": torch.exp(0.5 * text_logvar),
        }

    # ----------------------------------------------------------- serialization
    def trainable_parameters(self) -> List[nn.Parameter]:
        return [p for p in self.parameters() if p.requires_grad]

    def num_trainable_parameters(self) -> int:
        return sum(p.numel() for p in self.trainable_parameters())

    def save(self, path: str) -> None:
        """Save ProLIP weights (processor/tokenizer are fixed local artifacts)."""
        state = {
            "model_state_dict": self.prolip.state_dict(),
            "embed_dim": config.PROLIP_EMBED_DIM,
            "freeze": self.freeze,
        }
        torch.save(state, path)
        logger.info(f"ProLIP model saved to: {path}")

    def load(self, path: str, strict: bool = True) -> None:
        """Load ProLIP weights saved by ``save``."""
        state = torch.load(path, map_location="cpu", weights_only=False)
        self.prolip.load_state_dict(state["model_state_dict"], strict=strict)
        logger.info(f"ProLIP model loaded from: {path}")


if __name__ == "__main__":
    # Smoke test: load the model, run a tiny forward pass, print shapes.
    from utils.logger import setup_logger
    from utils.seed import set_seed

    setup_logger("prolip_model", config.LOG_DIR / "prolip_model_test.log")
    set_seed(config.SEED)

    model = ProLIPModel()
    print(f"Trainable parameters: {model.num_trainable_parameters():,}")
    print(f"Embed dim: {config.PROLIP_EMBED_DIM}")

    from PIL import Image
    dummy_images = [Image.new("RGB", (224, 224)) for _ in range(2)]
    dummy_texts = ["a photo of a cat", "a photo of a dog"]

    pixel_values = model.process_images(dummy_images)
    text_inputs = model.process_text(dummy_texts)
    print(f"pixel_values: {pixel_values.shape}")
    print(f"input_ids: {text_inputs['input_ids'].shape}")

    with torch.no_grad():
        out = model(pixel_values, text_inputs["input_ids"])
    print(f"img_mu: {out['img_mu'].shape}, text_mu: {out['text_mu'].shape}")
    print(f"img_logvar: {out['img_logvar'].shape}, text_logvar: {out['text_logvar'].shape}")
