"""
ProbVLM Model (B4)

Post-hoc probabilistic adapter (Upadhyay et al., ICCV 2023) over the
FROZEN CLIP ViT-L/14 projected embeddings: two independent BayesCap MLP
adapters (image + text) map each embedding to a per-dimension
generalized Gaussian (mu, 1/alpha, beta). Only the adapters are trained,
against the frozen CLIP features (intra-modal reconstruction +
cross-modal alignment, see losses/probvlm_losses.py).

Retrieval stays identity-preserving through mu; the per-dimension
uncertainty is converted to a Gaussian log-variance
(sigma^2 = alpha^2 * Gamma(3/beta) / Gamma(1/beta)) inside forward(), so
the shared multi-caption evaluation treats ProbVLM exactly like the
other probabilistic baselines.

The official paper used CLIP ViT-B/32 (D=512); here the adapters sit on
the project's frozen ViT-L/14 (D=768) for a backbone-matched comparison.
"""

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPModel, CLIPProcessor

import config
from losses.probvlm_losses import generalized_gaussian_logvar
from utils.logger import get_logger
from utils.image_preprocess import preprocess_images_on_gpu


logger = get_logger("probvlm_model")


class BayesCapMLP(nn.Module):
    """BayesCap MLP adapter (faithful port of the official BayesCap_MLP).

    Trunk: Linear(D, hid) - ReLU - Linear(hid, hid) - ReLU - Dropout -
    Linear(hid, D), followed by a residual add (x_intr = trunk(x) + x) and
    three parallel two-layer blocks producing (mu, 1/alpha, beta). The
    alpha/beta blocks end with ReLU so 1/alpha > 0 and beta > 0; mu is
    unconstrained. The construction loop is only exact for num_layers=3
    (the official default), hence the guard.
    """

    def __init__(
        self,
        inp_dim: int,
        out_dim: int,
        hid_dim: int = 256,
        num_layers: int = 3,
        p_drop: float = 0.1,
    ):
        super(BayesCapMLP, self).__init__()
        if num_layers != 3:
            raise ValueError(
                f"BayesCapMLP trunk is only defined for num_layers=3, got {num_layers}"
            )
        if inp_dim != out_dim:
            raise ValueError(
                "BayesCapMLP requires inp_dim == out_dim for the residual add, "
                f"got inp_dim={inp_dim}, out_dim={out_dim}"
            )
        mod = []
        for layer in range(num_layers):
            if layer == 0:
                incoming = inp_dim
                outgoing = hid_dim
                mod.append(nn.Linear(incoming, outgoing))
                mod.append(nn.ReLU())
            elif layer == num_layers // 2:
                incoming = hid_dim
                outgoing = hid_dim
                mod.append(nn.Linear(incoming, outgoing))
                mod.append(nn.ReLU())
                mod.append(nn.Dropout(p=p_drop))
            elif layer == num_layers - 1:
                incoming = hid_dim
                outgoing = out_dim
                mod.append(nn.Linear(incoming, outgoing))
        self.mod = nn.Sequential(*mod)

        self.block_mu = nn.Sequential(
            nn.Linear(out_dim, out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, out_dim),
        )

        self.block_alpha = nn.Sequential(
            nn.Linear(out_dim, out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, out_dim),
            nn.ReLU(),
        )

        self.block_beta = nn.Sequential(
            nn.Linear(out_dim, out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, out_dim),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x_intr = self.mod(x)
        x_intr = x_intr + x
        x_mu = self.block_mu(x_intr)
        x_1alpha = self.block_alpha(x_intr)
        x_beta = self.block_beta(x_intr)
        return x_mu, x_1alpha, x_beta


class ProbVLMModel(nn.Module):
    """
    ProbVLM baseline (B4): frozen CLIP + two BayesCap adapters.

    CLIP is always frozen (ProbVLM is defined on a frozen VLM); only the
    image/text adapters are trainable.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        embed_dim: Optional[int] = None,
        hidden_dim: Optional[int] = None,
        num_layers: Optional[int] = None,
        dropout: Optional[float] = None,
    ):
        """
        Args:
            model_path: Path to local CLIP model directory
                       (uses config.CLIP_VIT_L_14_PATH if None)
            embed_dim: CLIP projection dim (defaults to config.PROBVLM_EMBED_DIM)
            hidden_dim: Adapter hidden dim (defaults to config.PROBVLM_HIDDEN_DIM)
            num_layers: Adapter trunk depth (defaults to config.PROBVLM_NUM_LAYERS)
            dropout: Adapter dropout rate (defaults to config.PROBVLM_DROPOUT)
        """
        super().__init__()

        self.model_path = model_path or str(config.CLIP_VIT_L_14_PATH)
        self.embed_dim = embed_dim or config.PROBVLM_EMBED_DIM
        self.hidden_dim = hidden_dim or config.PROBVLM_HIDDEN_DIM
        self.num_layers = num_layers or config.PROBVLM_NUM_LAYERS
        self.dropout = dropout if dropout is not None else config.PROBVLM_DROPOUT

        logger.info(f"Loading CLIP model from: {self.model_path}")
        self.clip_model = CLIPModel.from_pretrained(
            self.model_path,
            local_files_only=True,
        )

        logger.info(f"Loading CLIP processor from: {self.model_path}")
        self.processor = CLIPProcessor.from_pretrained(
            self.model_path,
            local_files_only=True,
        )

        # ProbVLM is a post-hoc adapter: the backbone is always frozen.
        for param in self.clip_model.parameters():
            param.requires_grad = False
        self.clip_model.eval()
        logger.info("CLIP parameters frozen")

        # Built via _build_adapters() so load() can rebuild them to match a
        # checkpoint's meta before load_state_dict.
        self._build_adapters()

        logger.info("ProbVLM model initialized")

    def _build_adapters(self) -> None:
        self.img_bayescap = BayesCapMLP(
            self.embed_dim, self.embed_dim,
            hid_dim=self.hidden_dim, num_layers=self.num_layers,
            p_drop=self.dropout,
        )
        self.txt_bayescap = BayesCapMLP(
            self.embed_dim, self.embed_dim,
            hid_dim=self.hidden_dim, num_layers=self.num_layers,
            p_drop=self.dropout,
        )

    def process_images(
        self,
        images: List,
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
        texts: List[str],
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
            max_length=77,  # CLIP's max sequence length
        )

    def forward(
        self,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.

        Args:
            pixel_values: Image tensor (B, 3, 224, 224)
            input_ids: Token IDs, shape (B, L) or (B, K, L); a 3-D input is
                       flattened to B*K rows and encoded per caption
            attention_mask: Attention mask matching input_ids (recommended
                            for the CLIP text encoder; padded positions must
                            be masked)

        Returns:
            Dictionary containing:
                - image_features / text_features: {"mean", "std"} with the
                  mean L2-normalized and std = converted log-variance
                - img_mu / img_logvar / img_sigma: image distribution (B, D)
                - text_mu / text_logvar / text_sigma: one distribution per
                  input row (B*K, D) for 3-D input, else (B, D)
                - text_mus / text_logvars: per-caption tensors (B, K, D),
                  present only for 3-D input (no moment matching -- ProbVLM
                  has no caption-set distribution)
                - img_1alpha / img_beta / txt_1alpha / txt_beta: raw adapter
                  outputs (loss inputs)
                - i_features / t_features: frozen CLIP targets (detached)
        """
        with torch.no_grad():
            # transformers 5.x: get_*_features(...).pooler_output IS the
            # projected embedding -- do not apply the projection again.
            i_features = self.clip_model.get_image_features(pixel_values)
            i_features = i_features.pooler_output  # (B, D)

            was_3d = input_ids.dim() == 3
            if was_3d:
                B, K, L = input_ids.shape
                input_ids = input_ids.reshape(B * K, L)
                if attention_mask is not None:
                    attention_mask = attention_mask.reshape(B * K, L)

            t_features = self.clip_model.get_text_features(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
            t_features = t_features.pooler_output  # (B*, D)

        i_features = i_features.detach()
        t_features = t_features.detach()

        img_mu, img_1alpha, img_beta = self.img_bayescap(i_features)
        txt_mu, txt_1alpha, txt_beta = self.txt_bayescap(t_features)

        img_logvar = generalized_gaussian_logvar(img_1alpha, img_beta)
        txt_logvar = generalized_gaussian_logvar(txt_1alpha, txt_beta)

        img_sigma = torch.exp(0.5 * img_logvar)
        txt_sigma = torch.exp(0.5 * txt_logvar)

        outputs = {
            "image_features": {"mean": F.normalize(img_mu, dim=-1), "std": img_logvar},
            "text_features": {"mean": F.normalize(txt_mu, dim=-1), "std": txt_logvar},
            "img_mu": img_mu,
            "img_logvar": img_logvar,
            "img_sigma": img_sigma,
            "text_mu": txt_mu,
            "text_logvar": txt_logvar,
            "text_sigma": txt_sigma,
            "img_1alpha": img_1alpha,
            "img_beta": img_beta,
            "txt_1alpha": txt_1alpha,
            "txt_beta": txt_beta,
            "i_features": i_features,
            "t_features": t_features,
        }

        if was_3d:
            outputs["text_mus"] = txt_mu.view(B, K, -1)
            outputs["text_logvars"] = txt_logvar.view(B, K, -1)

        return outputs

    def trainable_parameters(self) -> List[nn.Parameter]:
        """
        Get list of trainable parameters (the two adapters; CLIP is frozen).
        """
        return [p for p in self.parameters() if p.requires_grad]

    def num_trainable_parameters(self) -> int:
        """
        Count trainable parameters.
        """
        return sum(p.numel() for p in self.trainable_parameters())

    def save(self, path: str) -> None:
        """
        Save model state. The backbone is always the untouched pretrained
        CLIP (rebuildable on load), so its weights are dropped from the
        checkpoint (~1.7GB -> a few MB).
        """
        sd = self.state_dict()
        sd = {k: v for k, v in sd.items() if not k.startswith("clip_model.")}
        state = {
            "model_state_dict": sd,
            "embed_dim": self.embed_dim,
            "hidden_dim": self.hidden_dim,
            "num_layers": self.num_layers,
            "dropout": self.dropout,
            "backbone_excluded": True,
        }
        torch.save(state, path)
        logger.info(f"Model saved to: {path}")

    def load(self, path: str, strict: bool = True) -> None:
        """
        Load model state. Rebuilds the adapters when the checkpoint meta
        (hidden_dim / num_layers / dropout) differs from the current
        construction, then loads heads-only (the pretrained CLIP already in
        self.clip_model is the excluded backbone). Extra training-state keys
        (optimizer etc., as saved in ..._last.pt) are ignored.
        """
        state = torch.load(path, map_location="cpu", weights_only=False)

        rebuild = False
        for key in ("hidden_dim", "num_layers", "dropout"):
            if key in state and state[key] != getattr(self, key):
                logger.info(f"{key} {getattr(self, key)} -> {state[key]} "
                            f"(from checkpoint); rebuilding adapters")
                setattr(self, key, state[key])
                rebuild = True
        if rebuild:
            self._build_adapters()

        if state.get("backbone_excluded"):
            # heads-only checkpoint: the pretrained CLIP weights already in
            # self.clip_model are the excluded backbone
            missing, unexpected = self.load_state_dict(state["model_state_dict"], strict=False)
            bad_missing = [k for k in missing if not k.startswith("clip_model.")]
            if bad_missing or unexpected:
                raise RuntimeError(
                    f"heads-only load mismatch: missing={bad_missing[:5]} "
                    f"unexpected={list(unexpected)[:5]}"
                )
            logger.info(f"Model loaded from: {path} (heads only; backbone from pretrained)")
        else:
            self.load_state_dict(state["model_state_dict"], strict=strict)
            logger.info(f"Model loaded from: {path}")

        if "embed_dim" in state:
            self.embed_dim = state["embed_dim"]


if __name__ == "__main__":
    # Test model (requires the local CLIP weights)
    from utils.logger import setup_logger
    from utils.seed import set_seed

    setup_logger("probvlm_model", config.LOG_DIR / "probvlm_model_test.log")
    set_seed(config.SEED)

    model = ProbVLMModel()
    print(f"Model created successfully")
    print(f"Trainable parameters: {model.num_trainable_parameters():,}")

    batch_size = 2
    num_captions = 5
    dummy_images = torch.randn(batch_size, 3, 224, 224)
    dummy_input_ids = torch.randint(0, 49408, (batch_size, num_captions, 77))
    dummy_attention_mask = torch.ones(batch_size, num_captions, 77, dtype=torch.long)

    with torch.no_grad():
        outputs = model(dummy_images, dummy_input_ids, dummy_attention_mask)

    print(f"\nOutput shapes:")
    for key, value in outputs.items():
        if isinstance(value, dict):
            print(f"  {key}: {{ {', '.join(f'{k}: {v.shape}' for k, v in value.items())} }}")
        else:
            print(f"  {key}: {value.shape}")

    assert outputs["img_mu"].shape == (batch_size, config.PROBVLM_EMBED_DIM)
    assert outputs["text_mus"].shape == (batch_size, num_captions, config.PROBVLM_EMBED_DIM)
    assert outputs["text_mu"].shape == (batch_size * num_captions, config.PROBVLM_EMBED_DIM)
    assert (outputs["img_1alpha"] >= 0).all() and (outputs["img_beta"] >= 0).all()
    assert torch.isfinite(outputs["img_logvar"]).all()
    print("\nShape / sign checks passed.")
