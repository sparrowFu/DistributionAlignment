"""
PCME++ Model (B5)

Wrapper around the vendored official PCME++ implementation (pcmepp/ package,
github.com/naver-ai/pcmepp, MIT). The backbone is OpenAI CLIP ViT-L/14 whose
last transformer layer splits into a mean branch (pretrained weights) and an
uncertainty branch (freshly initialized, n_unc_layers=1); per-token features
(no CLIP projection, no ln_post) are pooled by Linear + GPO into
mu (l2-normalized) and log sigma^2. The wrapper exposes the project-standard
feature interface (same keys as the other baselines) so the shared
multi-caption evaluation treats PCME++ identically.

The criterion (ClosedFormSampledDistanceLoss with its learnable
shift / negative_scale) is a submodule so its parameters round-trip through
checkpoints. Training recipe (AdamP, layerwise LR decay, freeze schedule,
MSDA) lives in scripts/train_pcmepp.py + losses/pcmepp_losses.py.
"""

from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn as nn

import config
from data.pcmepp_dataset import pcmepp_eval_image_transform  # noqa: F401 (re-exported)
from pcmepp.criterions.pcmepp import ClosedFormSampledDistanceLoss
from pcmepp.models.img_encoder import ProbEncoderClipImageFull
from pcmepp.models.txt_encoder import ProbEncoderTextClip
from pcmepp.modules import clip
from utils.logger import get_logger


logger = get_logger("pcmepp_model")


class PCMEPPModel(nn.Module):
    """
    PCME++ baseline (B5): probabilistic CLIP ViT-L/14, full fine-tune.

    Submodules are named exactly ``img_enc`` / ``txt_enc`` / ``criterion``
    because losses.pcmepp_losses.build_param_groups filters parameters by
    these names (verbatim port of the official optimizer grouping).
    """

    def __init__(
        self,
        clip_model: str = "ViT-L/14",
        img_dim: Optional[int] = None,
        embed_size: Optional[int] = None,
        n_unc_layers: Optional[int] = None,
        size_augment: Optional[float] = None,
        init_shift: Optional[float] = None,
        init_negative_scale: Optional[float] = None,
        vib_beta: Optional[float] = None,
        smoothness_alpha: Optional[float] = None,
    ):
        """
        Args:
            clip_model: CLIP variant name for the vendored loader
            img_dim: backbone token width (defaults to config.PCMEPP_IMG_DIM)
            embed_size: output embedding dim (defaults to config.PCMEPP_EMBED_SIZE)
            n_unc_layers: uncertainty-branch depth (defaults to config)
            size_augment: token-level SizeAugment rate (defaults to config)
            init_shift / init_negative_scale / vib_beta / smoothness_alpha:
                criterion hyperparameters (defaults from config)
        """
        super().__init__()

        self.clip_model_name = clip_model
        self.img_dim = img_dim if img_dim is not None else config.PCMEPP_IMG_DIM
        self.embed_size = embed_size if embed_size is not None else config.PCMEPP_EMBED_SIZE
        self.n_unc_layers = (n_unc_layers if n_unc_layers is not None
                             else config.PCMEPP_N_UNC_LAYERS)

        # Point the vendored loader at the converted local OpenAI-format
        # state dict BEFORE the encoders are constructed.
        if not Path(config.PCMEPP_OPENAI_CLIP_PATH).exists():
            logger.warning(
                f"{config.PCMEPP_OPENAI_CLIP_PATH} not found. Run "
                f"tools/convert_hf_clip_to_openai.py first."
            )
        clip.set_local_weights({clip_model: str(config.PCMEPP_OPENAI_CLIP_PATH)})

        logger.info(f"Building PCME++ encoders (backbone {clip_model}, "
                    f"img_dim {self.img_dim}, embed {self.embed_size}, "
                    f"n_unc_layers {self.n_unc_layers})")
        self.img_enc = ProbEncoderClipImageFull(
            clip_model=clip_model,
            img_dim=self.img_dim,
            embed_size=self.embed_size,
            size_augment=(size_augment if size_augment is not None
                          else config.PCMEPP_IMG_SIZE_AUGMENT),
            n_unc_layers=self.n_unc_layers,
            sigma_ln_init=None,  # official COCO config: null -> Identity LN, bias 0
        )
        self.txt_enc = ProbEncoderTextClip(
            embed_size=self.embed_size,
            clip_model=clip_model,
            no_txtnorm=False,
            n_unc_layers=self.n_unc_layers,
            sigma_ln_init=None,
            bias_init=0,
        )
        self.criterion = ClosedFormSampledDistanceLoss(
            init_shift=(init_shift if init_shift is not None else config.PCMEPP_INIT_SHIFT),
            init_negative_scale=(init_negative_scale if init_negative_scale is not None
                                 else config.PCMEPP_INIT_NEGATIVE_SCALE),
            vib_beta=(vib_beta if vib_beta is not None else config.PCMEPP_VIB_BETA),
            smoothness_alpha=(smoothness_alpha if smoothness_alpha is not None
                              else config.PCMEPP_SMOOTHNESS_ALPHA),
            prob_distance=config.PCMEPP_PROB_DISTANCE,
        )

        logger.info("PCME++ model initialized "
                    f"({sum(p.numel() for p in self.parameters()):,} parameters)")

    def process_images(self, images: List) -> torch.Tensor:
        """Official eval transform over a list of PIL images -> (B, 3, 224, 224)."""
        return torch.stack([pcmepp_eval_image_transform(im) for im in images])

    def process_text(self, texts: List[str]) -> Dict[str, torch.Tensor]:
        """Tokenize without perturbation (eval path): pad to 77 zeros.

        Returns input_ids (B, 77) int64 and lengths (B,) float (true token
        counts, GPO input convention).
        """
        token_lists = clip.tokenize2(list(texts))
        input_ids = torch.zeros(len(token_lists), 77, dtype=torch.long)
        lengths = torch.zeros(len(token_lists), dtype=torch.float32)
        for i, tokens in enumerate(token_lists):
            tokens = tokens[:77]
            if tokens[-1] != clip._tokenizer.encoder["<|endoftext|>"]:
                tokens[-1] = clip._tokenizer.encoder["<|endoftext|>"]
            input_ids[i, :len(tokens)] = torch.tensor(tokens, dtype=torch.long)
            lengths[i] = len(tokens)
        return {"input_ids": input_ids, "lengths": lengths}

    def apply_epoch_freeze(self, epoch: int, img_freeze_epochs: Optional[int] = None) -> None:
        """Official freeze schedule: image backbone frozen for the first
        ``img_freeze_epochs`` epochs (the uncertainty transformer stays
        trainable), fully unfrozen afterwards. Text backbone is never frozen
        (official txt_backbone_freeze_epochs: 0)."""
        if img_freeze_epochs is None:
            img_freeze_epochs = config.PCMEPP_IMG_BACKBONE_FREEZE_EPOCHS
        if epoch < img_freeze_epochs:
            self.img_enc.freeze_backbone()
        else:
            self.img_enc.unfreeze_backbone()

    def forward(
        self,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        lengths: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            pixel_values: (B, 3, 224, 224), official normalization
            input_ids: (B, 77) or (B, K, 77) int64, zero-padded
            lengths: (B,) or (B, K) float, true token counts per row

        Returns:
            img_mu (B, D) l2-normalized, img_logvar (B, D) = log sigma^2,
            text_mu/text_logvar one row per input row (B*K, D) for 3-D input,
            plus text_mus/text_logvars (B, K, D) for 3-D input and the
            ProLIP-style image_features/text_features dicts.
        """
        was_3d = input_ids.dim() == 3
        if was_3d:
            B, K, L = input_ids.shape
            input_ids = input_ids.reshape(B * K, L)
            lengths = lengths.reshape(B * K)

        img_emb = self.img_enc(pixel_values)
        cap_emb = self.txt_enc(input_ids, lengths)

        img_mu, img_logvar = img_emb["mean"], img_emb["std"]
        txt_mu, txt_logvar = cap_emb["mean"], cap_emb["std"]

        outputs = {
            "image_features": {"mean": img_mu, "std": img_logvar},
            "text_features": {"mean": txt_mu, "std": txt_logvar},
            "img_emb": img_emb,
            "cap_emb": cap_emb,
            "img_mu": img_mu,
            "img_logvar": img_logvar,
            "img_sigma": torch.exp(0.5 * img_logvar),
            "text_mu": txt_mu,
            "text_logvar": txt_logvar,
            "text_sigma": torch.exp(0.5 * txt_logvar),
        }
        if was_3d:
            outputs["text_mus"] = txt_mu.view(B, K, -1)
            outputs["text_logvars"] = txt_logvar.view(B, K, -1)
        return outputs

    def trainable_parameters(self) -> List[nn.Parameter]:
        return [p for p in self.parameters() if p.requires_grad]

    def num_trainable_parameters(self) -> int:
        return sum(p.numel() for p in self.trainable_parameters())

    def save(self, path: str) -> None:
        """Save the FULL state dict (the backbone is fine-tuned here, unlike
        the frozen-backbone baselines) plus model meta."""
        state = {
            "model_state_dict": self.state_dict(),
            "embed_size": self.embed_size,
            "img_dim": self.img_dim,
            "n_unc_layers": self.n_unc_layers,
            "clip_model_name": self.clip_model_name,
            "backbone_excluded": False,
        }
        torch.save(state, path)
        logger.info(f"Model saved to: {path}")

    def load(self, path: str, strict: bool = True) -> None:
        """Load a checkpoint saved by save() or the train script's last.pt
        (extra training-state keys are ignored). Rebuilds the encoders when
        the checkpoint meta differs from the current construction."""
        state = torch.load(path, map_location="cpu", weights_only=False)

        rebuild = False
        for key in ("embed_size", "img_dim", "n_unc_layers"):
            if key in state and state[key] != getattr(self, key):
                logger.info(f"{key} {getattr(self, key)} -> {state[key]} "
                            f"(from checkpoint); rebuilding encoders")
                setattr(self, key, state[key])
                rebuild = True
        if rebuild:
            clip.set_local_weights({self.clip_model_name: str(config.PCMEPP_OPENAI_CLIP_PATH)})
            self.img_enc = ProbEncoderClipImageFull(
                clip_model=self.clip_model_name,
                img_dim=self.img_dim,
                embed_size=self.embed_size,
                size_augment=config.PCMEPP_IMG_SIZE_AUGMENT,
                n_unc_layers=self.n_unc_layers,
                sigma_ln_init=None,
            )
            self.txt_enc = ProbEncoderTextClip(
                embed_size=self.embed_size,
                clip_model=self.clip_model_name,
                no_txtnorm=False,
                n_unc_layers=self.n_unc_layers,
                sigma_ln_init=None,
                bias_init=0,
            )

        self.load_state_dict(state["model_state_dict"], strict=strict)
        logger.info(f"Model loaded from: {path}")


if __name__ == "__main__":
    # Forward smoke test (requires the converted OpenAI-format weights + GPU)
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))

    from utils.logger import setup_logger
    from utils.seed import set_seed

    setup_logger("pcmepp_model", config.LOG_DIR / "pcmepp_model_test.log")
    set_seed(config.SEED)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = PCMEPPModel().to(device)
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")

    B, K = 2, 5
    dummy_images = torch.randn(B, 3, 224, 224, device=device)
    texts = ["a photo of a cat", "a dog runs in the park"]
    t1 = model.process_text(texts)
    ids3 = t1["input_ids"].unsqueeze(1).repeat(1, K, 1).to(device)
    lens3 = t1["lengths"].unsqueeze(1).repeat(1, K).to(device)

    with torch.no_grad():
        out = model(dummy_images, ids3, lens3)
    for key in ("img_mu", "img_logvar", "text_mu", "text_mus"):
        print(f"{key}: {tuple(out[key].shape)}")
    assert out["img_mu"].shape == (B, config.PCMEPP_EMBED_SIZE)
    assert out["text_mus"].shape == (B, K, config.PCMEPP_EMBED_SIZE)
    norms = out["img_mu"].norm(dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-4), "img mean should be l2-normalized"
    print("Shape / norm checks passed.")
