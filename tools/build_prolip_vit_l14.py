"""
Build a ProLIP-format ViT-L/14 checkpoint from the local OpenAI CLIP ViT-L/14.

The official ProLIP release has no ViT-L/14 weights (L/16 and L/14 are
architecturally non-convertible: different patch-conv kernels and position
grids). This tool assembles one instead: the OpenAI CLIP ViT-L/14 backbone
(the exact initialization used by Ours/B1/B2/B4/B5) is mapped onto ProLIP's
state-dict layout, and ProLIP's uncertainty parts (unc embeddings / unc
projections / cls+unc positional rows) are randomly initialized with ProLIP's
own init scheme (bias -10 etc., fixed seed). The result carries NO ProLIP
1.28B-pair probabilistic pretraining -- label runs honestly as
"ProLIP-objective fine-tune from CLIP init".

Usage:
    python tools/build_prolip_vit_l14.py [--self-check] [--output-dir PATH] [--seed 0]

Writes config.json + model.safetensors into the output dir (default:
/mnt/disk1/xpfu/PreTrainedModels/prolip-vit-l-16-style path, symlinked from
PreTrainedModels/prolip-vit-l-14).
"""

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

import config
from convert_hf_clip_to_openai import convert_hf_to_openai, load_hf_state_dict


# ProLIP architecture config for the OpenAI CLIP ViT-L/14 backbone.
# quick_gelu TRUE (OpenAI CLIP); contrast the official L/16 SigLIP config
# which omits it (nn.GELU).
PROLIP_L14_CONFIG = {
    "embed_dim": 768,
    "init_logit_bias": -10,
    "quick_gelu": True,
    "text_cfg": {
        "context_length": 77,
        "embed_cls": True,
        "embed_unc": True,
        "heads": 12,
        "init_unc_bias": -10,
        "layers": 12,
        "no_causal_mask": True,
        "pool_type": "last",   # vestigial: embed_cls/unc override pooling
        "proj_bias": True,
        "vocab_size": 49408,
        "width": 768,
    },
    "vision_cfg": {
        "embed_unc": True,
        "head_width": 64,
        "image_size": 224,
        "init_unc_bias": -10,
        "layers": 24,
        "patch_size": 14,
        "width": 1024,
    },
}


def build_prolip_l14_state_dict(hf_sd: Dict[str, torch.Tensor], seed: int = 0,
                                n_vision_layers: int = 24,
                                n_text_layers: int = 12
                                ) -> Dict[str, torch.Tensor]:
    """Assemble the ProLIP L/14 state dict from an HF CLIP state dict.

    Backbone keys reuse tools/convert_hf_clip_to_openai.convert_hf_to_openai
    (identical OpenAI layout). Two deliberate orientation rules:
      - text.text_projection.weight = HF weight AS-IS (ProLIP's nn.Linear
        (out, in) matches HF's convention; the converter's transposed OpenAI
        Parameter must NOT be reused here -- the matrix is square, so a wrong
        orientation would be silent).
      - visual.proj = converter output (= HF visual_projection.weight.T),
        matching ProLIP's ``pooled @ proj`` usage.
    Uncertainty parts use ProLIP's init scheme with a fixed generator.
    """
    oai = convert_hf_to_openai(hf_sd, n_vision_layers=n_vision_layers,
                               n_text_layers=n_text_layers)
    g = torch.Generator().manual_seed(seed)

    vision_width = hf_sd["vision_model.embeddings.patch_embedding.weight"].shape[0]
    text_width = hf_sd["text_model.embeddings.token_embedding.weight"].shape[1]
    embed_dim = hf_sd["text_projection.weight"].shape[0]

    out: Dict[str, torch.Tensor] = {}

    # ---- vision backbone (names identical to the converter output) ----
    for key in ("visual.conv1.weight", "visual.class_embedding",
                "visual.ln_pre.weight", "visual.ln_pre.bias",
                "visual.ln_post.weight", "visual.ln_post.bias", "visual.proj"):
        out[key] = oai[key].clone()
    for i in range(n_vision_layers):
        for suffix in ("ln_1.weight", "ln_1.bias", "ln_2.weight", "ln_2.bias",
                       "attn.in_proj_weight", "attn.in_proj_bias",
                       "attn.out_proj.weight", "attn.out_proj.bias",
                       "mlp.c_fc.weight", "mlp.c_fc.bias",
                       "mlp.c_proj.weight", "mlp.c_proj.bias"):
            key = f"visual.transformer.resblocks.{i}.{suffix}"
            out[key] = oai[key].clone()

    # vision positional embedding: OpenAI [cls, patches] (grid^2+1) ->
    # ProLIP [cls, unc, patches] (grid^2+2): insert the new unc row at index 1.
    old_pos = oai["visual.positional_embedding"]
    unc_row = torch.empty(1, vision_width).normal_(0.0, 0.01, generator=g)
    out["visual.positional_embedding"] = torch.cat(
        [old_pos[:1], unc_row, old_pos[1:]], dim=0).float().contiguous()

    # ---- vision uncertainty parts (ProLIP init scheme) ----
    out["visual.unc_embedding"] = torch.empty(vision_width).normal_(
        0.0, 0.01, generator=g)
    out["visual.uncertainty_proj.weight"] = torch.empty(
        embed_dim, vision_width).normal_(0.0, vision_width ** -0.5, generator=g)
    out["visual.uncertainty_proj.bias"] = torch.full((embed_dim,), -10.0)

    # ---- text backbone (converter text keys gain the text. prefix) ----
    out["text.token_embedding.weight"] = oai["token_embedding.weight"].clone()
    out["text.ln_final.weight"] = oai["ln_final.weight"].clone()
    out["text.ln_final.bias"] = oai["ln_final.bias"].clone()
    for i in range(n_text_layers):
        for suffix in ("ln_1.weight", "ln_1.bias", "ln_2.weight", "ln_2.bias",
                       "attn.in_proj_weight", "attn.in_proj_bias",
                       "attn.out_proj.weight", "attn.out_proj.bias",
                       "mlp.c_fc.weight", "mlp.c_fc.bias",
                       "mlp.c_proj.weight", "mlp.c_proj.bias"):
            key = f"transformer.resblocks.{i}.{suffix}"
            out[f"text.{key}"] = oai[key].clone()

    # text positional embedding: OpenAI (context, width) -> ProLIP
    # (context+2, width); the tower APPENDS [unc, cls] tokens, which read the
    # two new trailing rows.
    old_tpos = oai["positional_embedding"]
    extra_rows = torch.empty(2, text_width).normal_(0.0, 0.01, generator=g)
    out["text.positional_embedding"] = torch.cat(
        [old_tpos, extra_rows], dim=0).float().contiguous()

    # text projection: Linear WITH bias. HF weight AS-IS (orientation trap:
    # NOT the converter's transposed OpenAI parameter).
    out["text.text_projection.weight"] = hf_sd["text_projection.weight"].float().contiguous()
    out["text.text_projection.bias"] = torch.zeros(embed_dim)

    # ---- text uncertainty parts (ProLIP init scheme) ----
    out["text.cls_emb"] = torch.empty(text_width).normal_(0.0, 0.01, generator=g)
    out["text.unc_emb"] = torch.empty(text_width).normal_(0.0, 0.01, generator=g)
    out["text.text_uncertainty_projection.weight"] = torch.empty(
        embed_dim, text_width).normal_(0.0, text_width ** -0.5, generator=g)
    out["text.text_uncertainty_projection.bias"] = torch.full((embed_dim,), -10.0)

    # ---- logits (ProLIP's own CLIP->ProLIP convention, prolip/factory.py) ----
    out["logit_scale"] = torch.tensor(math.log(10.0))
    out["logit_bias"] = torch.tensor(-10.0)

    return out


def self_check(out_dir: str, semantic: bool = True) -> None:
    """Hard checks: ProLIPHF strict reload + dummy forward. Soft: real COCO
    matched-vs-mismatched cosine on the mean path (warning-only)."""
    from prolip.model import ProLIPHF

    model = ProLIPHF.from_pretrained(out_dir)
    model.eval()

    torch.manual_seed(0)
    images = torch.randn(2, 3, 224, 224)
    token_ids = torch.zeros(2, 77, dtype=torch.long)
    token_ids[:, 0] = 49406            # sot
    token_ids[:, 10] = 49407           # eot

    with torch.no_grad():
        img = model.encode_image(images)
        txt = model.encode_text(token_ids)

    assert img["mean"].shape == (2, 768), img["mean"].shape
    assert txt["mean"].shape == (2, 768), txt["mean"].shape
    assert torch.isfinite(img["mean"]).all() and torch.isfinite(txt["mean"]).all()
    # unc heads are bias-initialized at -10 -> std ~ -10
    assert img["std"].mean().item() < -9.0, img["std"].mean().item()
    assert txt["std"].mean().item() < -9.0, txt["std"].mean().item()
    print(f"forward check OK: mean (2,768), img_std={img['std'].mean().item():.3f}, "
          f"txt_std={txt['std'].mean().item():.3f}")

    if not semantic:
        return

    import torch.nn.functional as F
    from data.caption_dataset import ImageCaptionDataset
    from models.prolip_model import ProLIPModel

    ds = ImageCaptionDataset(
        captions_path=config.CAPTIONS_PATH,
        images_dir=config.IMAGES_DIR,
        num_captions=1,
    )
    pairs = [ds[i] for i in range(16)]
    pairs = [p for p in pairs if p is not None][:8]
    if len(pairs) < 4:
        print("semantic check skipped (not enough loaded samples)")
        return

    images = [p["image"] for p in pairs]
    captions = [p["captions"][0] for p in pairs]
    wrapper = ProLIPModel(backbone="l14", model_path=out_dir, freeze=True)
    with torch.no_grad():
        pixel_values = wrapper.process_images(images)   # the model's own CLIP processor
        img_f = F.normalize(wrapper.prolip.encode_image(pixel_values)["mean"], dim=-1)
        txt_f = F.normalize(wrapper.prolip.encode_text(wrapper.tokenizer(captions))["mean"], dim=-1)
    sims = img_f @ txt_f.T
    matched = sims.diag().mean().item()
    mismatched = (sims.sum() - sims.diag().sum()).item() / (len(pairs) * (len(pairs) - 1))
    print(f"semantic check: matched cosine={matched:.4f}, mismatched={mismatched:.4f}")
    if matched <= mismatched:
        print("WARNING: matched pairs are NOT above mismatched pairs on the mean path")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output dir (default: /mnt/disk1/xpfu/PreTrainedModels/"
                             "prolip-vit-l-14, symlinked from PreTrainedModels/)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--self-check", action="store_true",
                        help="Reload the built checkpoint and run forward checks")
    parser.add_argument("--no-semantic", action="store_true",
                        help="Skip the real-COCO soft semantic check")
    args = parser.parse_args()

    out_dir = args.output_dir
    if out_dir is None:
        host_dir = Path("/mnt/disk1/xpfu/PreTrainedModels/prolip-vit-l-14")
        out_dir = host_dir
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    hf_sd = load_hf_state_dict(str(config.CLIP_VIT_L_14_PATH))
    print(f"Loaded {len(hf_sd)} HF tensors from {config.CLIP_VIT_L_14_PATH}")
    sd = build_prolip_l14_state_dict(hf_sd, seed=args.seed)
    print(f"Assembled {len(sd)} ProLIP tensors")

    from safetensors.torch import save_file
    save_file({k: v.contiguous() for k, v in sd.items()},
              str(out_dir / "model.safetensors"))
    with open(out_dir / "config.json", "w") as f:
        json.dump(PROLIP_L14_CONFIG, f, indent=2)
    print(f"Wrote {out_dir/'config.json'} and {out_dir/'model.safetensors'}")

    # project symlink (config.PROLIP_VARIANTS['l14']['model_path'])
    link = config.PROLIP_VARIANTS["l14"]["model_path"]
    if args.output_dir is None and not link.exists():
        link.symlink_to(out_dir)
        print(f"Symlinked {link} -> {out_dir}")

    if args.self_check:
        self_check(str(out_dir), semantic=not args.no_semantic)
        print("Self-check passed")


if __name__ == "__main__":
    main()
