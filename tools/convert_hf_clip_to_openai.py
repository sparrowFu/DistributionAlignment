"""
Convert the local HF CLIP ViT-L/14 checkpoint to the OpenAI-format state dict.

The vendored PCME++ package (`pcmepp/modules/clip`) expects the original
OpenAI CLIP state dict layout (visual.*, transformer.*, ...). This tool maps
the HF `openai/clip-vit-large-patch14` weights (identical upstream weights,
different key names) onto that layout and saves a .pt file usable via
`pcmepp.modules.clip.set_local_weights({'ViT-L/14': <path>})`.

Run with --self-check to verify: builds the vanilla CLIP from the converted
dict via `build_model` and compares per-token outputs against HF
CLIPVisionModel / CLIPTextModel (cosine > 0.999 on both towers).

Usage:
    python tools/convert_hf_clip_to_openai.py [--self-check] [--output PATH]
"""

import argparse
import sys
from pathlib import Path
from typing import Dict

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

import config


def _map_block(hf_sd: Dict[str, torch.Tensor], out: Dict[str, torch.Tensor],
               hf_prefix: str, oai_prefix: str, n_layers: int) -> None:
    """Map one transformer tower (HF CLIPEncoderLayer -> OpenAI ResidualAttentionBlock)."""
    for i in range(n_layers):
        h = f"{hf_prefix}.encoder.layers.{i}"
        o = f"{oai_prefix}.resblocks.{i}"
        out[f"{o}.ln_1.weight"] = hf_sd[f"{h}.layer_norm1.weight"].float()
        out[f"{o}.ln_1.bias"] = hf_sd[f"{h}.layer_norm1.bias"].float()
        out[f"{o}.ln_2.weight"] = hf_sd[f"{h}.layer_norm2.weight"].float()
        out[f"{o}.ln_2.bias"] = hf_sd[f"{h}.layer_norm2.bias"].float()
        # nn.MultiheadAttention packs q/k/v into in_proj_{weight,bias} in that order
        out[f"{o}.attn.in_proj_weight"] = torch.cat([
            hf_sd[f"{h}.self_attn.q_proj.weight"],
            hf_sd[f"{h}.self_attn.k_proj.weight"],
            hf_sd[f"{h}.self_attn.v_proj.weight"],
        ], dim=0).float()
        out[f"{o}.attn.in_proj_bias"] = torch.cat([
            hf_sd[f"{h}.self_attn.q_proj.bias"],
            hf_sd[f"{h}.self_attn.k_proj.bias"],
            hf_sd[f"{h}.self_attn.v_proj.bias"],
        ], dim=0).float()
        out[f"{o}.attn.out_proj.weight"] = hf_sd[f"{h}.self_attn.out_proj.weight"].float()
        out[f"{o}.attn.out_proj.bias"] = hf_sd[f"{h}.self_attn.out_proj.bias"].float()
        out[f"{o}.mlp.c_fc.weight"] = hf_sd[f"{h}.mlp.fc1.weight"].float()
        out[f"{o}.mlp.c_fc.bias"] = hf_sd[f"{h}.mlp.fc1.bias"].float()
        out[f"{o}.mlp.c_proj.weight"] = hf_sd[f"{h}.mlp.fc2.weight"].float()
        out[f"{o}.mlp.c_proj.bias"] = hf_sd[f"{h}.mlp.fc2.bias"].float()


def convert_hf_to_openai(hf_sd: Dict[str, torch.Tensor],
                         n_vision_layers: int = 24, n_text_layers: int = 12
                         ) -> Dict[str, torch.Tensor]:
    """Map HF CLIPModel state dict keys onto the OpenAI CLIP layout (fp32).

    Note `vision_model.pre_layrnorm` -- the typo'd key name is real in the HF
    checkpoint and in transformers.
    """
    out = {}

    # ---- vision tower ----
    out["visual.conv1.weight"] = hf_sd["vision_model.embeddings.patch_embedding.weight"].float()
    out["visual.class_embedding"] = hf_sd["vision_model.embeddings.class_embedding"].float()
    out["visual.positional_embedding"] = hf_sd["vision_model.embeddings.position_embedding.weight"].float()
    out["visual.ln_pre.weight"] = hf_sd["vision_model.pre_layrnorm.weight"].float()
    out["visual.ln_pre.bias"] = hf_sd["vision_model.pre_layrnorm.bias"].float()
    # unused by PCME++ (ln_post/proj are skipped in ProbVisionTransformer) but required
    # by build_model's ViT detection from the state dict
    out["visual.ln_post.weight"] = hf_sd["vision_model.post_layernorm.weight"].float()
    out["visual.ln_post.bias"] = hf_sd["vision_model.post_layernorm.bias"].float()
    out["visual.proj"] = hf_sd["visual_projection.weight"].T.float().contiguous()
    _map_block(hf_sd, out, "vision_model", "visual.transformer", n_vision_layers)

    # ---- text tower ----
    out["token_embedding.weight"] = hf_sd["text_model.embeddings.token_embedding.weight"].float()
    out["positional_embedding"] = hf_sd["text_model.embeddings.position_embedding.weight"].float()
    out["ln_final.weight"] = hf_sd["text_model.final_layer_norm.weight"].float()
    out["ln_final.bias"] = hf_sd["text_model.final_layer_norm.bias"].float()
    out["text_projection"] = hf_sd["text_projection.weight"].T.float().contiguous()
    _map_block(hf_sd, out, "text_model", "transformer", n_text_layers)

    out["logit_scale"] = hf_sd["logit_scale"].float()
    return out


def load_hf_state_dict(model_dir: str) -> Dict[str, torch.Tensor]:
    model_dir = Path(model_dir)
    st_path = model_dir / "model.safetensors"
    if st_path.exists():
        from safetensors.torch import load_file
        sd = load_file(str(st_path))
    else:
        sd = torch.load(str(model_dir / "pytorch_model.bin"), map_location="cpu", weights_only=True)
    # position_ids buffers are not part of the OpenAI format
    return {k: v for k, v in sd.items() if not k.endswith("position_ids")}


def self_check(openai_sd: Dict[str, torch.Tensor], hf_dir: str) -> None:
    """Build the vanilla CLIP from the converted dict and compare both towers
    against the HF reference models on identical dummy inputs."""
    from pcmepp.modules.clip.model import build_model
    from transformers import CLIPTextModel, CLIPVisionModel

    model = build_model({k: v.clone() for k, v in openai_sd.items()},
                        "ViT-L/14", n_unc_layers=0).eval()

    # architecture assertions inferred by build_model
    assert len(model.visual.transformer.resblocks) == 24, "expected 24 vision blocks"
    assert model.visual.transformer.width == 1024, "expected vision width 1024"
    assert len(model.transformer.resblocks) == 12, "expected 12 text blocks"
    assert model.transformer.width == 768, "expected text width 768"
    assert model.vocab_size == 49408, f"expected vocab 49408, got {model.vocab_size}"

    hf_vis = CLIPVisionModel.from_pretrained(hf_dir, local_files_only=True).eval()
    hf_txt = CLIPTextModel.from_pretrained(hf_dir, local_files_only=True).eval()

    torch.manual_seed(0)
    images = torch.randn(2, 3, 224, 224)
    token_ids = torch.randint(0, 49408, (2, 77))
    token_ids[:, 0] = 49406
    token_ids[:, 20] = 49407  # eot in the middle, rest padding zeros
    attn = torch.ones_like(token_ids)

    with torch.no_grad():
        # vision: OpenAI returns pre-ln_post token features; HF last_hidden_state is the same
        ours_v = model.visual(images)["mean"].flatten(1)
        hf_v = hf_vis(pixel_values=images).last_hidden_state.flatten(1)
        # text: both apply ln_final and no projection at this point.
        # NOTE CLIP.encode_text (vanilla path, n_unc_layers=0) is broken upstream
        # (their Transformer returns a dict); replicate the tower manually as
        # CLIPTextBackbone does.
        x = model.token_embedding(token_ids) + model.positional_embedding
        x = x.permute(1, 0, 2)
        x = model.transformer(x)["mean"]
        x = x.permute(1, 0, 2)
        ours_t = model.ln_final(x).flatten(1)
        hf_t = hf_txt(input_ids=token_ids, attention_mask=attn).last_hidden_state.flatten(1)

    cos_v = torch.nn.functional.cosine_similarity(ours_v, hf_v, dim=1).min().item()
    cos_t = torch.nn.functional.cosine_similarity(ours_t, hf_t, dim=1).min().item()
    print(f"self-check cosine: vision={cos_v:.6f} text={cos_t:.6f}")
    assert cos_v > 0.999, f"vision mismatch: {cos_v}"
    assert cos_t > 0.999, f"text mismatch: {cos_t}"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=str, default=None,
                        help="Output .pt path (default: config.PCMEPP_OPENAI_CLIP_PATH)")
    parser.add_argument("--self-check", action="store_true",
                        help="Verify both towers against the HF reference models")
    args = parser.parse_args()

    hf_dir = str(config.CLIP_VIT_L_14_PATH)
    output = args.output or str(config.PCMEPP_OPENAI_CLIP_PATH)

    print(f"Loading HF checkpoint from {hf_dir}")
    hf_sd = load_hf_state_dict(hf_dir)
    print(f"Mapping {len(hf_sd)} HF tensors to the OpenAI layout")
    openai_sd = convert_hf_to_openai(hf_sd)
    print(f"Got {len(openai_sd)} OpenAI tensors")

    Path(output).parent.mkdir(parents=True, exist_ok=True)
    torch.save(openai_sd, output)
    print(f"Saved to {output}")

    if args.self_check:
        self_check(openai_sd, hf_dir)
        print("Self-check passed")


if __name__ == "__main__":
    main()
