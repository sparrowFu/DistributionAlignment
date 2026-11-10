"""
Tests for the ProLIP backbone variants (B3): the l14 checkpoint builder's key
mapping / positional-embedding splices / uncertainty init, and the variant
registry invariants. Pure-function tests only (no weights, no network) per
the tests/ convention.
"""

import math
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from tools.build_prolip_vit_l14 import build_prolip_l14_state_dict
from tools.convert_hf_clip_to_openai import convert_hf_to_openai


def _fake_hf_sd(W_v=8, W_t=768, E=768):
    """Tiny HF CLIP state dict (1 vision layer, 1 text layer, non-square
    projection where possible so orientation errors cannot stay silent."""
    def block(prefix, W):
        d = {}
        for ln in ("layer_norm1", "layer_norm2"):
            d[f"{prefix}.{ln}.weight"] = torch.randn(W)
            d[f"{prefix}.{ln}.bias"] = torch.randn(W)
        for qkv in ("q", "k", "v"):
            d[f"{prefix}.self_attn.{qkv}_proj.weight"] = torch.randn(W, W)
            d[f"{prefix}.self_attn.{qkv}_proj.bias"] = torch.randn(W)
        d[f"{prefix}.self_attn.out_proj.weight"] = torch.randn(W, W)
        d[f"{prefix}.self_attn.out_proj.bias"] = torch.randn(W)
        d[f"{prefix}.mlp.fc1.weight"] = torch.randn(W * 4, W)
        d[f"{prefix}.mlp.fc1.bias"] = torch.randn(W * 4)
        d[f"{prefix}.mlp.fc2.weight"] = torch.randn(W, W * 4)
        d[f"{prefix}.mlp.fc2.bias"] = torch.randn(W)
        return d

    torch.manual_seed(0)  # deterministic weights; text_projection made non-symmetric below
    sd = {
        "vision_model.embeddings.patch_embedding.weight": torch.randn(W_v, 3, 2, 2),
        "vision_model.embeddings.class_embedding": torch.randn(W_v),
        "vision_model.embeddings.position_embedding.weight": torch.randn(5, W_v),
        "vision_model.pre_layrnorm.weight": torch.randn(W_v),
        "vision_model.pre_layrnorm.bias": torch.randn(W_v),
        "vision_model.post_layernorm.weight": torch.randn(W_v),
        "vision_model.post_layernorm.bias": torch.randn(W_v),
        "visual_projection.weight": torch.randn(E, W_v),
        "text_model.embeddings.token_embedding.weight": torch.randn(100, W_t),
        "text_model.embeddings.position_embedding.weight": torch.randn(77, W_t),
        "text_model.final_layer_norm.weight": torch.randn(W_t),
        "text_model.final_layer_norm.bias": torch.randn(W_t),
        # deliberately non-symmetric square matrix
        "text_projection.weight": torch.tril(torch.randn(E, W_t)),
        "logit_scale": torch.tensor(4.6052),
    }
    sd.update(block("vision_model.encoder.layers.0", W_v))
    sd.update(block("text_model.encoder.layers.0", W_t))
    return sd


class TestL14BuilderMapping:
    def test_exact_key_set(self):
        hf = _fake_hf_sd()
        out = build_prolip_l14_state_dict(hf, seed=0, n_vision_layers=1, n_text_layers=1)
        expected = set()
        expected |= {"visual.conv1.weight", "visual.class_embedding",
                     "visual.ln_pre.weight", "visual.ln_pre.bias",
                     "visual.ln_post.weight", "visual.ln_post.bias", "visual.proj",
                     "visual.positional_embedding", "visual.unc_embedding",
                     "visual.uncertainty_proj.weight", "visual.uncertainty_proj.bias"}
        expected |= {"text.token_embedding.weight", "text.ln_final.weight",
                     "text.ln_final.bias", "text.positional_embedding",
                     "text.text_projection.weight", "text.text_projection.bias",
                     "text.cls_emb", "text.unc_emb",
                     "text.text_uncertainty_projection.weight",
                     "text.text_uncertainty_projection.bias"}
        expected |= {"logit_scale", "logit_bias"}
        for i in range(1):
            for s in ("ln_1.weight", "ln_1.bias", "ln_2.weight", "ln_2.bias",
                      "attn.in_proj_weight", "attn.in_proj_bias",
                      "attn.out_proj.weight", "attn.out_proj.bias",
                      "mlp.c_fc.weight", "mlp.c_fc.bias",
                      "mlp.c_proj.weight", "mlp.c_proj.bias"):
                expected.add(f"visual.transformer.resblocks.{i}.{s}")
                expected.add(f"text.transformer.resblocks.{i}.{s}")
        assert set(out.keys()) == expected

    def test_text_projection_orientation_regression(self):
        """ProLIP Linear (out, in) matches HF convention: weight AS-IS.

        The fake matrix is deliberately non-symmetric so a silent transpose
        fails this test."""
        hf = _fake_hf_sd()
        out = build_prolip_l14_state_dict(hf, seed=0, n_vision_layers=1, n_text_layers=1)
        got = out["text.text_projection.weight"]
        assert torch.equal(got, hf["text_projection.weight"])
        assert not torch.allclose(got, hf["text_projection.weight"].T)

    def test_visual_proj_transposed(self):
        hf = _fake_hf_sd()
        out = build_prolip_l14_state_dict(hf, seed=0, n_vision_layers=1, n_text_layers=1)
        assert torch.equal(out["visual.proj"], hf["visual_projection.weight"].T.contiguous())

    def test_backbone_tensors_copied(self):
        hf = _fake_hf_sd()
        out = build_prolip_l14_state_dict(hf, seed=0, n_vision_layers=1, n_text_layers=1)
        oai = convert_hf_to_openai(hf, n_vision_layers=1, n_text_layers=1)
        assert torch.equal(out["visual.conv1.weight"], hf["vision_model.embeddings.patch_embedding.weight"])
        assert torch.equal(out["text.token_embedding.weight"], hf["text_model.embeddings.token_embedding.weight"])
        assert torch.equal(out["text.ln_final.weight"], hf["text_model.final_layer_norm.weight"])
        expected_inproj = torch.cat([
            hf["vision_model.encoder.layers.0.self_attn.q_proj.weight"],
            hf["vision_model.encoder.layers.0.self_attn.k_proj.weight"],
            hf["vision_model.encoder.layers.0.self_attn.v_proj.weight"],
        ], dim=0)
        assert torch.equal(out["visual.transformer.resblocks.0.attn.in_proj_weight"], expected_inproj)

    def test_vision_pos_embed_splice(self):
        """[cls(kept), unc(new), patches(shifted by one)] -- new row at index 1."""
        hf = _fake_hf_sd()
        out = build_prolip_l14_state_dict(hf, seed=0, n_vision_layers=1, n_text_layers=1)
        old = hf["vision_model.embeddings.position_embedding.weight"]  # (5, W_v)
        pos = out["visual.positional_embedding"]
        assert pos.shape == (6, old.shape[1])
        assert torch.equal(pos[0], old[0])
        assert torch.equal(pos[2:], old[1:])
        assert not torch.equal(pos[1], old[1])
        assert pos[1].abs().max().item() < 0.1  # N(0, 0.01) row

    def test_text_pos_embed_splice(self):
        """Tower APPENDS [unc, cls]: token rows keep 0:77, two new trailing rows."""
        hf = _fake_hf_sd()
        out = build_prolip_l14_state_dict(hf, seed=0, n_vision_layers=1, n_text_layers=1)
        old = hf["text_model.embeddings.position_embedding.weight"]  # (77, W_t)
        pos = out["text.positional_embedding"]
        assert pos.shape == (79, old.shape[1])
        assert torch.equal(pos[:77], old)
        assert pos[77:].abs().max().item() < 0.1

    def test_unc_init_scheme_and_reproducibility(self):
        hf = _fake_hf_sd()
        out1 = build_prolip_l14_state_dict(hf, seed=0, n_vision_layers=1, n_text_layers=1)
        out2 = build_prolip_l14_state_dict(hf, seed=0, n_vision_layers=1, n_text_layers=1)
        out3 = build_prolip_l14_state_dict(hf, seed=1, n_vision_layers=1, n_text_layers=1)

        assert torch.equal(out1["visual.uncertainty_proj.bias"], out2["visual.uncertainty_proj.bias"])
        assert (out1["visual.uncertainty_proj.bias"] == -10.0).all()
        assert (out1["text.text_uncertainty_projection.bias"] == -10.0).all()
        assert torch.equal(out1["text.text_projection.bias"], torch.zeros(768))

        # same seed -> identical unc tensors; different seed -> different
        assert torch.equal(out1["visual.unc_embedding"], out2["visual.unc_embedding"])
        assert not torch.equal(out1["visual.unc_embedding"], out3["visual.unc_embedding"])
        # backbone untouched by the seed
        assert torch.equal(out1["visual.conv1.weight"], out3["visual.conv1.weight"])

        # logits per ProLIP's CLIP->ProLIP convention
        assert out1["logit_scale"].item() == pytest.approx(math.log(10.0))
        assert out1["logit_bias"].item() == -10.0


class TestVariantRegistry:
    def test_registry_structure(self):
        for tag, v in config.PROLIP_VARIANTS.items():
            for key in ("model_path", "processor", "processor_path", "tokenizer_path",
                        "context_length", "clean", "embed_dim", "ckpt_prefix",
                        "eval_results", "zero_shot_results", "label"):
                assert key in v, f"{tag} missing {key}"

    def test_h14_matches_legacy_constants(self):
        v = config.PROLIP_VARIANTS["h14"]
        assert v["model_path"] == config.PROLIP_MODEL_PATH
        assert v["processor_path"] == config.PROLIP_PROCESSOR_PATH
        assert v["tokenizer_path"] == config.PROLIP_TOKENIZER_PATH
        assert v["ckpt_prefix"] == "prolip"          # legacy checkpoint names
        assert v["eval_results"] == config.PROLIP_EVAL_RESULTS_PATH

    def test_checkpoint_name_composition(self):
        from utils.eval_common import resolve_checkpoint
        assert resolve_checkpoint("prolip_l14", "coco").name == "prolip_l14_coco_best.pt"
        assert resolve_checkpoint("prolip_l16", "coco").name == "prolip_l16_coco_best.pt"
        assert resolve_checkpoint("prolip", "coco").name == "prolip_coco_best.pt"

    def test_l14_matches_local_clip_backbone(self):
        v = config.PROLIP_VARIANTS["l14"]
        assert v["model_path"].name == "prolip-vit-l-14"
        assert v["processor"] == "clip"
        assert v["context_length"] == 77
        assert v["embed_dim"] == 768

    def test_l16_pins_official_config_facts(self):
        """Facts the pipeline depends on, pinned against the downloaded
        official config (SigLIP arch: 256px, ctx 64, canonicalize, 1024)."""
        v = config.PROLIP_VARIANTS["l16"]
        assert v["processor"] == "siglip"
        assert v["context_length"] == 64
        assert v["clean"] == "canonicalize"
        assert v["embed_dim"] == 1024
        assert v["tokenizer_path"].name == "prolip-siglip-tokenizer"
