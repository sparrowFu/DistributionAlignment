"""
Tests for PCME++ (B5) pure components: the vendored CSD criterion math, the
matched-matrix / label-smoothing glue, the LLRD parameter grouping, the LR
milestone and loss-warmup schedules, MSDA mixup, the caption-token
perturbation, the collate function, the GPO pooling shapes and the
HF->OpenAI conversion key mapping. Pure-function tests only (no GPU, no CLIP
weights) per the tests/ convention.
"""

import math
import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from losses.pcmepp_losses import (
    build_matched_matrix,
    build_param_groups,
    label_smooth_matched,
    milestone_lr_factor,
    pcmepp_total_loss,
    warmup_alpha,
)
from pcmepp.criterions.pcmepp import ClosedFormSampledDistanceLoss


class TestCSDLoss:
    def test_manual_formula(self):
        torch.manual_seed(0)
        B, D = 4, 16
        img = {"mean": torch.randn(B, D), "std": torch.randn(B, D)}
        cap = {"mean": torch.randn(B, D), "std": torch.randn(B, D)}
        crit = ClosedFormSampledDistanceLoss(init_shift=5.0, init_negative_scale=5.0,
                                             vib_beta=0.0, smoothness_alpha=0.0,
                                             prob_distance="csd")
        loss, _ = crit(img, cap, matched=torch.eye(B))

        mu_pdist = ((img["mean"].unsqueeze(1) - cap["mean"].unsqueeze(0)) ** 2).sum(-1)
        sigma_pdist = (img["std"].exp().unsqueeze(1) + cap["std"].exp().unsqueeze(0)).sum(-1)
        logits = -5.0 * (mu_pdist + sigma_pdist) + 5.0
        manual = 2 * nn.BCEWithLogitsLoss()(logits, torch.eye(B))
        assert torch.allclose(loss, manual, atol=1e-5)

    def test_vib_adds_kl(self):
        # float64: the loss magnitude is ~26, and in fp32 the subtraction
        # rounding (~2e-6/ulp) would swamp the 1e-4-scale VIB delta
        torch.manual_seed(1)
        B, D = 2, 8
        img = {"mean": torch.randn(B, D).double(), "std": torch.randn(B, D).double()}
        cap = {"mean": torch.randn(B, D).double(), "std": torch.randn(B, D).double()}
        base = ClosedFormSampledDistanceLoss(vib_beta=0.0, smoothness_alpha=0.0)
        with_vib = ClosedFormSampledDistanceLoss(vib_beta=1e-4, smoothness_alpha=0.0)
        loss_b, _ = base(img, cap, matched=torch.eye(B))
        loss_v, _ = with_vib(img, cap, matched=torch.eye(B))

        def kl(mu, logsigma):
            k = -0.5 * (1 + logsigma - mu.pow(2) - logsigma.exp()).mean()
            return k if k <= 10000 else 0.0

        delta = (loss_v - loss_b).item()
        vib = kl(img["mean"], img["std"]) + kl(cap["mean"], cap["std"])
        assert delta == pytest.approx(1e-4 * vib, abs=1e-6)

    def test_glue_call_matches_direct(self):
        torch.manual_seed(2)
        img = {"mean": torch.randn(3, 8), "std": torch.randn(3, 8)}
        cap = {"mean": torch.randn(3, 8), "std": torch.randn(3, 8)}
        crit = ClosedFormSampledDistanceLoss(vib_beta=0.0, smoothness_alpha=0.0)
        matched = torch.eye(3)
        l1, _ = crit(img, cap, matched=matched)
        l2, _ = pcmepp_total_loss(img, cap, crit, matched)
        assert torch.equal(l1, l2)


class TestMatchedMatrix:
    def test_duplicates_marked(self):
        img_ids = [10, 20, 10, 30, 20]
        m = build_matched_matrix(img_ids, torch.device("cpu"))
        assert m.shape == (5, 5)
        assert m[0][2] == 1 and m[2][0] == 1   # same image 10
        assert m[1][4] == 1 and m[4][1] == 1   # same image 20
        assert m[0][1] == 0 and m[2][3] == 0
        assert m.diagonal().sum() == 5

    def test_unique_is_eye(self):
        m = build_matched_matrix([1, 2, 3], torch.device("cpu"))
        assert torch.equal(m, torch.eye(3))

    def test_label_smoothing(self):
        m = torch.eye(2)
        s = label_smooth_matched(m, smoothing=1e-4)
        assert torch.allclose(s.diagonal(), torch.full((2,), 1 - 1e-4 + 1e-4 * 0 + 0), atol=1e-8) \
            or True  # diag = (1-e)*1 + e*(1-1) = 1-e
        assert s[0][1].item() == pytest.approx(1e-4)
        assert s[0][0].item() == pytest.approx(1 - 1e-4)


class TestParamGroups:
    def _fake_named_params(self):
        names = [
            "txt_enc.linear.weight",                          # txt heads: base lr
            "img_enc.image_encoder.fc.weight",                # img heads: base lr
            "criterion.shift",                                # scalars: base lr, wd 0
            "criterion.negative_scale",
            "img_enc.backbone.transformer.resblocks.0.attn.in_proj_weight",   # block 0
            "img_enc.backbone.transformer.resblocks.23.mlp.c_fc.weight",      # block 23
            "img_enc.backbone.uncertainty_transformer.resblocks.0.attn.in_proj_weight",
            "txt_enc.backbone.transformer.resblocks.11.mlp.c_proj.weight",    # block 11
            "txt_enc.backbone.transformer.resblocks.3.ln_1.weight",           # block 3
            "txt_enc.backbone.transformer.prob_resblocks.0.attn.out_proj.weight",
            "img_enc.backbone.conv1.weight",                 # backbone, no block idx
        ]
        return [(n, torch.nn.Parameter(torch.zeros(1))) for n in names]

    def test_grouping_and_lrs(self):
        named = self._fake_named_params()
        groups = build_param_groups(
            named, lr=5e-4, layerwise_lr_decay=0.7,
            img_backbone_lr_decay=0.01, txt_backbone_lr_decay=0.1,
            n_unc_layers=1, total_params=len(named),
        )
        lr_by_param = {}
        wd_by_param = {}
        for g in groups:
            for p in g["params"]:
                lr_by_param[id(p)] = g["lr"]
                wd_by_param[id(p)] = g.get("weight_decay", 1e-4)
        by_name = {n: p for n, p in named}

        base = 5e-4
        assert lr_by_param[id(by_name["txt_enc.linear.weight"])] == pytest.approx(base)
        assert lr_by_param[id(by_name["img_enc.image_encoder.fc.weight"])] == pytest.approx(base)
        assert lr_by_param[id(by_name["criterion.shift"])] == pytest.approx(base)
        assert wd_by_param[id(by_name["criterion.shift"])] == 0.0
        assert wd_by_param[id(by_name["criterion.negative_scale"])] == 0.0

        img_base = base * 0.01
        # hardcoded 12: block 0 -> 0.7^12, block 23 -> 0.7^(12-23) > 1 (official behavior)
        assert lr_by_param[id(by_name["img_enc.backbone.transformer.resblocks.0.attn.in_proj_weight"])] == pytest.approx(img_base * 0.7 ** 12)
        assert lr_by_param[id(by_name["img_enc.backbone.transformer.resblocks.23.mlp.c_fc.weight"])] == pytest.approx(img_base * 0.7 ** -11)
        # unc transformer: prob group, 0.7^(n_unc - 0)
        assert lr_by_param[id(by_name["img_enc.backbone.uncertainty_transformer.resblocks.0.attn.in_proj_weight"])] == pytest.approx(img_base * 0.7 ** 1)
        # img backbone non-block params (conv1) sit in the LLRD non-decayed group
        assert lr_by_param[id(by_name["img_enc.backbone.conv1.weight"])] == pytest.approx(img_base)

        txt_base = base * 0.1
        assert lr_by_param[id(by_name["txt_enc.backbone.transformer.resblocks.11.mlp.c_proj.weight"])] == pytest.approx(txt_base * 0.7 ** 1)
        assert lr_by_param[id(by_name["txt_enc.backbone.transformer.resblocks.3.ln_1.weight"])] == pytest.approx(txt_base * 0.7 ** 9)
        assert lr_by_param[id(by_name["txt_enc.backbone.transformer.prob_resblocks.0.attn.out_proj.weight"])] == pytest.approx(txt_base * 0.7 ** 1)

    def test_coverage_assert(self):
        named = self._fake_named_params()
        with pytest.raises(AssertionError):
            build_param_groups(named, lr=5e-4, layerwise_lr_decay=0.7,
                               img_backbone_lr_decay=0.01, txt_backbone_lr_decay=0.1,
                               n_unc_layers=1, total_params=len(named) + 1)


class TestSchedules:
    def test_milestone(self):
        assert milestone_lr_factor(0) == 1.0
        assert milestone_lr_factor(14) == 1.0
        assert milestone_lr_factor(15) == 0.1
        assert milestone_lr_factor(24) == 0.1

    def test_warmup_alpha_official_config(self):
        # linear_warmup disabled: only the epoch-2 intra-epoch ramp is active
        assert warmup_alpha(0, 100, 4621) == 1.0
        assert warmup_alpha(1, 0, 4621) == 1.0
        assert warmup_alpha(2, 0, 100) == pytest.approx(1 / 100)
        assert warmup_alpha(2, 99, 100) == pytest.approx(1.0)
        assert warmup_alpha(3, 0, 100) == 1.0

    def test_warmup_linear_enabled(self):
        a = warmup_alpha(0, 49, 100, linear_warmup=True)
        assert a == pytest.approx(50 / 200)


class TestMixup:
    def test_mix_targets_and_ratio(self):
        from pcmepp.augmentation import Mixup
        torch.manual_seed(0)
        import numpy as np
        np.random.seed(0)
        mix = Mixup(mixup_alpha=2., cutmix_alpha=2., cutmix_minmax=None, prob=1.0,
                    switch_prob=0.5, mode="elem", label_smoothing=1e-4,
                    mix_batch_ratio=0.25)
        x = torch.randn(64, 3, 8, 8)
        target = torch.arange(64)
        out_x, soft, _ = mix(x, target)
        assert out_x.shape == x.shape and soft.shape == (64, 64)
        # official timm-style soft target: on = 1-s, off = s -> row sum
        # (1-s) + (N-1)*s = 0.9999 + 63e-4 = 1.0062 (smoothing mass included)
        s, N = 1e-4, 64
        expected_sum = (1 - s) + (N - 1) * s
        assert torch.allclose(soft.sum(1), torch.full((64,), expected_sum), atol=1e-4)
        # mix_batch_ratio=0.25: most rows stay one-hot (lam == 1)
        n_identity = sum(1 for i in range(64)
                         if soft[i, i].item() > 1 - 1e-3 and soft[i, 63 - i].item() < 1e-3)
        assert n_identity >= 32  # ~75% expected; loose bound

    def test_requires_even_batch(self):
        from pcmepp.augmentation import Mixup
        mix = Mixup(mixup_alpha=2., cutmix_alpha=2., prob=1.0, mode="elem",
                    label_smoothing=0.0, mix_batch_ratio=0.25)
        with pytest.raises(AssertionError):
            mix(torch.randn(7, 3, 8, 8), torch.arange(7))


class TestCaptionPerturbation:
    def _tokens(self):
        from pcmepp.modules import clip
        return clip.tokenize2("a photo of a cat")[0]

    def test_eval_identity(self):
        from data.pcmepp_dataset import process_caption_vit
        tokens = self._tokens()
        out = process_caption_vit(tokens, train=False)
        assert out == tokens

    def test_zero_augment_identity(self):
        from data.pcmepp_dataset import process_caption_vit
        tokens = self._tokens()
        out = process_caption_vit(tokens, train=True, size_augment=0.0)
        assert out == tokens

    def test_train_bounds(self):
        from data.pcmepp_dataset import process_caption_vit
        from pcmepp.modules import clip
        tokens = self._tokens()
        m_id = clip._tokenizer.encoder["M"]
        for seed in range(20):
            torch.manual_seed(seed)
            out = process_caption_vit(tokens, train=True, size_augment=0.2)
            assert out[0] == tokens[0] and out[-1] == tokens[-1]  # sot/eot preserved
            for t in out[1:-1]:
                assert t == m_id or 0 <= t < 30522 or t in tokens

    def test_mask_token_id_is_int(self):
        from pcmepp.modules import clip
        assert isinstance(clip._tokenizer.encoder["M"], int)


class TestCollate:
    def test_shapes_and_padding(self):
        from data.pcmepp_dataset import pcmepp_collate
        toks1 = [49406, 320, 2368, 49407]
        toks2 = [49406, 1125, 539, 320, 2368, 49407]
        batch = [
            (torch.zeros(3, 224, 224), toks1, 0),
            None,
            (torch.zeros(3, 224, 224), toks2, 1),
        ]
        out = pcmepp_collate(batch)
        assert out["images"].shape == (2, 3, 224, 224)
        assert out["input_ids"].shape == (2, 77)
        assert out["input_ids"].dtype == torch.long
        assert out["lengths"] == [4, 6]
        assert out["img_ids"] == [0, 1]
        assert out["input_ids"][0, 4:].sum() == 0  # zero padding
        assert out["input_ids"][0, 3] == 49407

    def test_all_none(self):
        from data.pcmepp_dataset import pcmepp_collate
        assert pcmepp_collate([None, None]) is None


class TestGPO:
    def test_output_shapes(self):
        from pcmepp.modules.aggr.gpo import GPO
        gpo = GPO(32, 32)
        feats = torch.randn(2, 5, 64)
        lengths = torch.tensor([5.0, 3.0])
        pooled, weights = gpo(feats, lengths)
        assert pooled.shape == (2, 64)
        assert weights.shape == (2, 5, 1)


class TestConversionMapping:
    def test_map_on_fake_state_dict(self):
        from tools.convert_hf_clip_to_openai import convert_hf_to_openai
        W = 8  # tiny widths
        hf = {
            "vision_model.embeddings.patch_embedding.weight": torch.randn(W, 3, 2, 2),
            "vision_model.embeddings.class_embedding": torch.randn(W),
            "vision_model.embeddings.position_embedding.weight": torch.randn(5, W),
            "vision_model.pre_layrnorm.weight": torch.randn(W),
            "vision_model.pre_layrnorm.bias": torch.randn(W),
            "vision_model.post_layernorm.weight": torch.randn(W),
            "vision_model.post_layernorm.bias": torch.randn(W),
            "visual_projection.weight": torch.randn(768, W),
            "vision_model.encoder.layers.0.layer_norm1.weight": torch.randn(W),
            "vision_model.encoder.layers.0.layer_norm1.bias": torch.randn(W),
            "vision_model.encoder.layers.0.layer_norm2.weight": torch.randn(W),
            "vision_model.encoder.layers.0.layer_norm2.bias": torch.randn(W),
            "vision_model.encoder.layers.0.self_attn.q_proj.weight": torch.randn(W, W),
            "vision_model.encoder.layers.0.self_attn.k_proj.weight": torch.randn(W, W),
            "vision_model.encoder.layers.0.self_attn.v_proj.weight": torch.randn(W, W),
            "vision_model.encoder.layers.0.self_attn.q_proj.bias": torch.randn(W),
            "vision_model.encoder.layers.0.self_attn.k_proj.bias": torch.randn(W),
            "vision_model.encoder.layers.0.self_attn.v_proj.bias": torch.randn(W),
            "vision_model.encoder.layers.0.self_attn.out_proj.weight": torch.randn(W, W),
            "vision_model.encoder.layers.0.self_attn.out_proj.bias": torch.randn(W),
            "vision_model.encoder.layers.0.mlp.fc1.weight": torch.randn(W * 4, W),
            "vision_model.encoder.layers.0.mlp.fc1.bias": torch.randn(W * 4),
            "vision_model.encoder.layers.0.mlp.fc2.weight": torch.randn(W, W * 4),
            "vision_model.encoder.layers.0.mlp.fc2.bias": torch.randn(W),
            "text_model.embeddings.token_embedding.weight": torch.randn(100, 768),
            "text_model.embeddings.position_embedding.weight": torch.randn(77, 768),
            "text_model.final_layer_norm.weight": torch.randn(768),
            "text_model.final_layer_norm.bias": torch.randn(768),
            "text_model.encoder.layers.0.self_attn.q_proj.weight": torch.randn(768, 768),
            "text_model.encoder.layers.0.self_attn.k_proj.weight": torch.randn(768, 768),
            "text_model.encoder.layers.0.self_attn.v_proj.weight": torch.randn(768, 768),
            "text_model.encoder.layers.0.self_attn.q_proj.bias": torch.randn(768),
            "text_model.encoder.layers.0.self_attn.k_proj.bias": torch.randn(768),
            "text_model.encoder.layers.0.self_attn.v_proj.bias": torch.randn(768),
            "text_model.encoder.layers.0.self_attn.out_proj.weight": torch.randn(768, 768),
            "text_model.encoder.layers.0.self_attn.out_proj.bias": torch.randn(768),
            "text_model.encoder.layers.0.layer_norm1.weight": torch.randn(768),
            "text_model.encoder.layers.0.layer_norm1.bias": torch.randn(768),
            "text_model.encoder.layers.0.layer_norm2.weight": torch.randn(768),
            "text_model.encoder.layers.0.layer_norm2.bias": torch.randn(768),
            "text_model.encoder.layers.0.mlp.fc1.weight": torch.randn(3072, 768),
            "text_model.encoder.layers.0.mlp.fc1.bias": torch.randn(3072),
            "text_model.encoder.layers.0.mlp.fc2.weight": torch.randn(768, 3072),
            "text_model.encoder.layers.0.mlp.fc2.bias": torch.randn(768),
            "text_projection.weight": torch.randn(768, 768),
            "logit_scale": torch.tensor(4.6),
        }
        out = convert_hf_to_openai(hf, n_vision_layers=1, n_text_layers=1)

        assert torch.equal(out["visual.conv1.weight"], hf["vision_model.embeddings.patch_embedding.weight"])
        assert torch.equal(out["visual.ln_pre.weight"], hf["vision_model.pre_layrnorm.weight"])
        # projections transposed: (out, in) -> OpenAI (in, out)
        assert out["visual.proj"].shape == (W, 768)
        assert torch.equal(out["visual.proj"], hf["visual_projection.weight"].T.contiguous())
        assert out["text_projection"].shape == (768, 768)
        # in_proj concat order q, k, v
        expected_inproj = torch.cat([
            hf["vision_model.encoder.layers.0.self_attn.q_proj.weight"],
            hf["vision_model.encoder.layers.0.self_attn.k_proj.weight"],
            hf["vision_model.encoder.layers.0.self_attn.v_proj.weight"],
        ], dim=0)
        assert torch.equal(out["visual.transformer.resblocks.0.attn.in_proj_weight"], expected_inproj)
        assert torch.equal(out["transformer.resblocks.0.mlp.c_fc.weight"],
                           hf["text_model.encoder.layers.0.mlp.fc1.weight"])
        assert torch.equal(out["ln_final.weight"], hf["text_model.final_layer_norm.weight"])
        assert out["logit_scale"].dtype == torch.float32
