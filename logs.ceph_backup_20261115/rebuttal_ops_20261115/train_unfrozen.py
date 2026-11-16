"""Run the normal trainer with a one-step backbone-gradient/update audit.

The audit changes no training tensor, loss, RNG state, or optimizer setting.
"""
import json
from pathlib import Path
import runpy
import sys

sys.path.insert(0, str(Path.cwd()))
import torch
import utils.mcdisp_align_trainer as trainer

original_create_optimizer = trainer.create_optimizer


def checked_optimizer(model, freeze_clip, clip_lr, mlp_lr, weight_decay):
    assert freeze_clip is False and model.freeze_clip is False
    backbone = model.clip_model
    assert all(p.requires_grad for p in backbone.parameters())
    opt = original_create_optimizer(model, freeze_clip, clip_lr, mlp_lr, weight_decay)
    bb_ids = {id(p) for p in backbone.parameters()}
    opt_ids = {id(p) for g in opt.param_groups for p in g["params"]}
    assert bb_ids <= opt_ids
    print("BACKBONE_OPTIMIZER_AUDIT " + json.dumps({
        "freeze_clip": freeze_clip,
        "trainable_backbone_parameters": sum(p.numel() for p in backbone.parameters()),
        "backbone_lr": clip_lr, "head_lr": mlp_lr,
        "all_backbone_parameters_in_optimizer": True,
    }), flush=True)
    projections = {name: getattr(backbone, name).weight
                   for name in ("visual_projection", "text_projection")}
    saved = {}
    audited = {"done": False}

    def before_step(optimizer, args, kwargs):
        if audited["done"]:
            return
        report = {}
        for name, p in projections.items():
            assert p.grad is not None and bool(torch.isfinite(p.grad).all()), name
            maximum = p.grad.detach().abs().max().item()
            assert maximum > 0, (name, "backbone gradient is zero")
            report[name] = {"max_abs_gradient_after_clip": maximum}
            saved[name] = p.detach().clone()
        print("BACKBONE_GRADIENT_AUDIT " + json.dumps(report), flush=True)

    def after_step(optimizer, args, kwargs):
        if audited["done"]:
            return
        report = {}
        for name, p in projections.items():
            delta = (p.detach() - saved[name]).abs().max().item()
            report[name] = {"max_abs_weight_update": delta}
            if any(g["lr"] > 0 and any(id(q) == id(p) for q in g["params"])
                   for g in optimizer.param_groups):
                assert delta > 0, (name, "positive LR but no representable weight update")
        print("BACKBONE_WEIGHT_UPDATE_AUDIT " + json.dumps(report), flush=True)
        saved.clear()
        audited["done"] = True

    opt.register_step_pre_hook(before_step)
    opt.register_step_post_hook(after_step)
    return opt


trainer.create_optimizer = checked_optimizer
sys.argv[0] = "scripts/ablation_study/run.py"
runpy.run_path(sys.argv[0], run_name="__main__")
