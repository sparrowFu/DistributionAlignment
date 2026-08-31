"""
Tests for the MCDisp_Align warmup ramp (warmup_ramp), the loss-dict key
contract with the trainer's accumulators, and the A06 non-finite-gradient
guard. The trainer's end-to-end exports need a real model + data and are
covered by the training scripts.

warmup_ramp ramps L_var / L_dir linearly 0 -> 1 over the first warmup_frac of
the TOTAL optimizer steps; 1.0 afterwards; warmup_frac=0 disables the ramp.
"""

import sys
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from utils.mcdisp_align_trainer import train_epoch, warmup_ramp

# Every key the trainer's train_epoch accumulator requests from the loss dict
# (`totals[k] += loss_dict[k]`; "loss" maps to loss_dict["total"]); a rename on
# either side must keep these in sync (regression: KeyError 'img_var_avg' on
# the first batch, traincoco.log 2026-08-26). Four-group keys only -- the
# legacy aliases (ctr*, cal, img_var_*, u_*) are NOT consumed by the trainer.
TRAINER_TOTALS_KEYS = (
    "loss", "match", "match_i2t", "match_t2i", "mu", "var", "reg", "dir", "disp",
    "weighted_match", "weighted_mu", "weighted_var", "weighted_reg", "weighted_dir",
    "img_diag_var_mean", "img_diag_var_median", "img_diag_var_min", "img_diag_var_max",
    "img_marginal_var_mean", "img_marginal_var_median", "img_marginal_var_min",
    "img_marginal_var_max",
    "img_lowrank_var_mean",
    "text_var_mean", "cap_var_mean", "caption_spread_mean",
    "caption_spread_median", "caption_spread_max", "var_over_spread",
    "mu_mse_raw", "marginal_log_mse",
)


def test_loss_dict_covers_trainer_totals():
    from losses.mcdisp_align_losses import MCDispAlignLoss

    B, D, K, r = 2, 16, 5, 4
    crit = MCDispAlignLoss()
    _, d = crit(
        torch.randn(B, D), torch.randn(B, D), torch.randn(B, D, r),
        torch.randn(B, D), torch.randn(B, D),
        torch.randn(B, K, D), torch.randn(B, K, D),
    )
    missing = [k for k in TRAINER_TOTALS_KEYS
               if k != "loss" and k not in d]   # "loss" maps to d["total"] in the trainer
    assert not missing, f"loss dict missing trainer keys: {missing}"


def test_zero_frac_is_always_one():
    for epoch in range(3):
        for step in (0, 25, 49):
            assert warmup_ramp(epoch, step, 50, 10, 0.0) == 1.0


def test_starts_at_zero_and_reaches_one():
    spe = 50   # steps per epoch
    total = 10
    frac = 0.1  # 10% of 500 total steps = 50 warmup steps (one epoch here)
    assert warmup_ramp(0, 0, spe, total, frac) == 0.0     # very first step
    assert warmup_ramp(1, 0, spe, total, frac) == 1.0     # first step after the window
    mid = warmup_ramp(0, 25, spe, total, frac)            # mid-warmup
    assert 0.0 < mid < 1.0


def test_monotonic_within_warmup():
    spe, total, frac = 50, 10, 0.1
    prev = warmup_ramp(0, 0, spe, total, frac)
    for step in range(1, spe):
        cur = warmup_ramp(0, step, spe, total, frac)
        assert cur >= prev - 1e-9
        prev = cur


def test_capped_at_one_after_warmup():
    spe, total, frac = 50, 10, 0.1
    for epoch in range(1, total):
        for step in range(spe):
            assert warmup_ramp(epoch, step, spe, total, frac) == 1.0


class _ModeProbeModel(torch.nn.Module):
    """Minimal model exposing clip_model + head, for train/eval mode probes.

    nn.Module.train() is recursive: it puts EVERY submodule (including a
    frozen CLIP backbone) into train mode. train_epoch must reset a frozen
    backbone to eval so its features stay deterministic (dropout off). This
    is a no-op for the dropout=0.0 clip-vit-large-patch14 checkpoint but
    guards against backbones with non-zero dropout.
    """

    def __init__(self, freeze_clip: bool):
        super().__init__()
        self.clip_model = torch.nn.Dropout(0.5)
        self.head = torch.nn.Dropout(0.5)
        self.freeze_clip = freeze_clip


def _run_train_epoch(freeze_clip: bool) -> _ModeProbeModel:
    # Empty dataloader: the loop body never runs, so only the mode setup at
    # the top of train_epoch is exercised (criterion/optimizer stay untouched
    # by passing the base lambdas explicitly).
    model = _ModeProbeModel(freeze_clip)
    train_epoch(model, [], criterion=None, optimizer=None, device=None,
                epoch=0, base_lambda_var=1.0, base_lambda_dir=0.5)
    return model


def test_frozen_clip_backbone_stays_eval_in_train_epoch():
    model = _run_train_epoch(freeze_clip=True)
    assert model.clip_model.training is False, \
        "frozen CLIP backbone must stay in eval mode (deterministic features)"
    assert model.head.training is True, \
        "distribution heads must be in train mode (dropout active)"


def test_unfrozen_clip_backbone_trains_in_train_epoch():
    model = _run_train_epoch(freeze_clip=False)
    assert model.clip_model.training is True
    assert model.head.training is True


# ----------------------------------------------------------------- A06 grad guard

class _TrainProbeModel(torch.nn.Module):
    """Minimal stand-in for MCDispAlignModel over the trainer-facing surface
    (process_images/process_text/forward output keys + freeze_clip). ``body``
    gives the forward outputs a real parameter: backward() then populates
    .grad, so an (incorrectly executed) optimizer.step() would visibly move
    the weights -- that is what makes the skip assertion meaningful.
    """

    D = 6

    def __init__(self):
        super().__init__()
        self.clip_model = torch.nn.Dropout(0.5)
        self.head = torch.nn.Dropout(0.5)
        self.freeze_clip = True
        self.body = torch.nn.Linear(3 * 4 * 4, self.D)

    def process_images(self, images):
        return torch.randn(len(images), 3, 4, 4)

    def process_text(self, texts):
        n = len(texts)
        return {"input_ids": torch.randn(n, 5),
                "attention_mask": torch.ones(n, 5)}

    def forward(self, pixel_values, input_ids, attention_mask):
        B, K, _ = input_ids.shape
        D = self.D
        img_mu = self.body(pixel_values.flatten(1))          # (B, D), carries grad
        img_logvar = -3.0 + 0.1 * torch.randn(B, D)
        img_U = 0.1 * torch.randn(B, D, 2)
        text_mu = img_mu.detach() + 0.1 * torch.randn(B, D)
        text_logvar = -3.0 + 0.1 * torch.randn(B, D)
        text_mus = text_mu.unsqueeze(1) + 0.2 * torch.randn(B, K, D)
        text_logvars = -3.0 + 0.1 * torch.randn(B, K, D)
        return {
            "img_mu": img_mu, "img_logvar": img_logvar, "img_U": img_U,
            "text_mu": text_mu, "text_logvar": text_logvar,
            "text_mus": text_mus, "text_logvars": text_logvars,
        }


class _StubCriterion:
    """Callable returning a finite, parameter-connected loss and a loss dict
    covering every key train_epoch reads (totals + dir_valid/dir_total)."""

    lambda_var = 1.0
    lambda_dir = 0.5
    tau = 0.07

    def __call__(self, img_mu, *args):
        loss = 0.01 * img_mu.sum()
        d = {k: 0.5 for k in TRAINER_TOTALS_KEYS if k != "loss"}
        d.update({"total": 0.25, "dir_valid": 1, "dir_total": 1})
        return loss, d


def test_nonfinite_grad_guard_skips_step():
    """A06: a non-finite PRE-CLIP grad norm must abort the step -- grads are
    zeroed, the optimizer state stays untouched, and the batch is counted in
    nonfinite_grad_steps."""
    torch.manual_seed(0)
    model = _TrainProbeModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-1)
    weight_before = model.body.weight.detach().clone()
    batch = [{"image": ["img0", "img1"],
              "captions": [["a", "b"], ["c", "d"]]}]      # B=2, K=2

    with mock.patch("torch.nn.utils.clip_grad_norm_",
                    return_value=torch.tensor(float("nan"))):
        metrics = train_epoch(model, batch, _StubCriterion(), optimizer,
                              device="cpu", epoch=0,
                              base_lambda_var=1.0, base_lambda_dir=0.5)

    assert metrics["nonfinite_grad_steps"] == 1.0, metrics.get("nonfinite_grad_steps")
    assert torch.equal(weight_before, model.body.weight.detach()), \
        "optimizer.step() must not run on a non-finite grad norm (A06)"


def main():
    tests = [
        v for k, v in sorted(globals().items())
        if k.startswith("test_") and callable(v)
    ]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"PASS  {t.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"FAIL  {t.__name__}: {e}")
        except Exception as e:  # noqa
            failed += 1
            print(f"ERROR {t.__name__}: {type(e).__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
