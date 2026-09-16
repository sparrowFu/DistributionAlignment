#!/usr/bin/env python
"""Path-1c: distribution-overlap scoring (user proposal, untested before).

Uses BOTH Gaussians: image N(mu_v, diag+U) and single-caption N(mu_t, diag).
Zero training, no caption grouping read. Variants:

  V0   cos(mu_v, mu_t)                                     baseline
  O2   V0 - lam * E-quad/D,  E-quad = q(i,t) + tr(Sigma_v^-1 Sigma_t)/D
       (expected Mahalanobis of the caption DISTRIBUTION under the image
        Gaussian -- closed form; the trace term is the new cross-term V4
        lacked)
  O1   V0 + lam * logBC/D,  Bhattacharyya coefficient between diagonal
       approximations (image low-rank folded into its diagonal)
  OP2  pure O2 (no cosine), reference only

Lambdas selected on the dev half. MCDisp-only (needs text logvars).
"""

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

import config
from utils.eval_common import build_eval_dataloader
from utils.logger import get_logger
from utils.seed import set_seed
from utils.mcdisp_coverage_score import prepare_image_covariance, _quad_per_dim
from scripts.score_variant_sweep import extract, recalls_from_scores

logger = get_logger("score_sweep3", config.LOG_DIR / "score_sweep3.log")

LAMBDAS = [0.02, 0.05, 0.15, 0.5, 1.5]


def overlap_terms(f, device):
    """(q, trace_term, logBC) per (image, caption) — chunked."""
    img_mu, cap_mus = f["img_mu"].to(device), f["text_mus"].to(device)
    t_lv = f["text_logvars"].to(device)                      # (N, K, D)
    N, K, D = cap_mus.shape
    cap_mu = cap_mus.reshape(N * K, D)
    cap_lv = t_lv.reshape(N * K, D)
    state = prepare_image_covariance(f["img_logvar"].to(device),
                                     None if f["img_U"] is None else f["img_U"].to(device))
    inv_v = state["inv_var"]                                 # (N, D) = A^-1 diag
    q = torch.empty(N, N * K, device=device)
    tr = torch.empty(N, N * K, device=device)
    logbc = torch.empty(N, N * K, device=device)
    if f["img_U"] is not None:
        diag_total = torch.exp(f["img_logvar"].to(device)) + (f["img_U"].to(device) ** 2).sum(-1)
    else:
        diag_total = torch.exp(f["img_logvar"].to(device))
    for i0 in range(0, N, 256):
        i1 = min(i0 + 256, N)
        for c0 in range(0, N * K, 4096):
            c1 = min(c0 + 4096, N * K)
            delta = cap_mu[c0:c1].unsqueeze(0) - img_mu[i0:i1].unsqueeze(1)   # (b, m, D)
            q[i0:i1, c0:c1] = _quad_per_dim(
                delta, state["inv_var"][i0:i1].unsqueeze(1),
                None if state["U_scaled"] is None else state["U_scaled"][i0:i1].unsqueeze(1),
                state["chol"][i0:i1].unsqueeze(1) if state["chol"] is not None else None)
            # trace term: tr(A^-1 Sigma_t)/D  (diagonal part of Sigma_v^-1
            # against the caption's diagonal; low-rank correction of the
            # inverse via b^T G^-1 b with b = U^T A^-1 Sigma_t^{1/2} is folded
            # approximately by using the full diag_total in its place)
            tr[i0:i1, c0:c1] = (torch.exp(cap_lv[c0:c1]).unsqueeze(0)
                                / diag_total[i0:i1].unsqueeze(1)).sum(-1) / D
            # Bhattacharyya (diagonal approx, per-dim averaged)
            s1 = diag_total[i0:i1].unsqueeze(1)                               # (b, 1, D)
            s2 = torch.exp(cap_lv[c0:c1]).unsqueeze(0)                        # (1, m, D)
            lb = (0.25 * torch.log(s1 * s2) - 0.5 * torch.log((s1 + s2) / 2)
                  - delta ** 2 / (4 * (s1 + s2))).sum(-1) / D
            logbc[i0:i1, c0:c1] = lb
    return q, tr, logbc


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--num-samples", type=int, default=5000)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()
    set_seed(config.SEED)

    jobs = [
        ("mcdisp_kl_unfrozen_s42", "checkpoints/unfrozen_seed42/mcdisp_align_kl_coco_best.pt"),
        ("mcdisp_kl_frozen_s42", "checkpoints/seed42/mcdisp_align_kl_coco_best.pt"),
    ]
    from models.mcdisp_align_model import MCDispAlignModel
    loader, _ = build_eval_dataloader("coco", batch_size=args.batch_size,
                                      num_workers=4, num_samples=args.num_samples)
    results = {}
    for key, ckpt in jobs:
        model = MCDispAlignModel()
        model.load(ckpt)
        model = model.to(args.device).eval()
        f = extract(model, "mcdisp", loader, args.device, args.num_samples)
        del model
        torch.cuda.empty_cache()

        img_mu = f["img_mu"].to(args.device)
        N, K = img_mu.shape[0], f["text_mus"].shape[1]
        cap_mu = f["text_mus"].to(args.device).reshape(N * K, -1)
        C = F.normalize(img_mu, dim=-1) @ F.normalize(cap_mu, dim=-1).T
        q, tr, logbc = overlap_terms(f, args.device)
        eq = q + tr

        variants = {"V0": C.clone()}
        for lam in LAMBDAS:
            variants[f"O2:lam={lam}"] = C - lam * eq
            variants[f"O1:lam={lam}"] = C + lam * logbc
        variants["OP2:pure"] = -eq
        del q, tr, logbc, eq
        torch.cuda.empty_cache()

        dev_idx = torch.arange(0, N, 2, device=args.device)
        test_idx = torch.arange(1, N, 2, device=args.device)

        def _cols(idx):
            return (idx * K).repeat_interleave(K) + torch.arange(K, device=args.device).repeat(len(idx))

        report, best = {}, None
        for name, S in variants.items():
            r_dev = recalls_from_scores(S[dev_idx][:, _cols(dev_idx)], len(dev_idx), K, args.device)
            r_test = recalls_from_scores(S[test_idx][:, _cols(test_idx)], len(test_idx), K, args.device)
            del S
            report[name] = {"dev_mr": r_dev["mr"], "test": r_test}
            if best is None or r_dev["mr"] > best[1]:
                best = (name, r_dev["mr"], r_test)
        results[key] = {"variants": report, "dev_selected": {"name": best[0], "test": best[2]}}
        logger.info(f"{key}: dev-selected {best[0]} -> test mr={best[2]['mr']:.4f} "
                    f"(V0 {report['V0']['test']['mr']:.4f})")
        del variants, C
        torch.cuda.empty_cache()

    out = Path("outputs/score_sweep")
    (out / "results_overlap.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
