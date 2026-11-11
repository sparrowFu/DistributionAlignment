"""Read-only source/log audit and NumPy mechanism checks (no model weights).

Run from any directory. Writes only audit_results.json next to this script.
The synthetic checks establish available shortcuts, not empirical causality.
"""
import ast
import json
import re
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def parse_runs():
    runs = {}
    current = None
    for line in (ROOT / "logs/train_mcdisp_align.log").read_text(encoding="utf-8").splitlines():
        m = re.search(r"Dataset: (\w+) \| Epochs:", line)
        if m:
            current = runs.setdefault(m[1], {"epochs": []})
        if current is None:
            continue
        m = re.search(r"Epoch (\d+)/\d+ - Train Loss: ([\d.]+), NCE: ([\d.]+).*mR\(mc\): ([\d.]+)", line)
        if m:
            current["epochs"].append(dict(epoch=int(m[1]), train_loss=float(m[2]), nce=float(m[3]), mc_mr=float(m[4])))
        if "dist-health:" in line:
            for key, pattern in {
                "between": r"between=([\d.]+)",
                "variance_cv": r"\(CV ([\d.]+)\)",
                "negative_in_spherical_proxy_percent": r"neg_in_ball=([\d.]+)%",
            }.items():
                current["epochs"][-1][key] = float(re.search(pattern, line)[1])
        if "Best model saved" in line:
            current["best_saved_epoch"] = current["epochs"][-1]["epoch"]
        if "selected by: cosine_mr" in line:
            current["best_cosine_mr"] = float(re.search(r"recall@1: ([\d.]+)", line)[1])
    return runs


def static_checks():
    tree = ast.parse((ROOT / "utils/mcdisp_align_trainer.py").read_text(encoding="utf-8"))
    run = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "run_mcdisp_align_training")
    constructor = next(n for n in ast.walk(run) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "MCDispAlignModel")
    train_call = next(n for n in ast.walk(run) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "train_epoch")
    return {
        "model_constructor_line": constructor.lineno,
        "model_constructor_keywords": [k.arg for k in constructor.keywords],
        "mu_residual_forwarded": any(k.arg == "mu_residual" for k in constructor.keywords),
        "no_staged_forwarded_to_train_epoch": any(k.arg == "no_staged" for k in train_call.keywords),
    }


def normalize(x):
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


def nce(logits):
    def ce(x):
        z = x - x.max(axis=1, keepdims=True)
        return np.mean(np.log(np.exp(z).sum(axis=1)) - np.diag(z))
    return float((ce(logits) + ce(logits.T)) / 2)


def mechanism_checks():
    rng = np.random.default_rng(42)
    b, k, d = 32, 5, 64
    centers = rng.normal(size=(b, d)) * .25
    captions = centers[:, None, :] + rng.normal(size=(b, k, d)) * .18
    means = captions.mean(axis=1)
    images = means + rng.normal(size=(b, d)) * .10
    cosine = normalize(images) @ normalize(means).T
    rows = []
    for scale in [1., .5, .25]:
        im, cap = images * scale, captions * scale
        mu = cap.mean(axis=1)
        spread = cap.var(axis=1)
        target_var = .04 + spread
        cos = normalize(im) @ normalize(mu).T
        discounted = cos / (.07 * np.sqrt(1.04) * np.sqrt(1 + target_var.mean(axis=1))[None, :])
        kl_mu = np.mean(.5 * (im - mu) ** 2 / target_var)
        kl_var = np.mean(.5 * (.04 / target_var + np.log(target_var / .04) - 1))
        rows.append(dict(scale=scale, cosine_nce=nce(cos / .07), discounted_nce=nce(discounted), kl_mean=float(kl_mu), kl_var=float(kl_var), caption_spread=float(spread.mean())))
        assert np.allclose(cos, cosine)
    assert rows[-1]["discounted_nce"] < rows[0]["discounted_nce"]
    assert rows[-1]["kl_mean"] < rows[0]["kl_mean"]
    # The text scale derivative is positive through merged variance alone.
    def text_scale_loss(s):
        var = .04 + (captions * s).var(axis=1)
        return nce(cosine / (.07 * np.sqrt(1.04) * np.sqrt(1 + var.mean(axis=1))[None, :]))
    derivative = (text_scale_loss(1.00001) - text_scale_loss(.99999)) / .00002
    assert derivative > 0
    # An image-query variance factor cannot change that row's ranking.
    gallery_scale = np.sqrt(1 + rng.uniform(.01, .2, b))
    row_a = cosine / gallery_scale[None, :]
    row_b = row_a / np.sqrt(1 + rng.uniform(.01, 10, b))[:, None]
    assert np.array_equal(np.argsort(row_a, axis=1), np.argsort(row_b, axis=1))
    # Coverage can be improved by growing U, without changing the means.
    diff = rng.normal(size=d)
    u = diff / np.linalg.norm(diff)
    coverage = []
    for scale in [.1, 1., 10.]:
        covariance = .04 * np.eye(d) + scale ** 2 * np.outer(u, u)
        mahal = diff @ np.linalg.solve(covariance, diff) / d
        coverage.append(dict(u_scale=scale, mahal_per_dim=float(mahal)))
    assert coverage[-1]["mahal_per_dim"] < coverage[0]["mahal_per_dim"]
    return {"joint_scale_path": rows, "text_scale_nce_derivative": derivative,
            "query_variance_does_not_change_ranking": True,
            "coverage_u_inflation_path": coverage,
            "scope": "Synthetic forward/finite-difference checks; not a training replay or a measured gradient conflict."}


if __name__ == "__main__":
    results = {"runs": parse_runs(), "source": static_checks(), "mechanisms": mechanism_checks()}
    output = Path(__file__).with_name("audit_results.json")
    output.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(results, ensure_ascii=False, indent=2))
