"""本地轻量集合级评估：只算 rebuttal 表需要的 set_mc / set_cos。

与 run.py --phase eval 的区别：跳过标准检索（mc/cos/likelihood 的 N×N×K 打分）
与 H1/H2/H3，只做集合级 N×N——峰值内存从 ~30GB 降到 <2GB，本机可稳定跑完。
输出 eval.json 的 retrieval.set_mc / set_cos 字段与 phase_eval 完全同构。

用法：python scripts/eval_setonly_local.py --name no_cover \
        --checkpoint outputs/ablation/checkpoints/no_cover/best.pt
"""

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

import config  # noqa: E402
from scripts.ablation_study import mc_ranking  # noqa: E402
from scripts.ablation_study.feature_extraction import extract_features  # noqa: E402
from utils.logger import get_logger  # noqa: E402

logger = get_logger("eval_setonly", config.LOG_DIR / "eval_setonly.log")

K_VALUES = (1, 5, 10)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True, help="组名，如 no_cover / no_direction")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--test-manifest",
                    default="outputs/ablation_study/manifests/manifest_coco_legacyval.json")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    feats = extract_features(
        Path(args.checkpoint), Path(args.test_manifest), num_captions=5,
        batch_size=args.batch_size, device=args.device)
    img_mu, img_lv = feats["img_mu"], feats["img_logvar"]
    t_mus, t_lvs = feats["text_mus"], feats["text_logvars"]
    N, K, D = t_mus.shape

    # 集合表示：矩匹配（与 run.py phase_eval 完全一致）
    set_mu = t_mus.mean(dim=1)
    spread2 = ((t_mus - set_mu.unsqueeze(1)) ** 2).mean(dim=1)
    set_lv = torch.log(spread2 + torch.exp(t_lvs).mean(dim=1) + config.MCDISP_ALIGN_COV_EPS)

    r_set_mc = mc_ranking.ranks_from_sim(
        mc_ranking.mc_sim_rows(img_mu, img_lv, set_mu, set_lv,
                               tau=config.MCDISP_ALIGN_TAU, dtype=torch.float32),
        1, K_VALUES)
    r_set_cos = mc_ranking.ranks_from_sim(
        mc_ranking.cos_sim_rows(img_mu, img_lv, set_mu, set_lv, dtype=torch.float32),
        1, K_VALUES)

    def _retr(r):
        return {
            **{f"i2t_r@{k}": r["i2t_r"][k] for k in K_VALUES},
            **{f"t2i_r@{k}": r["t2i_r"][k] for k in K_VALUES},
        }

    result = {
        "experiment": args.name, "seed": 42, "checkpoint": args.checkpoint,
        "num_images": N, "K": K,
        "protocol": "set-level only (legacy-val), lightweight local eval",
        "retrieval": {"set_mc": _retr(r_set_mc), "set_cos": _retr(r_set_cos)},
    }
    out = Path(config.OUTPUT_DIR) / "ablation_study" / args.name / "seed42" / "eval.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    mc = result["retrieval"]["set_mc"]
    mr = sum(mc[f"{d}_r@{k}"] for d in ("i2t", "t2i") for k in K_VALUES) / 6
    logger.info(f"{args.name}: set_mc i2t@1={mc['i2t_r@1']:.4f} t2i@1={mc['t2i_r@1']:.4f} "
                f"mR={mr:.4f} -> {out}")


if __name__ == "__main__":
    main()
