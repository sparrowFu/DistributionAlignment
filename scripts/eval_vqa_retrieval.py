"""
VQA-as-Retrieval 评估脚本(gemma caption)。

把 VQA 重构为双向图文检索,评估 5 个模型:
  clip_zero_shot / clip_baseline / dist_align / prolip_zero_shot / prolip
两套指标(同图 R@K + answer-match@K),都按 类型 × split × overall 拆。

纯新增文件,不修改现有代码;import config 只读复用路径与常量。

Usage:
    python scripts/eval_vqa_retrieval.py --model dist_align
    python scripts/eval_vqa_retrieval.py --model all
    python scripts/eval_vqa_retrieval.py --model clip_zero_shot --num-samples 64  # 冒烟
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from data.vqa_expansion_dataset import VQAExpansionDataset
from utils.logger import get_logger, log_exception
from utils.vqa_retrieval_metrics import recall_at_k_from_relevance, answer_match_at_k

logger = get_logger("eval_vqa_retrieval", config.LOG_DIR / "eval_vqa_retrieval.log")


@torch.no_grad()
def encode_all(adapter, image_paths: List[Path], captions: List[str],
               batch_size: int, device: str, logger=logger) -> Tuple[
        torch.Tensor, Optional[torch.Tensor], torch.Tensor, Optional[torch.Tensor]]:
    """编码全部唯一图 + 全部 entry caption(文本按字符串去重编码再展开)。

    Returns: (img_mean[N,D], img_logvar|None, cap_mean[M,D], cap_logvar|None)
    图片缺失则用黑图占位以保持 image_id 对齐,并告警。
    """
    from PIL import Image

    # ---- images ----
    img_means, img_logvars = [], []
    for start in tqdm(range(0, len(image_paths), batch_size), desc="encode images"):
        batch_paths = image_paths[start:start + batch_size]
        pils = []
        for p in batch_paths:
            try:
                im = Image.open(p)
                if im.mode != "RGB":
                    im = im.convert("RGB")
                pils.append(im)
            except Exception as e:
                logger.warning(f"image load failed {p}: {e}; using black placeholder")
                pils.append(Image.new("RGB", (224, 224)))
        pixel_values = adapter.process_images(pils).to(device)
        m, lv = adapter.encode_images(pixel_values)
        img_means.append(m)                       # 保留在 device(GPU)上,供指标阶段直接用
        if lv is not None:
            img_logvars.append(lv)
    img_mean = torch.cat(img_means, dim=0)
    img_logvar = torch.cat(img_logvars, dim=0) if img_logvars else None

    # ---- captions (dedup by string) ----
    unique_strs = list(dict.fromkeys(captions))          # 保序去重
    str_to_uidx = {s: i for i, s in enumerate(unique_strs)}
    cap_means_u, cap_logvars_u = [], []
    for start in tqdm(range(0, len(unique_strs), batch_size), desc="encode captions"):
        batch = unique_strs[start:start + batch_size]
        ti = adapter.process_text(batch)
        input_ids = ti["input_ids"].to(device)
        attn = ti["attention_mask"].to(device)
        m, lv = adapter.encode_texts(input_ids, attn)
        cap_means_u.append(m)
        if lv is not None:
            cap_logvars_u.append(lv)
    cap_mean_u = torch.cat(cap_means_u, dim=0)
    cap_logvar_u = torch.cat(cap_logvars_u, dim=0) if cap_logvars_u else None
    # 展开到 entry 级(uidx 与特征同 device,避免跨 device 索引报错)
    uidx = torch.tensor([str_to_uidx[c] for c in captions], dtype=torch.long,
                        device=cap_mean_u.device)
    cap_mean = cap_mean_u[uidx]
    cap_logvar = cap_logvar_u[uidx] if cap_logvar_u is not None else None

    logger.info(f"encoded: images {img_mean.shape}, captions {cap_mean.shape} "
                f"(unique strings {len(unique_strs)})")
    return img_mean, img_logvar, cap_mean, cap_logvar


TYPE_NAMES = {0: "object", 1: "number", 2: "color", 3: "location"}


def compute_all_metrics(ds, img_mean, img_logvar, cap_mean, cap_logvar,
                        k_values, use_csd):
    """计算 Track A(同图 R@K,双向)+ Track B(answer-match@K),分 类型×split×overall。

    ds 需暴露:answer_ids()->Tensor, types()->list, splits()->list,
    image_splits()->list, img_to_caps(list[list[int]]), cap_to_img(list[int])。
    """
    n_img = img_mean.shape[0]
    n_cap = cap_mean.shape[0]
    device = img_mean.device

    answer_ids = ds.answer_ids().to(device)            # (n_cap,)
    types = ds.types()                                 # (n_cap,)
    splits = ds.splits()                               # (n_cap,)
    img_splits = ds.image_splits()                     # (n_img,)
    img_to_caps = ds.img_to_caps                       # image_id -> [entry_idx]
    cap_to_img = ds.cap_to_img                         # entry_idx -> image_id

    # ---- Track A I2T: 查询=图, relevant=该图所有 entry ----
    i2t_rel_all = [set(img_to_caps[i]) for i in range(n_img)]
    # per-type: 只把该类型 entry 当 relevant
    i2t_rel_by_type = {t: [set(c for c in img_to_caps[i] if types[c] == t)
                           for i in range(n_img)] for t in TYPE_NAMES}

    def _i2t(csd, query_mask=None):
        if query_mask is None:
            idx = list(range(n_img))
        else:
            idx = [i for i, m in enumerate(query_mask) if m]
        if not idx:
            return {f"recall@{k}": 0.0 for k in k_values}
        return recall_at_k_from_relevance(
            img_mean[idx], cap_mean, cap_logvar,
            [i2t_rel_all[i] for i in idx], k_values, use_csd=csd)

    def _i2t_type(csd, t):
        rel = i2t_rel_by_type[t]
        mask = [bool(rel[i]) for i in range(n_img)]   # 仅拥有该类型 caption 的图
        if not any(mask):
            return {f"recall@{k}": 0.0 for k in k_values}
        return recall_at_k_from_relevance(
            img_mean[mask], cap_mean, cap_logvar,
            [rel[i] for i, mm in enumerate(mask) if mm], k_values, use_csd=csd)

    def _t2i(csd, query_mask=None):
        idx = list(range(n_cap)) if query_mask is None else [i for i, m in enumerate(query_mask) if m]
        if not idx:
            return {f"recall@{k}": 0.0 for k in k_values}
        return recall_at_k_from_relevance(
            cap_mean[idx], img_mean, img_logvar,
            [{cap_to_img[i]} for i in idx], k_values, use_csd=csd)

    # ---- Track B answer-match: 查询=entry 的图, gallery=entries, 排除自身 ----
    entry_img_mean = img_mean[torch.tensor(cap_to_img, device=device)]  # (n_cap, D)
    exclude = torch.arange(n_cap, device=device)

    def _ans_match(csd, query_mask=None):
        idx = list(range(n_cap)) if query_mask is None else [i for i, m in enumerate(query_mask) if m]
        if not idx:
            return {f"answer_match@{k}": 0.0 for k in k_values}
        return answer_match_at_k(
            entry_img_mean[idx], cap_mean, cap_logvar,
            answer_ids, answer_ids[idx], exclude[idx], k_values, use_csd=csd)

    # ---- 组装:overall + per-split + per-type(cosine;csd 仅当 use_csd)----
    split_set = sorted(set(splits))
    result = {"track_a_same_image": {"i2t": {}, "t2i": {}},
              "track_b_answer_match": {"i2t": {}}}

    for sim_name, csd in (("cosine", False), ("csd", True)):
        if sim_name == "csd" and not (use_csd and cap_logvar is not None and img_logvar is not None):
            continue
        # I2T
        i2t_block = {"overall": _i2t(csd)}
        for sp in split_set:
            i2t_block[sp] = _i2t(csd, [s == sp for s in img_splits])
        i2t_block["per_type"] = {str(t): _i2t_type(csd, t) for t in TYPE_NAMES}
        result["track_a_same_image"]["i2t"][sim_name] = i2t_block
        # T2I
        t2i_block = {"overall": _t2i(csd)}
        for sp in split_set:
            t2i_block[sp] = _t2i(csd, [s == sp for s in splits])
        t2i_block["per_type"] = {str(t): _t2i(csd, [ty == t for ty in types]) for t in TYPE_NAMES}
        result["track_a_same_image"]["t2i"][sim_name] = t2i_block
        # Track B
        b_block = {"overall": _ans_match(csd)}
        for sp in split_set:
            b_block[sp] = _ans_match(csd, [s == sp for s in splits])
        b_block["per_type"] = {str(t): _ans_match(csd, [ty == t for ty in types]) for t in TYPE_NAMES}
        result["track_b_answer_match"]["i2t"][sim_name] = b_block

    return result


ALL_MODELS = ["clip_zero_shot", "clip_baseline", "dist_align",
              "prolip_zero_shot", "prolip"]


class ModelAdapter:
    """统一 5 个底层模型的图像/文本编码接口。

    encode_images/encode_texts 返回 (mean, logvar|None);logvar = log sigma^2
    (仅 dist_align / prolip 提供,用于 CSD)。CLIP 系 logvar=None(只算 cosine)。
    """

    def __init__(self, name: str, device: str):
        self.name = name
        self.device = device
        self.supports_csd = False
        self._build()

    def _build(self):
        name = self.name
        if name in ("clip_zero_shot", "clip_baseline"):
            from models.clip_baseline import CLIPFineTuneBaseline
            self.base = CLIPFineTuneBaseline(freeze_image=True, freeze_text=True)
            if name == "clip_baseline":
                self.base.load(str(config.CLIP_BASELINE_BEST_CKPT))
            self.supports_csd = False
        elif name == "dist_align":
            from models.dist_align_model import DistributionAlignmentModel
            self.base = DistributionAlignmentModel(
                freeze_clip=config.DIST_ALIGN_FREEZE_CLIP,
                distribution_merging=config.DIST_ALIGN_DISTRIBUTION_MERGING,
                cov_rank=config.MSDA_COV_RANK,
            )
            self.base.load(str(config.DIST_ALIGN_BEST_CKPT))
            self.supports_csd = True
        elif name in ("prolip_zero_shot", "prolip"):
            from models.prolip_model import ProLIPModel
            self.base = ProLIPModel(freeze=(name == "prolip_zero_shot"))
            if name == "prolip":
                self.base.load(str(config.PROLIP_BEST_CKPT))
            self.supports_csd = True
        else:
            raise ValueError(f"unknown model {name!r}")
        self.base = self.base.to(self.device).eval()

    @property
    def dim(self):
        return 1024 if self.name.startswith("prolip") else 768

    # ----- 预处理(转发到底层模型)-----
    def process_images(self, pils):
        return self.base.process_images(pils)

    def process_text(self, captions):
        return self.base.process_text(captions)

    # ----- 编码 -----
    def encode_images(self, pixel_values):
        name = self.name
        if name in ("clip_zero_shot", "clip_baseline"):
            m = self.base.encode_image(pixel_values, normalize=True)
            return m, None
        if name == "dist_align":
            feat = self.base.clip_model.get_image_features(pixel_values).pooler_output
            mu = self.base.img_mu_head(feat)
            lv = self.base._floor_logvar(self.base.img_logvar_head(feat))
            return mu, lv
        # prolip:用 dummy 文本凑一次 forward,取 image 侧
        dummy_ids = torch.zeros((pixel_values.shape[0], 77), dtype=torch.long,
                                device=pixel_values.device)
        dummy_mask = torch.ones_like(dummy_ids)
        out = self.base(pixel_values, dummy_ids, dummy_mask)
        return out["image_features"]["mean"], out["img_logvar"]

    def encode_texts(self, input_ids, attention_mask):
        name = self.name
        if name in ("clip_zero_shot", "clip_baseline"):
            m = self.base.encode_text(input_ids, attention_mask, normalize=True)
            return m, None
        if name == "dist_align":
            feat = self.base.clip_model.get_text_features(
                input_ids=input_ids, attention_mask=attention_mask).pooler_output
            mu = self.base.text_mu_head(feat)
            lv = self.base._floor_logvar(self.base.text_logvar_head(feat))
            return mu, lv
        # prolip:用 dummy 图凑 forward,取 text 侧
        dummy_img = torch.zeros((input_ids.shape[0], 3, 224, 224),
                                device=input_ids.device)
        out = self.base(dummy_img, input_ids, attention_mask)
        return out["text_features"]["mean"], out["text_logvar"]


def _build_subset(ds, max_entries):
    """冒烟用:取前 max_entries 条 entry 及其图,重映射索引,返回 (sub_ds, img_paths, captions)。
    sub_ds 暴露 compute_all_metrics 需要的协议(answer_ids/types/splits/image_splits/img_to_caps/cap_to_img)。"""
    cap_idx = list(range(min(max_entries, ds.num_entries)))
    kept_img_old = sorted({ds.cap_to_img[i] for i in cap_idx})
    old2new = {o: n for n, o in enumerate(kept_img_old)}

    class _Sub:
        pass
    s = _Sub()
    s.answer_vocab = ds.answer_vocab
    _cap_idx = cap_idx
    s.answer_ids = lambda: ds.answer_ids()[_cap_idx]
    s.types = lambda: [ds.types()[i] for i in _cap_idx]
    s.splits = lambda: [ds.splits()[i] for i in _cap_idx]
    s.image_splits = lambda: [ds.image_splits()[i] for i in kept_img_old]
    s.cap_to_img = [old2new[ds.cap_to_img[i]] for i in cap_idx]
    new_img_to_caps = [[] for _ in kept_img_old]
    for new_c, old_c in enumerate(cap_idx):
        new_img_to_caps[old2new[ds.cap_to_img[old_c]]].append(new_c)
    s.img_to_caps = new_img_to_caps
    img_paths = [ds.image_paths()[i] for i in kept_img_old]
    captions = [ds.captions()[i] for i in cap_idx]
    return s, img_paths, captions


def _flatten(res):
    """summary 表:抽 overall 的关键数。"""
    row = {"model": res["model"], "dim": res["dim"], "supports_csd": res["supports_csd"]}
    for trk, blk in res["metrics"].items():
        for direction, sims in blk.items():
            for sim, body in sims.items():
                for k, v in body.get("overall", {}).items():
                    row[f"{trk}.{direction}.{sim}.{k}"] = round(v, 4)
    return row


def run_one(name, ds, args):
    logger.info(f"===== {name} =====")
    adapter = ModelAdapter(name, args.device)
    if args.num_samples and args.num_samples > 0:
        sub, img_paths, captions = _build_subset(ds, args.num_samples)
        logger.info(f"smoke subset: {len(captions)} entries, {len(img_paths)} images")
    else:
        sub, img_paths, captions = ds, ds.image_paths(), ds.captions()
    img_mean, img_lv, cap_mean, cap_lv = encode_all(
        adapter, img_paths, captions, args.batch_size, args.device)
    use_csd = adapter.supports_csd
    metrics = compute_all_metrics(sub, img_mean, img_lv, cap_mean, cap_lv,
                                  args.recall_at_k, use_csd)
    return {"model": name, "dim": adapter.dim, "supports_csd": use_csd,
            "num_images": int(img_mean.shape[0]),
            "num_entries": int(cap_mean.shape[0]),
            "metrics": metrics}


def parse_args():
    p = argparse.ArgumentParser(description="VQA-as-Retrieval evaluation (gemma caption)")
    p.add_argument("--model", type=str, default="all",
                   help="单模型名 或 'all'(跑全部 5 个)")
    p.add_argument("--train-path", type=str,
                   default=str(config.OUTPUT_DIR / "vqa_expansions" / "train_gemma_expansions.jsonl"))
    p.add_argument("--test-path", type=str,
                   default=str(config.OUTPUT_DIR / "vqa_expansions" / "test_gemma_expansions.jsonl"))
    p.add_argument("--images-dir", type=str, default=str(config.VQA_IMAGES_DIR))
    p.add_argument("--output-dir", type=str,
                   default=str(config.OUTPUT_DIR / "vqa_retrieval"))
    p.add_argument("--batch-size", type=int, default=config.EVAL_BATCH_SIZE)
    p.add_argument("--recall-at-k", type=int, nargs="+", default=config.RECALL_AT_K)
    p.add_argument("--num-samples", type=int, default=0,
                   help=">0:只取前 N 条 entry(+其图)做冒烟测试;0=全量")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ds = VQAExpansionDataset(Path(args.train_path), Path(args.test_path), Path(args.images_dir))
    logger.info(f"dataset: {ds.num_entries} entries, {ds.num_images} images, "
                f"{len(ds.answer_vocab)} answers")

    names = ALL_MODELS if args.model == "all" else [args.model]
    summary = []
    for name in names:
        try:
            res = run_one(name, ds, args)
        except Exception as e:
            logger.exception(f"{name} failed: {e}")
            continue
        with open(out_dir / f"{name}_results.json", "w", encoding="utf-8") as f:
            json.dump(res, f, ensure_ascii=False, indent=2)
        logger.info(f"saved {name}_results.json")
        summary.append(_flatten(res))
    if summary:
        with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        logger.info(f"saved summary.json ({len(summary)} models)")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        log_exception(logger, e, "eval_vqa_retrieval failed")
        raise
