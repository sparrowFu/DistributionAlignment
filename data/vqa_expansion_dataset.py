"""
VQA-as-Retrieval 数据集:读 train/test 两条 gemma expansion jsonl,合并成
entries(112,857)+ 唯一图(66,362)+ GT 映射。

每条 jsonl:
  {"imagefilename","imagepath","type","question","answer","caption"}

风格对齐 data/vqa_dataset.py;只存元数据,不在 __init__ 打开图片(编码时按需打开)。
"""

import json
from pathlib import Path
from typing import Dict, List

import torch

from utils.logger import get_logger

logger = get_logger("vqa_expansion_dataset")


class VQAExpansionDataset:
    def __init__(self, train_path: Path, test_path: Path, images_dir: Path):
        self.images_dir = Path(images_dir)
        self.entries: List[dict] = []          # entry_idx -> {caption,answer,type,split,image_id,imagefilename}
        self.images: List[dict] = []           # image_id -> {filename,path,split}
        self.image_id_of: Dict[str, int] = {}
        self.img_to_caps: List[List[int]] = []  # image_id -> [entry_idx]
        self.cap_to_img: List[int] = []          # entry_idx -> image_id
        self.answer_vocab: Dict[str, int] = {}

        for split, p in (("train", train_path), ("test", test_path)):
            self._load(split, Path(p))

        # answer vocab 从全量 entries 构建(稳定排序)
        self.answer_vocab = {a: i for i, a in enumerate(sorted({e["answer"] for e in self.entries}))}
        logger.info(f"VQAExpansionDataset: {self.num_entries} entries, "
                    f"{self.num_images} unique images, {len(self.answer_vocab)} answers")

    def _load(self, split: str, path: Path):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                fn = rec["imagefilename"]
                if fn not in self.image_id_of:
                    self.image_id_of[fn] = len(self.images)
                    self.images.append({
                        "filename": fn,
                        "path": self.images_dir / fn,
                        "split": split,
                    })
                    self.img_to_caps.append([])
                img_id = self.image_id_of[fn]
                entry_idx = len(self.entries)
                self.entries.append({
                    "caption": rec["caption"],
                    "answer": rec["answer"],
                    "type": int(rec["type"]),
                    "split": split,
                    "image_id": img_id,
                    "imagefilename": fn,
                })
                self.cap_to_img.append(img_id)
                self.img_to_caps[img_id].append(entry_idx)

    @property
    def num_entries(self) -> int:
        return len(self.entries)

    @property
    def num_images(self) -> int:
        return len(self.images)

    def image_paths(self) -> List[Path]:
        return [im["path"] for im in self.images]

    def image_splits(self) -> List[str]:
        return [im["split"] for im in self.images]

    def captions(self) -> List[str]:
        return [e["caption"] for e in self.entries]

    def answers(self) -> List[str]:
        return [e["answer"] for e in self.entries]

    def answer_ids(self) -> torch.Tensor:
        return torch.tensor([self.answer_vocab[e["answer"]] for e in self.entries], dtype=torch.long)

    def types(self) -> List[int]:
        return [e["type"] for e in self.entries]

    def splits(self) -> List[str]:
        return [e["split"] for e in self.entries]


if __name__ == "__main__":
    import config
    ds = VQAExpansionDataset(
        train_path=config.OUTPUT_DIR / "vqa_expansions" / "train_gemma_expansions.jsonl",
        test_path=config.OUTPUT_DIR / "vqa_expansions" / "test_gemma_expansions.jsonl",
        images_dir=config.VQA_IMAGES_DIR,
    )
    print(f"entries={ds.num_entries} images={ds.num_images} answers={len(ds.answer_vocab)}")
