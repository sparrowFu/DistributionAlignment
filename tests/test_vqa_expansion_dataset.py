"""
Tests for the VQA expansion dataset loader.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from data.vqa_expansion_dataset import VQAExpansionDataset


def _write_jsonl(path: Path, recs):
    with open(path, "w", encoding="utf-8") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _make_fixture(tmp_path: Path):
    imgdir = tmp_path / "imgs"
    imgdir.mkdir()
    train = tmp_path / "train.jsonl"
    test = tmp_path / "test.jsonl"
    _write_jsonl(train, [
        {"imagefilename": "a.jpg", "imagepath": "imgs/a.jpg", "type": 0,
         "question": "q1", "answer": "cat", "caption": "a cat"},
        {"imagefilename": "a.jpg", "imagepath": "imgs/a.jpg", "type": 2,
         "question": "q2", "answer": "red", "caption": "a red cat"},
        {"imagefilename": "b.jpg", "imagepath": "imgs/b.jpg", "type": 1,
         "question": "q3", "answer": "two", "caption": "two cats"},
    ])
    _write_jsonl(test, [
        {"imagefilename": "c.jpg", "imagepath": "imgs/c.jpg", "type": 0,
         "question": "q4", "answer": "dog", "caption": "a dog"},
    ])
    return train, test, imgdir


def test_merge_and_counts(tmp_path):
    train, test, imgdir = _make_fixture(tmp_path)
    ds = VQAExpansionDataset(train, test, images_dir=imgdir)
    assert ds.num_entries == 4
    assert ds.num_images == 3            # a, b, c
    assert ds.captions() == ["a cat", "a red cat", "two cats", "a dog"]
    assert set(ds.answer_vocab.keys()) == {"cat", "red", "two", "dog"}


def test_gt_maps(tmp_path):
    train, test, imgdir = _make_fixture(tmp_path)
    ds = VQAExpansionDataset(train, test, images_dir=imgdir)
    a_id = ds.image_id_of["a.jpg"]
    assert set(ds.img_to_caps[a_id]) == {0, 1}     # 图 a 有 2 条 caption
    assert ds.cap_to_img[0] == a_id
    assert ds.cap_to_img[1] == a_id
    assert ds.cap_to_img[3] == ds.image_id_of["c.jpg"]


def test_splits_disjoint(tmp_path):
    train, test, imgdir = _make_fixture(tmp_path)
    ds = VQAExpansionDataset(train, test, images_dir=imgdir)
    assert ds.images[ds.image_id_of["a.jpg"]]["split"] == "train"
    assert ds.images[ds.image_id_of["c.jpg"]]["split"] == "test"
    assert set(ds.image_splits()) == {"train", "test"}


def test_answer_ids_and_image_paths(tmp_path):
    train, test, imgdir = _make_fixture(tmp_path)
    ds = VQAExpansionDataset(train, test, images_dir=imgdir)
    aids = ds.answer_ids()
    assert aids.shape[0] == 4
    assert aids.dtype == torch.long
    names = sorted(p.name for p in ds.image_paths())
    assert names == ["a.jpg", "b.jpg", "c.jpg"]


if __name__ == "__main__":
    import tempfile
    fails = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            with tempfile.TemporaryDirectory() as td:
                try:
                    fn(Path(td))
                    print(f"PASS {name}")
                except AssertionError as e:
                    fails += 1
                    print(f"FAIL {name}: {e}")
    if fails:
        print(f"{fails} tests failed")
        sys.exit(1)
    print("all dataset tests passed")
