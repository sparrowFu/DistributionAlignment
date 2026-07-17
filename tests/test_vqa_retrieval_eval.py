"""
Tests for scripts/eval_vqa_retrieval.py 的 encode_all / compute_all_metrics。
用 fake adapter 跑编排逻辑,不加载真实模型。

Runnable without pytest:
    python tests/test_vqa_retrieval_eval.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import torch

from eval_vqa_retrieval import encode_all


class FakeAdapter:
    """mean = caption 首字符 ascii;logvar=None;supports_csd=False。"""
    supports_csd = False
    dim = 4

    def process_images(self, pils):
        return torch.randn(len(pils), 3, 4, 4)

    def process_text(self, captions):
        ids = torch.tensor([[ord(c[0]) % 7] for c in captions])
        return {"input_ids": ids, "attention_mask": torch.ones_like(ids)}

    def encode_images(self, pixel_values):
        n = pixel_values.shape[0]
        return torch.arange(n).float().unsqueeze(1).expand(n, self.dim), None

    def encode_texts(self, input_ids, attention_mask):
        n = input_ids.shape[0]
        return input_ids[:, 0].float().unsqueeze(1).expand(n, self.dim), None


class _PrintLogger:
    def info(self, msg): print(msg)
    def warning(self, msg): print(msg)


def test_encode_all_dedups_captions(tmp_path):
    # 两张图(真黑图占位),三条 caption 其中两条文本相同
    from PIL import Image
    imgdir = tmp_path / "imgs"
    imgdir.mkdir()
    for n in ("a.png", "b.png"):
        Image.new("RGB", (4, 4)).save(imgdir / n)
    caps = ["red bus", "red bus", "green bus"]
    adapter = FakeAdapter()
    img_mean, img_lv, cap_mean, cap_lv = encode_all(
        adapter, [imgdir / "a.png", imgdir / "b.png"], caps,
        batch_size=8, device="cpu", logger=_PrintLogger(),
    )
    assert img_mean.shape == (2, 4)
    assert cap_mean.shape == (3, 4)
    # 文本去重:red bus 编码一次,两条 entry 的特征应完全相同
    assert torch.equal(cap_mean[0], cap_mean[1])
    assert not torch.equal(cap_mean[0], cap_mean[2])
    assert img_lv is None and cap_lv is None


from eval_vqa_retrieval import compute_all_metrics


def _tiny_ds_and_feats():
    """2 图(a,b)+ 1 test 图(c);3 train entry + 1 test entry;特征使检索可解析。"""
    import tempfile, json
    from data.vqa_expansion_dataset import VQAExpansionDataset
    tmp = Path(tempfile.mkdtemp())
    imgdir = tmp / "imgs"; imgdir.mkdir()
    train = tmp / "train.jsonl"; test = tmp / "test.jsonl"
    recs_train = [
        {"imagefilename": "a.jpg", "imagepath": "imgs/a.jpg", "type": 0, "question": "q", "answer": "cat", "caption": "a cat"},
        {"imagefilename": "a.jpg", "imagepath": "imgs/a.jpg", "type": 2, "question": "q", "answer": "red", "caption": "a red cat"},
        {"imagefilename": "b.jpg", "imagepath": "imgs/b.jpg", "type": 0, "question": "q", "answer": "dog", "caption": "a dog"},
    ]
    with open(train, "w") as f:
        for r in recs_train:
            f.write(json.dumps(r) + "\n")
    with open(test, "w") as f:
        f.write(json.dumps({"imagefilename": "c.jpg", "imagepath": "imgs/c.jpg", "type": 1, "question": "q", "answer": "two", "caption": "two dogs"}) + "\n")
    ds = VQAExpansionDataset(train, test, images_dir=imgdir)
    # image_id: a=0, b=1, c=2 ; entries: [a/obj, a/color, b/obj, c/num]
    img_mean = torch.tensor([
        [1., 0., 0., 0.],   # img a
        [0., 1., 0., 0.],   # img b
        [0., 0., 1., 0.],   # img c
    ])
    cap_mean = torch.tensor([
        [1., 0., 0., 0.],   # entry0 (a cat)      -> img a
        [1., 0., 0., 0.],   # entry1 (a red cat)  -> img a
        [0., 1., 0., 0.],   # entry2 (a dog)      -> img b
        [0., 0., 1., 0.],   # entry3 (two dogs)   -> img c
    ])
    return ds, img_mean, cap_mean


def test_trackA_i2t_and_t2i_overall():
    ds, img_mean, cap_mean = _tiny_ds_and_feats()
    out = compute_all_metrics(ds, img_mean, None, cap_mean, None, [1], use_csd=False)
    # I2T: 图 a GT={0,1} top1 落 0/1; b->2; c->3 => 1.0
    assert out["track_a_same_image"]["i2t"]["cosine"]["overall"]["recall@1"] == 1.0
    # T2I: entry0->a, entry1->a, entry2->b, entry3->c => all hit
    assert out["track_a_same_image"]["t2i"]["cosine"]["overall"]["recall@1"] == 1.0


def test_trackB_answer_match_overall():
    ds, img_mean, cap_mean = _tiny_ds_and_feats()
    out = compute_all_metrics(ds, img_mean, None, cap_mean, None, [1], use_csd=False)
    # 每个图只与自身 caption 同方向,排除自身后 top1 落到别的答案 -> 全错 => 0.0
    assert out["track_b_answer_match"]["i2t"]["cosine"]["overall"]["answer_match@1"] == 0.0


def test_csd_block_uses_csd_not_cosine():
    """回归:cosine 块与 csd 块必须分别计算(防止闭包捕获错误的 use_csd)。
    cap0 高方差 -> CSD 下被折扣,img0 的 top1 从 cap0 翻到 cap1。"""
    import tempfile, json
    from data.vqa_expansion_dataset import VQAExpansionDataset
    tmp = Path(tempfile.mkdtemp())
    imgdir = tmp / "imgs"; imgdir.mkdir()
    train = tmp / "train.jsonl"
    with open(train, "w") as f:
        for r in [
            {"imagefilename": "a.jpg", "imagepath": "imgs/a.jpg", "type": 0, "question": "q", "answer": "cat", "caption": "a cat"},
            {"imagefilename": "b.jpg", "imagepath": "imgs/b.jpg", "type": 0, "question": "q", "answer": "dog", "caption": "a dog"},
        ]:
            f.write(json.dumps(r) + "\n")
    test = tmp / "test.jsonl"; open(test, "w").write("")  # 空 test
    ds = VQAExpansionDataset(train, test, images_dir=imgdir)
    img_mean = torch.tensor([[1., 0.], [0., 1.]])
    cap_mean = torch.tensor([[1., 0.], [0., 1.]])
    cap_logvar = torch.tensor([[5., 5.], [0., 0.]])   # cap0 高方差
    img_logvar = torch.zeros(2, 2)
    out = compute_all_metrics(ds, img_mean, img_logvar, cap_mean, cap_logvar, [1], use_csd=True)
    cos = out["track_a_same_image"]["i2t"]["cosine"]["overall"]["recall@1"]
    csd = out["track_a_same_image"]["i2t"]["csd"]["overall"]["recall@1"]
    assert cos == 1.0, f"cosine should be 1.0: {cos}"           # cosine: img0->cap0, img1->cap1
    assert csd == 0.5, f"csd should be 0.5: {csd}"              # csd: img0 top1 翻到 cap1(miss), img1 仍 hit
    assert cos != csd, "cosine 与 csd 块必须不同(否则闭包捕获了错误的 use_csd)"


if __name__ == "__main__":
    import inspect
    import tempfile
    fails = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            nargs = len(inspect.signature(fn).parameters)
            try:
                if nargs:
                    with tempfile.TemporaryDirectory() as td:
                        fn(Path(td))
                else:
                    fn()
                print(f"PASS {name}")
            except AssertionError as e:
                fails += 1
                print(f"FAIL {name}: {e}")
    if fails:
        print(f"{fails} tests failed")
        sys.exit(1)
    print("all eval tests passed")
