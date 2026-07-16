"""
Tests for utils/vqa_retrieval_metrics.py
非对角、支持 1-to-many 的 Recall@K,以及排除自身的 answer-match@K。

Runnable without pytest:
    python tests/test_vqa_retrieval_metrics.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from utils.vqa_retrieval_metrics import (
    recall_at_k_from_relevance,
    answer_match_at_k,
)


def test_recall_diagonal_top1():
    feats = torch.eye(3)
    rel = [{0}, {1}, {2}]
    r = recall_at_k_from_relevance(feats, feats, None, rel, [1, 5])
    assert r["recall@1"] == 1.0
    assert r["recall@5"] == 1.0


def test_recall_one_to_many():
    # query0 同时匹配 gallery0 和 gallery1(都 == [1,0,0]);q1/q2 各匹配一条
    q = torch.tensor([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])
    g = torch.tensor([[1., 0., 0.], [1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])
    rel = [{0, 1}, {2}, {3}]
    r = recall_at_k_from_relevance(q, g, None, rel, [1])
    assert r["recall@1"] == 1.0  # q0 top1 必落在 {0,1};q1->2;q2->3


def test_recall_miss():
    q = torch.eye(2)
    g = torch.tensor([[0., 1.], [1., 0.]])  # 与 q 对调,query0 最近 gallery1
    rel = [{0}, {1}]
    r = recall_at_k_from_relevance(q, g, None, rel, [1])
    assert r["recall@1"] == 0.0


def test_recall_csd_prefers_low_variance_gallery():
    # 两个 gallery 均值相同;高方差者被 CSD 折扣,排到后面
    q = torch.tensor([[1., 0.]])
    g = torch.tensor([[1., 0.], [1., 0.]])
    logvar = torch.tensor([[0., 0.], [5., 5.]])  # gallery1 方差大
    rel = [{0}]
    r = recall_at_k_from_relevance(q, g, logvar, rel, [1], use_csd=True)
    assert r["recall@1"] == 1.0


def test_answer_match_excludes_self_and_hits_other():
    # query(图)与自身 entry0(answer A)及 entry1(answer A)都相同
    feats = torch.tensor([[1., 0.], [1., 0.], [0., 1.]])
    gallery_ans = torch.tensor([0, 0, 1])   # entry0=A, entry1=A, entry2=B
    query_ans = torch.tensor([0])           # query answer=A
    exclude = torch.tensor([0])             # 排除 entry0(自身)
    m = answer_match_at_k(feats[:1], feats, None, gallery_ans, query_ans, exclude, [1])
    assert m["answer_match@1"] == 1.0       # 排除自身后 top1=entry1(A)


def test_answer_match_self_only_then_wrong():
    # query 只与自身 caption 相同(answer A),次近是 answer B -> 排除自身后答错
    q = torch.tensor([[1., 0.]])
    g = torch.tensor([[1., 0.], [0.8, 0.6]])
    g_ans = torch.tensor([0, 1])
    q_ans = torch.tensor([0])
    excl = torch.tensor([0])
    m = answer_match_at_k(q, g, None, g_ans, q_ans, excl, [1])
    assert m["answer_match@1"] == 0.0


if __name__ == "__main__":
    fails = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS {name}")
            except AssertionError as e:
                fails += 1
                print(f"FAIL {name}: {e}")
    if fails:
        print(f"{fails} tests failed")
        sys.exit(1)
    print("all metrics tests passed")
