"""One-batch validation of the CLIP Set-FT objective (rebuttal exp 1).

Checks, without loading CLIP weights:
  - the per-image set vector is the L2-normalized mean of that image's own
    K caption features (pooling order of the flattened B*K list),
  - the BxB similarity matrix has its positives on the diagonal
    (images scored against their own set vector -> accuracy 1.0, loss ~0),
  - a deliberately wrong pooling (off-by-image) does NOT pass.
"""

import importlib.util
from pathlib import Path

import torch
import torch.nn.functional as F

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "train_clip_setft.py"
_spec = importlib.util.spec_from_file_location("train_clip_setft", _SCRIPT)
setft = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(setft)


class _StubModel:
    """process_text/encode_text stub: caption "img{i}_cap{j}" -> base_i + eps_j."""

    def __init__(self, B, K, D):
        torch.manual_seed(0)
        self.features = {}
        for i in range(B):
            base = torch.randn(D)
            for j in range(K):
                self.features[f"img{i}_cap{j}"] = base + 0.01 * torch.randn(D)
        self._order = None

    def process_text(self, texts):
        self._order = list(texts)
        ids = torch.arange(len(texts)).unsqueeze(1)
        return {"input_ids": ids, "attention_mask": torch.ones_like(ids)}

    def encode_text(self, input_ids, attention_mask, normalize=False):
        feats = torch.stack([self.features[t] for t in self._order])
        return F.normalize(feats, dim=-1) if normalize else feats


def _make_batch(B, K):
    return [[f"img{i}_cap{j}" for j in range(K)] for i in range(B)]


def test_set_vector_pooling_order():
    B, K, D = 4, 5, 16
    model = _StubModel(B, K, D)
    captions_list = _make_batch(B, K)

    set_feats = setft._set_text_features(model, captions_list, torch.device("cpu"))

    assert set_feats.shape == (B, D)
    for i in range(B):
        raw_mean = torch.stack(
            [model.features[f"img{i}_cap{j}"] for j in range(K)]).mean(dim=0)
        assert torch.allclose(set_feats[i], F.normalize(raw_mean, dim=-1), atol=1e-6), \
            f"image {i}: set vector is not the normalized mean of its own captions"


def test_similarity_matrix_diagonal_positives():
    B, K, D = 4, 5, 16
    model = _StubModel(B, K, D)
    captions_list = _make_batch(B, K)

    set_feats = setft._set_text_features(model, captions_list, torch.device("cpu"))
    image_feats = set_feats.clone()  # each image "matches" its own set vector

    loss, info = setft.clip_contrastive_loss(
        image_feats, set_feats, temperature=0.07)

    assert info["acc_i2t"] == 1.0 and info["acc_t2i"] == 1.0
    assert loss.item() < 1e-4


def test_wrong_pooling_fails_the_check():
    """Sanity check the test itself: rotate images by one -> positives off
    the diagonal, accuracy must drop below 1.0."""
    B, K, D = 4, 5, 16
    model = _StubModel(B, K, D)
    captions_list = _make_batch(B, K)

    set_feats = setft._set_text_features(model, captions_list, torch.device("cpu"))
    image_feats = torch.roll(set_feats, shifts=1, dims=0)

    _, info = setft.clip_contrastive_loss(
        image_feats, set_feats, temperature=0.07)

    assert info["acc"] < 1.0


def test_uneven_caption_counts_raise():
    B, K, D = 4, 5, 16
    model = _StubModel(B, K, D)
    captions_list = _make_batch(B, K)
    captions_list[-1] = captions_list[-1][:-1]  # last image only has K-1

    try:
        setft._set_text_features(model, captions_list, torch.device("cpu"))
    except ValueError:
        return
    raise AssertionError("uneven caption counts must raise ValueError")
