"""
GaussianImageDistribution - Shared image-text retrieval utilities.

Centralizes Recall@K computation so every evaluation script reports the same
metrics in the same way, including BOTH retrieval directions:

    - I2T (Image -> Text): query = image, gallery = text
    - T2I (Text -> Image): query = text,  gallery = image

Previously each eval script carried its own (I2T-only) copy of this logic.
"""

from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F
from tqdm import tqdm


@torch.no_grad()
def compute_recall_chunked(
    query: torch.Tensor,
    gallery: torch.Tensor,
    k_values: List[int],
    chunk_size: int = 1000,
    normalize: bool = True,
) -> Dict[int, float]:
    """
    Recall@K where query[i]'s positive match is gallery[i] (diagonal pairing).

    Direction-agnostic: pass images as ``query`` for I2T, texts as ``query``
    for T2I.

    Args:
        query: Query features (N, D)
        gallery: Gallery features (N, D); the i-th gallery item matches query i
        k_values: List of K values for Recall@K
        chunk_size: Chunk size to avoid OOM on large similarity matrices
        normalize: L2-normalize features before computing cosine similarity

    Returns:
        Dict mapping K -> recall@K (raw K ints, e.g. {1: 0.82, 5: 0.97, 10: 0.99})
    """
    n = query.shape[0]
    if normalize:
        query = F.normalize(query, dim=-1)
        gallery = F.normalize(gallery, dim=-1)

    hits = {k: 0 for k in k_values}

    for start in tqdm(range(0, n, chunk_size), desc="Recall chunks", leave=False):
        end = min(start + chunk_size, n)
        sim_chunk = torch.matmul(query[start:end], gallery.T)  # (chunk, N)
        ranked = torch.argsort(sim_chunk, dim=1, descending=True)
        gt = torch.arange(start, end, device=query.device).unsqueeze(1)
        for k in k_values:
            hits[k] += (ranked[:, :k] == gt).any(dim=1).sum().item()

    return {k: hits[k] / n for k in k_values}


@torch.no_grad()
def compute_recall_bidirectional(
    img_features: torch.Tensor,
    text_features: torch.Tensor,
    k_values: List[int],
    chunk_size: int = 1000,
    normalize: bool = True,
) -> Dict[str, float]:
    """
    Bidirectional (I2T + T2I) Recall@K between aligned image/text feature sets.

    Returns keys: ``recall_i2t@{k}``, ``recall_t2i@{k}``, ``recall@{k}`` (mean).
    """
    i2t = compute_recall_chunked(img_features, text_features, k_values, chunk_size, normalize)
    t2i = compute_recall_chunked(text_features, img_features, k_values, chunk_size, normalize)

    out: Dict[str, float] = {}
    for k in k_values:
        out[f"recall_i2t@{k}"] = i2t[k]
        out[f"recall_t2i@{k}"] = t2i[k]
        out[f"recall@{k}"] = (i2t[k] + t2i[k]) / 2
    return out


@torch.no_grad()
def compute_recall_uc_chunked(
    img_mu: torch.Tensor,
    img_logvar: torch.Tensor,
    text_mu: torch.Tensor,
    text_logvar: torch.Tensor,
    k_values: List[int],
    temperature: float = 0.07,
    chunk_size: int = 1000,
) -> Dict[str, float]:
    """
    Bidirectional Recall@K using uncertainty-calibrated similarity.

        sim(x, y) = (mu_x . mu_y) / (tau * sqrt(1 + mean(sigma_x^2)) * sqrt(1 + mean(sigma_y^2)))

    Returns keys: ``uc_recall_i2t@{k}``, ``uc_recall_t2i@{k}``, ``uc_recall@{k}`` (mean).
    """
    img_var_avg = torch.exp(img_logvar).mean(dim=-1)    # (N,)
    text_var_avg = torch.exp(text_logvar).mean(dim=-1)  # (N,)
    img_scale = torch.sqrt(1.0 + img_var_avg)
    text_scale = torch.sqrt(1.0 + text_var_avg)

    img_mu_n = F.normalize(img_mu, dim=-1)
    text_mu_n = F.normalize(text_mu, dim=-1)
    n = img_mu.shape[0]

    def _direction(query_mu, gallery_mu, query_scale, gallery_scale, desc):
        hits = {k: 0 for k in k_values}
        for start in tqdm(range(0, n, chunk_size), desc=desc, leave=False):
            end = min(start + chunk_size, n)
            sim = torch.matmul(query_mu[start:end], gallery_mu.T)
            scale = query_scale[start:end].unsqueeze(1) * gallery_scale.unsqueeze(0)
            sim = sim / (temperature * scale)
            ranked = torch.argsort(sim, dim=1, descending=True)
            gt = torch.arange(start, end).unsqueeze(1)
            for k in k_values:
                hits[k] += (ranked[:, :k] == gt).any(dim=1).sum().item()
        return {k: hits[k] / n for k in k_values}

    i2t = _direction(img_mu_n, text_mu_n, img_scale, text_scale, "UC I2T chunks")
    t2i = _direction(text_mu_n, img_mu_n, text_scale, img_scale, "UC T2I chunks")

    out: Dict[str, float] = {}
    for k in k_values:
        out[f"uc_recall_i2t@{k}"] = i2t[k]
        out[f"uc_recall_t2i@{k}"] = t2i[k]
        out[f"uc_recall@{k}"] = (i2t[k] + t2i[k]) / 2
    return out


@torch.no_grad()
def extract_distribution_features(
    model,
    dataloader,
    device,
    num_samples: int = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Extract distribution parameters (mu, logvar) from any model whose
    ``forward(pixel_values, input_ids, attention_mask)`` returns a dict with
    ``img_mu / text_mu / img_logvar / text_logvar`` (dist_align, ProLIP, GroVE).

    Returns:
        (img_mu, text_mu, img_logvar, text_logvar), each (N, D)
    """
    model.eval()

    img_mu_list, text_mu_list, img_lv_list, text_lv_list = [], [], [], []
    sample_count = 0

    for batch in tqdm(dataloader, desc="Extracting features", leave=False):
        if batch is None:
            continue

        pil_images = batch["image"]
        caption_lists = batch["captions"]
        B = len(pil_images)
        K = len(caption_lists[0])

        pixel_values = model.process_images(pil_images).to(device)
        all_captions = [c for cl in caption_lists for c in cl]
        text_inputs = model.process_text(all_captions)
        input_ids = text_inputs["input_ids"].view(B, K, -1).to(device)
        attn_mask = text_inputs["attention_mask"].view(B, K, -1).to(device)

        out = model(pixel_values, input_ids, attn_mask)
        img_mu_list.append(out["img_mu"].cpu())
        text_mu_list.append(out["text_mu"].cpu())
        img_lv_list.append(out["img_logvar"].cpu())
        text_lv_list.append(out["text_logvar"].cpu())

        sample_count += B
        if num_samples and sample_count >= num_samples:
            break

    def _cat(xs):
        t = torch.cat(xs, dim=0)
        return t[:num_samples] if num_samples else t

    return (_cat(img_mu_list), _cat(text_mu_list),
            _cat(img_lv_list), _cat(text_lv_list))
