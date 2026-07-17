"""
VQA-as-Retrieval 指标:非对角、支持 1-to-many 的 Recall@K,以及排除自身的
answer-match@K。cosine(mean·mean,均值已归一化)与 CSD(ProLIP:减去 gallery 侧
0.5*sum(sigma^2))两种相似度。按 query 轴分块以防大图库 OOM。

与 utils/retrieval_metrics.py 的区别:后者假定对角 GT(query i ↔ gallery i);
本模块用每查询的 relevant 索引集合,支持一张图 ↔ 多条 caption。
"""

from typing import Dict, List, Optional, Sequence

import torch
import torch.nn.functional as F


def recall_at_k_from_relevance(
    query_mean: torch.Tensor,
    gallery_mean: torch.Tensor,
    gallery_logvar: Optional[torch.Tensor],
    query_relevant: Sequence[Sequence[int]],
    k_values: List[int],
    chunk_size: int = 1000,
    use_csd: bool = False,
    query_var: Optional[torch.Tensor] = None,
    query_U: Optional[torch.Tensor] = None,
    gallery_var: Optional[torch.Tensor] = None,
    gallery_U: Optional[torch.Tensor] = None,
    use_loglik: bool = False,
    per_dim_normalize: bool = True,
    use_logdet: bool = True,
) -> Dict[str, float]:
    """每查询给定 relevant gallery 索引集合的 Recall@K。

    Args:
        query_mean: (N, D)
        gallery_mean: (G, D)
        gallery_logvar: (G, D) = log sigma^2,仅 use_csd=True 时使用
        query_relevant: 长度 N;每元素是该查询的 GT gallery 索引集合(可为多个)
        k_values: K 列表
        chunk_size: query 轴分块
        use_csd: 是否用 CSD 相似度
        query_var / query_U: 图像侧(image=query)的方差与低秩因子,仅 use_loglik=True 时使用
        gallery_var / gallery_U: 图像侧(image=gallery)的方差与低秩因子,仅 use_loglik=True 时使用
        use_loglik: 用分布对数似然打分(image 侧提供 var/U),覆盖 cosine/csd
        per_dim_normalize / use_logdet: 透传给 image_text_loglik_matrix
    """
    query_mean = F.normalize(query_mean, dim=-1)
    gallery_mean = F.normalize(gallery_mean, dim=-1)
    n = query_mean.shape[0]
    max_k = min(max(k_values), gallery_mean.shape[0])   # topk 要求 k <= gallery 大小
    if use_csd:
        if gallery_logvar is None:
            raise ValueError("use_csd=True 需要 gallery_logvar")
        gallery_unc = torch.exp(gallery_logvar).sum(dim=-1)  # (G,)
    rel_sets = [set(int(x) for x in r) for r in query_relevant]
    hits = {k: 0 for k in k_values}
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        if use_loglik:
            from utils.distribution_score import image_text_loglik_matrix
            if query_var is not None:   # image is the query (distribution center)
                sim = image_text_loglik_matrix(
                    query_mean[start:end], query_var[start:end],
                    query_U[start:end] if query_U is not None else None,
                    gallery_mean, per_dim_normalize=per_dim_normalize,
                    use_logdet=use_logdet, chunk_size=end - start,
                )  # (c, G)
            elif gallery_var is not None:   # image is the gallery
                sim = image_text_loglik_matrix(
                    gallery_mean, gallery_var,
                    gallery_U, query_mean[start:end],
                    per_dim_normalize=per_dim_normalize, use_logdet=use_logdet,
                    chunk_size=256,
                ).T  # (G, c) -> (c, G)
            else:
                raise ValueError("use_loglik=True requires query_var/query_U (image=query) or gallery_var/gallery_U (image=gallery)")
        else:
            sim = query_mean[start:end] @ gallery_mean.T  # (c, G)
            if use_csd:
                sim = sim - 0.5 * gallery_unc.unsqueeze(0)
        top_idx = torch.topk(sim, max_k, dim=1).indices  # (c, max_k) 只取前 K,不全排序
        top_idx = top_idx.cpu()                          # 小批量搬 CPU 做集合判定
        for row in range(end - start):
            top = top_idx[row].tolist()
            rel = rel_sets[start + row]
            for k in k_values:
                eff_k = min(k, max_k)                    # gallery 不足 k 时按实际容量算
                if any(idx in rel for idx in top[:eff_k]):
                    hits[k] += 1
    return {f"recall@{k}": hits[k] / n for k in k_values}


def answer_match_at_k(
    query_mean: torch.Tensor,
    gallery_mean: torch.Tensor,
    gallery_logvar: Optional[torch.Tensor],
    gallery_answers: torch.Tensor,
    query_answers: torch.Tensor,
    query_exclude_idx: torch.Tensor,
    k_values: List[int],
    chunk_size: int = 1000,
    use_csd: bool = False,
    query_var: Optional[torch.Tensor] = None,
    query_U: Optional[torch.Tensor] = None,
    gallery_var: Optional[torch.Tensor] = None,
    gallery_U: Optional[torch.Tensor] = None,
    use_loglik: bool = False,
    per_dim_normalize: bool = True,
    use_logdet: bool = True,
) -> Dict[str, float]:
    """Track B:top-K 召回 caption 的 answer 是否 == 查询 answer(排除自身 entry)。

    Args:
        query_mean: (N, D)
        gallery_mean: (G, D)
        gallery_logvar: (G, D) 或 None
        gallery_answers: (G,) long,每条 gallery 的 answer id
        query_answers: (N,) long,每查询的 answer id
        query_exclude_idx: (N,) long,每查询要屏蔽的自身 gallery 索引
        query_var / query_U: entry 图像侧(image=query)的方差与低秩因子,仅 use_loglik=True 时使用
        gallery_var / gallery_U: 图像侧(image=gallery)的方差与低秩因子,仅 use_loglik=True 时使用(对称保留)
        use_loglik: 用分布对数似然打分(image 侧提供 var/U),覆盖 cosine/csd
        per_dim_normalize / use_logdet: 透传给 image_text_loglik_matrix
    """
    query_mean = F.normalize(query_mean, dim=-1)
    gallery_mean = F.normalize(gallery_mean, dim=-1)
    n = query_mean.shape[0]
    max_k = min(max(k_values), gallery_mean.shape[0])
    device = query_mean.device
    gallery_answers = gallery_answers.to(device)
    query_answers = query_answers.to(device)
    query_exclude_idx = query_exclude_idx.to(device)
    if use_csd:
        if gallery_logvar is None:
            raise ValueError("use_csd=True 需要 gallery_logvar")
        gallery_unc = torch.exp(gallery_logvar).sum(dim=-1)
    hits = {k: 0 for k in k_values}
    rows_arange = torch.arange(chunk_size, device=device)
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        c = end - start
        if use_loglik:
            from utils.distribution_score import image_text_loglik_matrix
            if query_var is not None:   # image is the query (distribution center)
                sim = image_text_loglik_matrix(
                    query_mean[start:end], query_var[start:end],
                    query_U[start:end] if query_U is not None else None,
                    gallery_mean, per_dim_normalize=per_dim_normalize,
                    use_logdet=use_logdet, chunk_size=end - start,
                )  # (c, G)
            elif gallery_var is not None:   # image is the gallery
                sim = image_text_loglik_matrix(
                    gallery_mean, gallery_var,
                    gallery_U, query_mean[start:end],
                    per_dim_normalize=per_dim_normalize, use_logdet=use_logdet,
                    chunk_size=256,
                ).T  # (G, c) -> (c, G)
            else:
                raise ValueError("use_loglik=True requires query_var/query_U (image=query) or gallery_var/gallery_U (image=gallery)")
        else:
            sim = query_mean[start:end] @ gallery_mean.T  # (c, G)
            if use_csd:
                sim = sim - 0.5 * gallery_unc.unsqueeze(0)
        rows = rows_arange[:c]
        sim[rows, query_exclude_idx[start:end]] = float("-inf")  # 屏蔽自身
        top_idx = torch.topk(sim, max_k, dim=1).indices    # (c, max_k) 只取前 K
        top_ans = gallery_answers[top_idx]                  # (c, max_k)
        q_ans = query_answers[start:end].unsqueeze(1)       # (c, 1)
        match = (top_ans == q_ans)                          # (c, max_k)
        for k in k_values:
            eff_k = min(k, max_k)
            hits[k] += match[:, :eff_k].any(dim=1).sum().item()
    return {f"answer_match@{k}": hits[k] / n for k in k_values}
