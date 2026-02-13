"""
Evaluation metrics for content-based movie recommendation (Mindminer).

Metrics implemented:
- Precision@K, Recall@K: relevance based on tag overlap (proxy for "similar content")
- MRR: Mean Reciprocal Rank of first relevant item
- NDCG@K: Normalized Discounted Cumulative Gain
- Catalog coverage: fraction of movies that appear in at least one top-K list
- Intra-list diversity: average pairwise dissimilarity within recommendations
"""

import numpy as np
from typing import List, Callable, Optional


def _tag_set(tags: str) -> set:
    """Parse tags string into set of tokens (space-separated)."""
    if not isinstance(tags, str) or not tags.strip():
        return set()
    return set(t.strip().lower() for t in tags.split() if t.strip())


def relevance_by_tag_overlap(
    query_tags: str, candidate_tags: str, min_common: int = 3
) -> bool:
    """True if candidate shares at least min_common tags with query."""
    a, b = _tag_set(query_tags), _tag_set(candidate_tags)
    return len(a & b) >= min_common


def relevance_score_tag_overlap(query_tags: str, candidate_tags: str) -> float:
    """Jaccard similarity of tag sets (0..1). Used as relevance grade for NDCG."""
    a, b = _tag_set(query_tags), _tag_set(candidate_tags)
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def precision_at_k(
    recommended: List[int],
    relevant: List[int],
    k: Optional[int] = None,
) -> float:
    """Precision@K: |recommended[:k] ∩ relevant| / k."""
    k = k or len(recommended)
    rec_k = set(recommended[:k])
    rel = set(relevant)
    if k == 0:
        return 0.0
    return len(rec_k & rel) / k


def recall_at_k(
    recommended: List[int],
    relevant: List[int],
    k: Optional[int] = None,
) -> float:
    """Recall@K: |recommended[:k] ∩ relevant| / |relevant|."""
    k = k or len(recommended)
    rec_k = set(recommended[:k])
    rel = set(relevant)
    if not rel:
        return 0.0
    return len(rec_k & rel) / len(rel)


def mean_reciprocal_rank(
    recommended: List[int],
    relevant: List[int],
) -> float:
    """MRR: 1 / rank of first relevant item (1-indexed). 0 if none relevant."""
    rel = set(relevant)
    for rank, item in enumerate(recommended, start=1):
        if item in rel:
            return 1.0 / rank
    return 0.0


def dcg_at_k(relevance_grades: List[float], k: int) -> float:
    """DCG@K: sum_{i=1}^{k} (rel_i / log2(i+1))."""
    grades = relevance_grades[:k]
    if not grades:
        return 0.0
    return sum(g / np.log2(i + 2) for i, g in enumerate(grades))


def ndcg_at_k(
    recommended: List[int],
    relevance_fn: Callable[[int], float],
    k: Optional[int] = None,
) -> float:
    """
    NDCG@K. relevance_fn(item_index) returns relevance grade for that item.
    NDCG = DCG / IDCG (IDCG from ideal ordering by relevance).
    """
    k = k or len(recommended)
    rec_k = recommended[:k]
    grades = [relevance_fn(i) for i in rec_k]
    idcg_grades = sorted(
        [relevance_fn(i) for i in range(len(recommended))],
        reverse=True,
    )[:k]
    dcg = dcg_at_k(grades, k)
    idcg = dcg_at_k(idcg_grades, k)
    if idcg <= 0:
        return 0.0
    return dcg / idcg


def catalog_coverage(
    all_recommendations: List[List[int]],
    catalog_size: int,
) -> float:
    """Fraction of catalog (0..catalog_size-1) that appears in at least one list."""
    if catalog_size == 0:
        return 0.0
    covered = set()
    for rec_list in all_recommendations:
        covered.update(rec_list)
    return len(covered) / catalog_size


def intra_list_diversity(
    similarity_matrix: np.ndarray,
    recommended_indices: List[int],
) -> float:
    """
    Average pairwise (1 - cosine_similarity) among recommended items.
    Higher = more diverse. similarity_matrix[i,j] = cosine sim between i and j.
    """
    if len(recommended_indices) < 2:
        return 0.0
    n = len(recommended_indices)
    total = 0.0
    count = 0
    for i in range(n):
        for j in range(i + 1, n):
            ii, jj = recommended_indices[i], recommended_indices[j]
            sim = float(similarity_matrix[ii, jj])
            total += 1.0 - sim
            count += 1
    return total / count if count else 0.0


def average_diversity(
    similarity_matrix: np.ndarray,
    all_recommendations: List[List[int]],
) -> float:
    """Mean intra-list diversity over all recommendation lists."""
    if not all_recommendations:
        return 0.0
    divs = [
        intra_list_diversity(similarity_matrix, rec)
        for rec in all_recommendations
        if len(rec) >= 2
    ]
    return float(np.mean(divs)) if divs else 0.0
