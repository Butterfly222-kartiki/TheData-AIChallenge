"""
engine/channel_semantic.py

CHANNEL 1 — Semantic Text Relevance ("sees beyond keywords").
The single most important channel. Matches what a candidate actually DID
(career narrative embedding) against what the JD actually NEEDS (semantic
query embeddings from config). 

Aggregation: 0.5 * mean + 0.5 * max cosine similarity across queries.
Using a blend of mean and max addresses two failure modes:
  - Pure mean dilutes strong matches when a JD has many queries of varying
    quality (or a new JD with 3 vs 15 requirements).
  - Pure max ignores breadth of coverage across multiple requirements.
The blend ratio (alpha) is configurable via stuffer_detection.semantic_blend_alpha
in jd_config.yaml; default is 0.5.

Optionally, queries can be weighted by their polarity (must_have queries
weigh more than nice_to_have) via the query_weights parameter. The compiler
generates these weights automatically from its polarity classification.

This is computed in a batched, vectorized way across all candidates at
once (career_embeddings matrix vs query_embeddings matrix) — this is the
~2 second step described in the ranking-time budget.
"""

from __future__ import annotations
import numpy as np
from engine.embedder import embed_texts, cosine_sim_matrix


def embed_queries(semantic_queries: list[str]) -> np.ndarray:
    return embed_texts(semantic_queries)


def compute_semantic_scores(
    career_embeddings: np.ndarray,
    query_embeddings: np.ndarray,
    query_weights: np.ndarray | None = None,
    blend_alpha: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """
    career_embeddings: (N, d) float — candidate career embeddings
    query_embeddings:  (Q, d) float — JD semantic query embeddings
    query_weights:     (Q,) float — optional per-query weights (polarity-
                       derived: must_have=1.0, nice_to_have=0.6, not_wanted=0.0).
                       If None, all queries are weighted equally.
    blend_alpha:       weight of mean vs max: score = alpha*mean + (1-alpha)*max.
                       Default 0.5 balances breadth and depth.

    Returns: ((N,) semantic_score in [0,1], (N, Q) raw sim matrix)
    """
    sims = cosine_sim_matrix(
        career_embeddings.astype(np.float32),
        query_embeddings.astype(np.float32),
    )  # (N, Q)

    if query_weights is not None and len(query_weights) == sims.shape[1]:
        w = np.array(query_weights, dtype=np.float32)
        w_sum = w.sum()
        if w_sum > 1e-8:
            # Weighted mean
            weighted_mean = (sims * w[np.newaxis, :]).sum(axis=1) / w_sum
        else:
            weighted_mean = sims.mean(axis=1)
        # Max is always unweighted (we want the best match regardless of weight)
        raw_max = sims.max(axis=1)
        blended = blend_alpha * weighted_mean + (1.0 - blend_alpha) * raw_max
    else:
        mean_sim = sims.mean(axis=1)   # (N,)
        max_sim = sims.max(axis=1)     # (N,)
        blended = blend_alpha * mean_sim + (1.0 - blend_alpha) * max_sim

    # cosine sim for normalized sentence embeddings is typically in [0,1]-ish
    # for related text, but can dip negative for unrelated text. Rescale
    # defensively into [0,1].
    rescaled = (blended + 1.0) / 2.0
    return np.clip(rescaled, 0.0, 1.0), sims
