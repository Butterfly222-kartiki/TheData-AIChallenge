"""
engine/embedder.py

Single shared embedding utility. Wraps a sentence-transformers model.

Active model: paraphrase-MiniLM-L3-v2
  - 3 transformer layers (vs 6 in all-MiniLM-L6-v2)
  - Same 384-dim output — fully compatible with existing pipeline
  - ~17MB on disk vs ~91MB for L6
  - ~3x faster on CPU-only inference (relevant for 100K-candidate runs)
  - Slightly lower cosine similarity fidelity than L6, but meaningfully
    better than TF-IDF+LSA on semantic equivalence tasks

Why centralized: every channel that touches text (semantic, integrity's
title-skill/skill-career coherence, stuffer detection) must use the exact
same model and the exact same preprocessing, or cosine similarities
computed in different places stop being comparable.

THREE-TIER FALLBACK:
  Tier 1 (best): sentence-transformers/all-MiniLM-L6-v2 — true neural
    semantic embeddings. Loaded from a local path if present (zero network)
    via scripts/download_model.py, otherwise falls back to HuggingFace Hub
    download. See scripts/download_model.py for the one-time offline setup.
  Tier 2 (good, no internet ever required): TF-IDF + Truncated SVD (LSA)
    fitted on the candidate corpus itself. This is a real, legitimate,
    CPU-only "semantic-lite" retrieval technique (latent semantic
    analysis) — meaningfully better than raw keyword matching because the
    SVD captures co-occurrence patterns between related terms, and TF-IDF
    naturally downweights generic words. It activates automatically if
    Tier 1 is unavailable and a corpus has been registered via
    `ensure_corpus_fitted`.
  Tier 3 (last resort): a deterministic hash-projection embedding used
    only if scikit-learn itself is unavailable. This has no real semantic
    understanding and exists purely so the pipeline never crashes.

Whichever tier is active, a loud warning is printed when it's not Tier 1.
"""

from __future__ import annotations
import hashlib
import logging
import os
import pickle
import re
import numpy as np

logger = logging.getLogger("embedder")

_MODEL = None
_MODEL_NAME = None
_DIM = 384
_TIER = None  # "real" | "lsa" | "hash"

_LSA_VECTORIZER = None
_LSA_SVD = None


def _try_load_real_model(model_name: str, artifacts_dir: str | None = None):
    """
    Attempt to load the sentence-transformers model. Checks for a vendored
    local copy first (artifacts_dir/models/<model_name>/), then falls back
    to the HuggingFace Hub download. Returns None on any failure.
    """
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as e:
        logger.warning(
            "sentence-transformers not installed (%s). "
            "Will attempt the TF-IDF+LSA fallback instead. "
            "Run: pip install sentence-transformers", e
        )
        return None

    # 1. Try vendored local path first (zero network)
    if artifacts_dir:
        local_path = os.path.join(artifacts_dir, "models", model_name)
        if os.path.isdir(local_path):
            try:
                model = SentenceTransformer(local_path)
                logger.info("Loaded sentence-transformers model from local path: %s", local_path)
                return model
            except Exception as e:
                logger.warning(
                    "Local model at '%s' failed to load (%s); trying hub ...", local_path, e
                )

    # 2. Try HuggingFace Hub (requires internet on first use; cached thereafter)
    try:
        model = SentenceTransformer(model_name)
        return model
    except Exception as e:
        logger.warning(
            "Could not load sentence-transformers model '%s' (%s). "
            "Will attempt the TF-IDF+LSA fallback instead. "
            "Run scripts/download_model.py for zero-network offline setup.", model_name, e
        )
        return None


def get_embedder(model_name: str = "all-MiniLM-L6-v2",
                 dimension: int = 384,
                 artifacts_dir: str | None = "artifacts"):
    """
    Initialize (once) and return the embedding function. Idempotent.

    artifacts_dir: if provided, checks for a locally vendored model at
      <artifacts_dir>/models/<model_name>/ before hitting the network.
      Run scripts/download_model.py once to populate this for offline use.
    """
    global _MODEL, _MODEL_NAME, _DIM, _TIER
    if _TIER is not None:
        return embed_texts

    _DIM = dimension
    _MODEL_NAME = model_name
    _MODEL = _try_load_real_model(model_name, artifacts_dir=artifacts_dir)
    _TIER = "real" if _MODEL is not None else None
    return embed_texts


def is_using_fallback() -> bool:
    return _TIER != "real"


def active_tier() -> str:
    return _TIER or "unset"


# ---------------------------------------------------------------------------
# Tier 2: TF-IDF + LSA — fit once on the candidate corpus, persist, reuse.
# ---------------------------------------------------------------------------

def ensure_corpus_fitted(corpus_texts: list[str] | None = None, artifacts_dir: str | None = None) -> None:
    """
    Call this once, early, whenever Tier 1 (real model) might be
    unavailable. If a previously-fitted LSA model exists at
    {artifacts_dir}/lsa_model.pkl, loads it. Otherwise, if corpus_texts is
    provided, fits a new one and saves it there for reuse by later runs
    (e.g. rank.py reusing the model fitted during precompute).
    """
    global _TIER, _LSA_VECTORIZER, _LSA_SVD
    if _TIER == "real":
        return  # Tier 1 already active, no need for LSA

    model_path = os.path.join(artifacts_dir, "lsa_model.pkl") if artifacts_dir else None

    if model_path and os.path.exists(model_path):
        with open(model_path, "rb") as f:
            saved = pickle.load(f)
        _LSA_VECTORIZER, _LSA_SVD = saved["vectorizer"], saved["svd"]
        _TIER = "lsa"
        logger.warning("Using TF-IDF+LSA embedding fallback (loaded from %s). "
                        "Run scripts/download_model.py for zero-network Tier 1 setup.", model_path)
        return

    if corpus_texts:
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.decomposition import TruncatedSVD

            vectorizer = TfidfVectorizer(
                max_features=30000, ngram_range=(1, 2), sublinear_tf=True,
                min_df=2, stop_words="english",
            )
            tfidf = vectorizer.fit_transform(corpus_texts)
            n_components = max(2, min(_DIM, tfidf.shape[1] - 1, tfidf.shape[0] - 1))
            svd = TruncatedSVD(n_components=n_components, random_state=42)
            svd.fit(tfidf)

            _LSA_VECTORIZER, _LSA_SVD = vectorizer, svd
            _TIER = "lsa"
            logger.warning("Fitted TF-IDF+LSA embedding fallback on %d documents (%d components). "
                            "Run scripts/download_model.py for zero-network Tier 1 setup.",
                            len(corpus_texts), n_components)
            if model_path:
                os.makedirs(os.path.dirname(model_path), exist_ok=True)
                with open(model_path, "wb") as f:
                    pickle.dump({"vectorizer": vectorizer, "svd": svd}, f)
            return
        except Exception as e:
            logger.warning("TF-IDF+LSA fit failed (%s); falling back to hash embedding.", e)

    _TIER = "hash"
    logger.warning("Using deterministic hash-projection fallback (no real semantics). "
                    "This should only happen if scikit-learn is unavailable.")


def _lsa_embed(texts: list[str]) -> np.ndarray:
    tfidf = _LSA_VECTORIZER.transform(texts)
    reduced = _LSA_SVD.transform(tfidf).astype(np.float32)
    if reduced.shape[1] < _DIM:
        pad = np.zeros((reduced.shape[0], _DIM - reduced.shape[1]), dtype=np.float32)
        reduced = np.concatenate([reduced, pad], axis=1)
    elif reduced.shape[1] > _DIM:
        reduced = reduced[:, :_DIM]
    norms = np.linalg.norm(reduced, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return reduced / norms


# ---------------------------------------------------------------------------
# Tier 3: deterministic hash-projection (true last resort)
# ---------------------------------------------------------------------------

_word_re = re.compile(r"[a-z0-9]+")
_token_cache: dict[str, tuple] = {}


def _token_vector_components(tok: str):
    """Deterministic (indices, signs) for a token, cached across calls."""
    cached = _token_cache.get(tok)
    if cached is not None:
        return cached
    h = hashlib.sha256(tok.encode("utf-8")).digest()
    idx = np.frombuffer(h[:16], dtype=np.uint8).astype(np.int64) % _DIM
    signs = np.where(np.frombuffer(h[16:24], dtype=np.uint8) % 2 == 0, 1.0, -1.0)
    n = min(len(idx), len(signs))
    result = (idx[:n], signs[:n])
    if len(_token_cache) < 200000:
        _token_cache[tok] = result
    return result


def _fallback_embed(texts: list[str]) -> np.ndarray:
    out = np.zeros((len(texts), _DIM), dtype=np.float32)
    for i, t in enumerate(texts):
        t = (t or "").lower()
        tokens = _word_re.findall(t)
        if not tokens:
            continue
        vec = np.zeros(_DIM, dtype=np.float32)
        for tok in tokens:
            idx, signs = _token_vector_components(tok)
            np.add.at(vec, idx, signs)
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        out[i] = vec
    return out


def embed_texts(texts: list[str], batch_size: int = 64, show_progress: bool = False) -> np.ndarray:
    """
    Embed a list of strings -> (N, dim) float32 numpy array, L2-normalized rows.
    Tier is selected by get_embedder() + ensure_corpus_fitted(); if neither
    has been called, defaults to the hash fallback (Tier 3) for safety.
    """
    global _TIER
    if _TIER is None:
        get_embedder()
    if _TIER is None or (_TIER != "real" and _LSA_VECTORIZER is None):
        # Tier 1 unavailable and no corpus fit attempted/succeeded yet.
        ensure_corpus_fitted(None, None)

    if _TIER == "real":
        embs = _MODEL.encode(
            texts, batch_size=batch_size, show_progress_bar=show_progress,
            normalize_embeddings=True, convert_to_numpy=True,
        )
        return embs.astype(np.float32)

    if _TIER == "lsa":
        return _lsa_embed(texts)

    return _fallback_embed(texts)


def cosine_sim_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    a: (N, d) candidate embeddings (assumed L2-normalized)
    b: (Q, d) query embeddings (assumed L2-normalized)
    returns (N, Q) cosine similarity matrix.
    If rows aren't normalized, normalizes defensively.
    """
    a_norm = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-8)
    b_norm = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-8)
    return a_norm @ b_norm.T


def cosine_sim_pairwise(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    a: (N, d), b: (N, d) — same N, row-wise cosine similarity -> (N,)
    """
    a_norm = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-8)
    b_norm = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-8)
    return np.sum(a_norm * b_norm, axis=1)
