"""
engine/skill_synonyms.py

Maps rare, plain-language skill-name variants found in the real Redrob
dataset to their canonical cluster names. Without this, exact-string skill
matching would badly under-credit "Tier 5" candidates who describe the
same expertise with different words — exactly the trap the JD calls out:

  "A Tier 5 candidate may not use the words 'RAG' or 'Pinecone'... if
   their career history shows they built a recommendation system..."

ITEM 2 — Embedding-derived skill clusters:
  With embedding-based cluster assignment (precompute/derive_skill_clusters.py),
  synonym handling falls out automatically — a skill with a different name
  but similar embedding is assigned to the same cluster as its canonical
  equivalent. The static map below is now used as a MANUAL OVERRIDE dict:
  any skill listed here bypasses the embedding assignment and goes directly
  into the specified cluster.

  To add a manual override, add entries to skill_overrides in jd_config.yaml:
    skill_overrides:
      core:
        - "My Custom Skill Name"

  The static SKILL_SYNONYMS dict is kept for backward compatibility with
  configs that haven't run derive_skill_clusters.py yet, and for cases
  where the embedding model assigns a known synonym to the wrong cluster.

This map was built by scanning the actual candidate pool's skill-name
vocabulary (133 unique skill strings total) and identifying the handful
of single-digit-frequency variants that clearly mean the same thing as a
canonical cluster skill. It is intentionally short and literal — it is
NOT a general synonym dictionary, it's calibrated to this dataset.

If you point this system at a different dataset, re-run
`scripts/inspect_skill_vocab.py` (or similar) to discover its own rare
variants before trusting this map. Or use derive_skill_clusters.py with
the real model — synonym handling then falls out automatically.
"""

from __future__ import annotations

SKILL_SYNONYMS: dict[str, str] = {
    "information retrieval systems": "Information Retrieval",
    "search backend": "Search Infrastructure",
    "text encoders": "Sentence Transformers",
    "vector representations": "Embeddings",
    "content matching": "Recommendation Systems",
    "model adaptation": "Fine-tuning LLMs",
    "ranking systems": "Learning to Rank",
    "search & discovery": "Semantic Search",
    "workflow orchestration": "MLOps",
    "search infrastructure": "Search Infrastructure",
    "indexing algorithms": "Vector Search",
    "open-source ml libraries": "Open-source ML libraries",
    "natural language processing": "NLP",
    "document processing": "Information Retrieval",
}

# Config-driven overrides loaded at runtime (populated by load_overrides())
_RUNTIME_OVERRIDES: dict[str, str] = {}


def load_overrides(skill_overrides: dict | None) -> None:
    """
    Load manual overrides from jd_config.yaml's skill_overrides block.
    Maps skill_name.lower() → cluster_name.

    skill_overrides format:
      core:
        - "My Custom Skill"
      secondary:
        - "Another Skill"
    """
    global _RUNTIME_OVERRIDES
    _RUNTIME_OVERRIDES = {}
    if not skill_overrides:
        return
    for cluster_name, skills in skill_overrides.items():
        for sk in (skills or []):
            _RUNTIME_OVERRIDES[(sk or "").strip().lower()] = cluster_name


def canonicalize(skill_name: str) -> str:
    """
    Returns the canonical cluster name for a skill:
    1. Checks runtime overrides (from config's skill_overrides) first.
    2. Falls back to the static SKILL_SYNONYMS map.
    3. Returns the original name unchanged if no match.
    """
    key = (skill_name or "").strip().lower()
    if key in _RUNTIME_OVERRIDES:
        return _RUNTIME_OVERRIDES[key]
    return SKILL_SYNONYMS.get(key, skill_name)
