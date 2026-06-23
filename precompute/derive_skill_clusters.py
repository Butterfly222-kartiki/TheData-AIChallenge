"""
precompute/derive_skill_clusters.py — EMBEDDING-DERIVED SKILL CLUSTERS

Replaces hand-curated skill_clusters in jd_config.yaml with automatically
derived clusters using embedding similarity between the pool's skill
vocabulary and the JD's polarity-bucketed requirement sentences.

Algorithm:
  1. Collect the full skill vocabulary from the candidate pool (all unique
     skill name strings).
  2. Embed the vocabulary (one batch, ~133–300 vectors for typical pools).
  3. Embed the JD's must_have, nice_to_have, and not_wanted sentences
     (already parsed by jd_compiler.py logic).
  4. For each skill, compute cosine similarity to the centroid of each
     polarity group.
  5. Assign:
       core             if closest to must_have centroid   AND sim ≥ floor
       secondary        if closest to nice_to_have centroid AND sim ≥ floor
       nice_to_have     if sim to nice_to_have is second-best AND sim ≥ floor
       domain_mismatch  if closest to not_wanted AND domain-mismatch heuristic
       business_mismatch if closest to not_wanted AND business-mismatch heuristic
  6. Apply manual overrides from config's skill_overrides block.
  7. Writes derived clusters into the config YAML (in-place, preserving
     all other keys) AND into artifacts/skill_clusters.yaml for review.

Usage:
    python precompute/derive_skill_clusters.py \\
        --candidates path/to/candidates.jsonl \\
        --config config/jd_config.yaml \\
        --artifacts-dir artifacts \\
        [--similarity-floor 0.35] \\
        [--dry-run]

Options:
    --similarity-floor  Minimum cosine similarity to assign a skill to any
                        cluster. Skills below this threshold in all polarities
                        get weight=0 (irrelevant for this JD). Default: 0.35.
    --dry-run           Print derived clusters to stdout without modifying
                        the config YAML.
"""

from __future__ import annotations
import argparse
import json
import os
import sys
import numpy as np
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from engine.embedder import get_embedder, embed_texts, is_using_fallback, ensure_corpus_fitted
# Import jd_compiler's parse_jd so we share the same polarity parsing logic
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from jd_compiler import parse_jd


# ---------------------------------------------------------------------------
# Domain / business mismatch classification heuristics
# These distinguish the *type* of not_wanted skill (domain mismatch = a
# valid field just not relevant here; business mismatch = non-technical role
# identity that's always penalized).
# ---------------------------------------------------------------------------
_DOMAIN_MISMATCH_KEYWORDS = {
    "vision", "image", "object", "yolo", "opencv", "cnn", "gan", "diffusion",
    "speech", "asr", "tts", "audio", "lidar", "slam", "robotics", "autonomous",
}
_BUSINESS_MISMATCH_KEYWORDS = {
    "sales", "accounting", "tally", "sap", "marketing", "seo",
    "content writing", "crm", "illustrator", "photoshop", "six sigma",
    "salesforce", "erp", "supply chain", "logistics", "legal",
}


def _is_domain_mismatch(skill: str) -> bool:
    sl = skill.lower()
    return any(kw in sl for kw in _DOMAIN_MISMATCH_KEYWORDS)


def _is_business_mismatch(skill: str) -> bool:
    sl = skill.lower()
    return any(kw in sl for kw in _BUSINESS_MISMATCH_KEYWORDS)


def collect_skill_vocabulary(candidates_path: str, limit: int | None = None) -> list[str]:
    """Scan candidate pool and collect unique skill name strings."""
    vocab: set[str] = set()
    with open(candidates_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if limit and i >= limit:
                break
            try:
                c = json.loads(line)
                for sk in c.get("skills", []):
                    name = (sk.get("name") or "").strip()
                    if name:
                        vocab.add(name)
            except (json.JSONDecodeError, AttributeError):
                continue
    return sorted(vocab)


def _centroid(matrix: np.ndarray) -> np.ndarray:
    """L2-normalized centroid of a set of embeddings."""
    if matrix.shape[0] == 0:
        return np.zeros(matrix.shape[1], dtype=np.float32)
    c = matrix.mean(axis=0)
    norm = np.linalg.norm(c)
    return (c / norm).astype(np.float32) if norm > 1e-8 else c.astype(np.float32)


def derive_clusters(
    skill_vocab: list[str],
    must_have_sentences: list[str],
    nice_to_have_sentences: list[str],
    not_wanted_sentences: list[str],
    similarity_floor: float = 0.35,
    manual_overrides: dict | None = None,
) -> dict[str, list[str]]:
    """
    Core clustering function. Returns a dict with keys:
      core, secondary, nice_to_have,
      domain_mismatch_anti_skills, business_mismatch_anti_skills
    """
    # Embed all sentences per polarity
    all_query_texts = must_have_sentences + nice_to_have_sentences + not_wanted_sentences
    if not all_query_texts:
        return {
            "core": [], "secondary": [], "nice_to_have": [],
            "domain_mismatch_anti_skills": [], "business_mismatch_anti_skills": [],
        }

    print(f"[derive_skill_clusters] embedding {len(skill_vocab)} skills "
          f"and {len(all_query_texts)} JD sentences ...")
    skill_embs = embed_texts(skill_vocab)  # (V, d)

    def embed_group(sentences: list[str]) -> np.ndarray:
        if not sentences:
            return np.zeros((0, skill_embs.shape[1]), dtype=np.float32)
        return embed_texts(sentences).astype(np.float32)

    must_embs = embed_group(must_have_sentences)
    nice_embs = embed_group(nice_to_have_sentences)
    not_embs = embed_group(not_wanted_sentences)

    # Compute centroids
    must_centroid = _centroid(must_embs)    # (d,)
    nice_centroid = _centroid(nice_embs)    # (d,)
    not_centroid = _centroid(not_embs)      # (d,)

    # Cosine sims: skill_embs @ centroid (both L2-normalized)
    def sims_to_centroid(centroid: np.ndarray) -> np.ndarray:
        if np.linalg.norm(centroid) < 1e-8:
            return np.zeros(len(skill_vocab), dtype=np.float32)
        return skill_embs @ centroid  # (V,) — already normalized

    must_sims = sims_to_centroid(must_centroid)
    nice_sims = sims_to_centroid(nice_centroid)
    not_sims = sims_to_centroid(not_centroid)

    # Assign each skill
    clusters: dict[str, list[str]] = {
        "core": [],
        "secondary": [],
        "nice_to_have": [],
        "domain_mismatch_anti_skills": [],
        "business_mismatch_anti_skills": [],
    }

    # Manual overrides (skill_name → cluster_name)
    overrides: dict[str, str] = {}
    if manual_overrides:
        for cluster_name, skills in manual_overrides.items():
            for sk in (skills or []):
                overrides[sk.strip().lower()] = cluster_name

    for i, skill in enumerate(skill_vocab):
        skill_lower = skill.strip().lower()

        # Check manual override first
        if skill_lower in overrides:
            target = overrides[skill_lower]
            if target in clusters:
                clusters[target].append(skill)
            continue

        ms, ns, nts = float(must_sims[i]), float(nice_sims[i]), float(not_sims[i])

        # If closest to not_wanted and above floor → anti-skill
        best_positive = max(ms, ns)
        if nts > best_positive and nts >= similarity_floor:
            if _is_domain_mismatch(skill):
                clusters["domain_mismatch_anti_skills"].append(skill)
            elif _is_business_mismatch(skill):
                clusters["business_mismatch_anti_skills"].append(skill)
            # Else: not_wanted but unclassified — skip (don't add to anti-lists
            # without a clear category; conservative approach)
            continue

        # Otherwise assign to positive clusters if above floor
        if ms >= similarity_floor and ms >= ns:
            clusters["core"].append(skill)
        elif ns >= similarity_floor and ns > ms:
            clusters["secondary"].append(skill)
        elif ms >= similarity_floor * 0.8:  # slightly lower bar for nice_to_have
            clusters["nice_to_have"].append(skill)
        # else: below floor in all polarities — unassigned (weight=0)

    # Sort for deterministic output
    for k in clusters:
        clusters[k] = sorted(set(clusters[k]))

    return clusters


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", required=True, help="Path to candidates.jsonl")
    ap.add_argument("--config", default="config/jd_config.yaml", help="JD config YAML")
    ap.add_argument("--artifacts-dir", default="artifacts")
    ap.add_argument("--similarity-floor", type=float, default=0.35,
                    help="Min cosine sim to assign a skill to any cluster")
    ap.add_argument("--limit", type=int, default=None,
                    help="Cap candidate scan at N (for quick testing)")
    ap.add_argument("--dry-run", action="store_true",
                    help="Print derived clusters without modifying the config")
    args = ap.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    embed_cfg = cfg.get("embedding", {"model_name": "all-MiniLM-L6-v2", "dimension": 384})
    get_embedder(embed_cfg.get("model_name", "all-MiniLM-L6-v2"),
                 embed_cfg.get("dimension", 384),
                 artifacts_dir=args.artifacts_dir)
    if is_using_fallback():
        ensure_corpus_fitted(None, artifacts_dir=args.artifacts_dir)
        print("[derive_skill_clusters] WARNING: running with TF-IDF+LSA fallback; "
              "cluster quality will be lower than with the real model. "
              "Run scripts/download_model.py first for best results.")

    print(f"[derive_skill_clusters] scanning candidate pool for skill vocabulary ...")
    skill_vocab = collect_skill_vocabulary(args.candidates, limit=args.limit)
    print(f"[derive_skill_clusters] found {len(skill_vocab)} unique skills")

    # Parse JD text to get polarized sentences
    # We reconstruct JD text from the config's semantic_queries as a proxy
    # if a raw JD file isn't provided. Better: use the actual must_have /
    # nice_to_have / not_wanted sentences stored by the compiler.
    # For now, use semantic_queries as must_have and emit nice_to_have/not_wanted
    # from config if present.
    must_have = cfg.get("semantic_queries", [])
    nice_to_have = cfg.get("_jd_sentences", {}).get("nice_to_have", [])
    not_wanted = cfg.get("_jd_sentences", {}).get("not_wanted", [])

    # Enhance with JD raw text if stored in config
    jd_raw = cfg.get("_jd_raw_text", "")
    if jd_raw:
        buckets = parse_jd(jd_raw)
        must_have = must_have or buckets["must_have"]
        nice_to_have = nice_to_have or buckets["nice_to_have"]
        not_wanted = not_wanted or buckets["not_wanted"]

    print(f"[derive_skill_clusters] polarity sentences: "
          f"{len(must_have)} must / {len(nice_to_have)} nice / {len(not_wanted)} not_wanted")

    manual_overrides = cfg.get("skill_overrides", {})
    clusters = derive_clusters(
        skill_vocab, must_have, nice_to_have, not_wanted,
        similarity_floor=args.similarity_floor,
        manual_overrides=manual_overrides,
    )

    print("[derive_skill_clusters] derived clusters:")
    for cluster_name, skills in clusters.items():
        print(f"  {cluster_name:35s}: {len(skills):3d} skills")

    # Write artifacts/skill_clusters.yaml for manual review
    os.makedirs(args.artifacts_dir, exist_ok=True)
    clusters_path = os.path.join(args.artifacts_dir, "skill_clusters.yaml")
    with open(clusters_path, "w", encoding="utf-8") as f:
        yaml.dump({"skill_clusters": clusters,
                   "similarity_floor": args.similarity_floor}, f,
                  default_flow_style=False, allow_unicode=True)
    print(f"[derive_skill_clusters] wrote {clusters_path} for review")

    if not args.dry_run:
        # Patch skill_clusters in the config YAML in-place
        cfg["skill_clusters"] = clusters
        with open(args.config, "w", encoding="utf-8") as f:
            yaml.dump(cfg, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
        print(f"[derive_skill_clusters] updated skill_clusters in {args.config}")
        print(f"[derive_skill_clusters] NEXT STEP: run rank.py")
    else:
        print("[derive_skill_clusters] --dry-run: config not modified")
        print(yaml.dump({"skill_clusters": clusters}, default_flow_style=False))


if __name__ == "__main__":
    main()
