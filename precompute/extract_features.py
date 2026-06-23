"""
precompute/extract_features.py

Generates artifacts/features.json: per-candidate skill credibility table,
career metadata, integrity check results (Channel 5), and behavioral
multiplier (Channel 4) — everything JD-independent worth precomputing once.

IMPORTANT CALIBRATION NOTE on the two semantic integrity checks
(title-skill coherence, skill-career coherence):

A fixed absolute similarity threshold (e.g. "0.25") only makes sense for
one specific embedding backend's similarity scale. Real neural embeddings
(sentence-transformers) and the TF-IDF+LSA fallback produce similarity
values on genuinely different scales, especially for very short text
("ML Engineer" vs a 5-word skill list) — an absolute threshold calibrated
for one will misfire on the other (we found this empirically: it flagged
genuinely strong candidates as incoherent under the LSA fallback).

So instead of a fixed threshold, this script computes the EMPIRICAL
distribution of each coherence similarity across the whole candidate pool
and uses a low percentile of that distribution (default: 5th percentile)
as the cutoff. This self-calibrates to whichever embedding backend is
actually active, and is robust to swapping models later.

Also reuses the precomputed career/skill embeddings (if
embed_candidates.py has already been run) instead of re-embedding, and
batches the title embeddings — this is what makes pass 2 fast even at
100K scale (single-candidate-at-a-time embedding calls were the dominant
cost before this rewrite).

Usage:
    python precompute/extract_features.py \
        --candidates path/to/candidates.jsonl \
        --artifacts-dir artifacts \
        --integrity-config config/jd_config.yaml
"""

from __future__ import annotations
import argparse
import json
import math
import os
import sys
import time

import numpy as np
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from engine.data import iter_candidates, cid, proficiency_weight, total_career_months, \
    career_history, skills as get_skills, parse_date, career_text, skill_text, current_title
from engine.embedder import get_embedder, is_using_fallback, ensure_corpus_fitted, embed_texts, cosine_sim_matrix
from engine.channel_integrity import compute_integrity
from engine.channel_behavioral import compute_behavioral


def skill_credibility_table(candidate: dict) -> dict:
    table = {}
    for s in get_skills(candidate):
        name = s.get("name", "")
        if not name:
            continue
        prof = proficiency_weight(s.get("proficiency", "beginner"))
        dur = max(0, int(s.get("duration_months", 0) or 0))
        endorse = max(0, int(s.get("endorsements", 0) or 0))
        table[name] = round(prof * math.log(1 + dur) * math.log(1 + endorse), 4)
    return table


def career_metadata(candidate: dict) -> dict:
    career = career_history(candidate)
    industries = list({r.get("industry", "") for r in career if r.get("industry")})
    role_durations = [int(r.get("duration_months", 0) or 0) for r in career]
    return {
        "industries": industries,
        "total_career_months": total_career_months(candidate),
        "role_durations": role_durations,
        "max_role_duration": max(role_durations) if role_durations else 0,
        "n_roles": len(career),
    }


def load_npz(path):
    data = np.load(path, allow_pickle=True)
    return list(data["ids"]), data["embeddings"].astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", required=True)
    ap.add_argument("--artifacts-dir", default="artifacts")
    ap.add_argument("--integrity-config", default=None)
    ap.add_argument("--coherence-percentile", type=float, default=5.0,
                     help="percentile of the empirical similarity distribution used as the coherence cutoff")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    os.makedirs(args.artifacts_dir, exist_ok=True)

    cfg = {}
    if args.integrity_config:
        with open(args.integrity_config, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
    integrity_params = dict(cfg.get("integrity", {}))
    embed_cfg = cfg.get("embedding", {"model_name": "all-MiniLM-L6-v2", "dimension": 384})

    print("[extract_features] loading candidate pool ...")
    t0 = time.time()
    candidates = []
    for i, c in enumerate(iter_candidates(args.candidates)):
        if args.limit and i >= args.limit:
            break
        candidates.append(c)
    n = len(candidates)
    ids = [cid(c) for c in candidates]
    print(f"[extract_features] loaded {n} candidates in {time.time()-t0:.1f}s")

    max_date = None
    for c in candidates:
        d = parse_date(c.get("redrob_signals", {}).get("last_active_date"))
        if d and (max_date is None or d > max_date):
            max_date = d
    print(f"[extract_features] reference_date = {max_date}")

    # --- Load or compute career/skill embeddings (reuse if precomputed) ---
    career_path = os.path.join(args.artifacts_dir, "career_embeddings.npz")
    skill_path = os.path.join(args.artifacts_dir, "skill_embeddings.npz")
    # Item 5: pass artifacts_dir so local vendored model weights are found first
    get_embedder(embed_cfg["model_name"], embed_cfg["dimension"], artifacts_dir=args.artifacts_dir)
    if is_using_fallback():
        # Try to load the LSA model already fit during embed_candidates.py
        # FIRST, before any embed_texts() call lazily falls through to the
        # much weaker hash tier for lack of an artifacts_dir to find it.
        ensure_corpus_fitted(None, artifacts_dir=args.artifacts_dir)

    if os.path.exists(career_path) and os.path.exists(skill_path):
        print("[extract_features] reusing precomputed career/skill embeddings ...")
        ids_career, career_embs = load_npz(career_path)
        _, skill_embs = load_npz(skill_path)
        assert ids_career[:n] == ids, "embedding ids don't match candidate order; re-run embed_candidates.py"
        career_embs, skill_embs = career_embs[:n], skill_embs[:n]
    else:
        print("[extract_features] no precomputed embeddings found; computing now ...")
        if is_using_fallback():
            corpus = [career_text(c) for c in candidates] + [skill_text(c) for c in candidates]
            ensure_corpus_fitted(corpus, artifacts_dir=args.artifacts_dir)
        career_embs = embed_texts([career_text(c) for c in candidates])
        skill_embs = embed_texts([skill_text(c) for c in candidates])

    if is_using_fallback() and not os.path.exists(os.path.join(args.artifacts_dir, "lsa_model.pkl")):
        corpus = [career_text(c) for c in candidates] + [skill_text(c) for c in candidates]
        ensure_corpus_fitted(corpus, artifacts_dir=args.artifacts_dir)
    if is_using_fallback():
        print("[extract_features] WARNING: not using the real sentence-transformers model.")

    print("[extract_features] batch-embedding titles ...")
    title_embs = embed_texts([current_title(c) for c in candidates])

    # --- Batched coherence similarity (no per-candidate embed calls) ---
    print("[extract_features] computing title-skill and skill-career coherence similarities ...")
    title_skill_sim = ((np.sum(title_embs * skill_embs, axis=1) /
                        (np.linalg.norm(title_embs, axis=1) * np.linalg.norm(skill_embs, axis=1) + 1e-8)) + 1.0) / 2.0
    skill_career_sim = ((np.sum(skill_embs * career_embs, axis=1) /
                         (np.linalg.norm(skill_embs, axis=1) * np.linalg.norm(career_embs, axis=1) + 1e-8)) + 1.0) / 2.0

    pct = args.coherence_percentile
    title_skill_threshold = float(np.percentile(title_skill_sim, pct))
    skill_career_threshold = float(np.percentile(skill_career_sim, pct))
    print(f"[extract_features] adaptive thresholds (p{pct}): "
          f"title_skill_coherence_min={title_skill_threshold:.4f}, "
          f"skill_career_coherence_min={skill_career_threshold:.4f} "
          f"(configured defaults were {integrity_params.get('title_skill_coherence_min')}, "
          f"{integrity_params.get('skill_career_coherence_min')})")

    integrity_params["title_skill_coherence_min"] = title_skill_threshold
    integrity_params["skill_career_coherence_min"] = skill_career_threshold

    print("[extract_features] computing per-candidate features ...")
    t0 = time.time()
    features = {}
    for i, c in enumerate(candidates):
        integrity = compute_integrity(
            c, integrity_params,
            title_emb=title_embs[i:i+1], skill_emb=skill_embs[i:i+1], career_emb=career_embs[i:i+1],
        )
        behavioral = compute_behavioral(c, cfg.get("behavioral"), reference_date=max_date)
        features[ids[i]] = {
            "skill_credibility": skill_credibility_table(c),
            "career_metadata": career_metadata(c),
            "integrity": integrity,
            "behavioral": behavioral,
        }
        if (i + 1) % 20000 == 0:
            print(f"[extract_features]   ... {i+1} candidates processed ({time.time()-t0:.1f}s)")

    out_path = os.path.join(args.artifacts_dir, "features.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "reference_date": str(max_date),
            "coherence_thresholds": {
                "title_skill_coherence_min": title_skill_threshold,
                "skill_career_coherence_min": skill_career_threshold,
            },
            "candidates": features,
        }, f)

    print(f"[extract_features] wrote {out_path} "
          f"({os.path.getsize(out_path)/1e6:.1f} MB) for {len(features)} candidates "
          f"in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
