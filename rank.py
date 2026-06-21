"""
rank.py — MAIN ENTRY POINT.

Loads config + pre-computed artifacts, runs the five-channel scoring
engine, fuses scores, applies stuffer detection, generates reasoning, and
writes the final submission CSV in the EXACT format required by
validate_submission.py:

  header: candidate_id,rank,score,reasoning
  exactly 100 data rows, ranks 1-100 unique, score non-increasing by rank,
  ties broken by candidate_id ascending.

Changes from original:
  Item 2: loads derived skill_clusters from artifacts/skill_clusters.yaml
          if present (falls back to config's static clusters).
  Item 4b: computes percentile-based score floors from top-500 distribution
           and passes them to generate_reasoning().
  Item 5: passes artifacts_dir to get_embedder() so local vendored model
          weights are found before hitting the network.
  Item 6: passes query polarity weights + blend_alpha to semantic scoring;
          passes ramp_width to compute_stuffer_penalty().

Usage:
    python rank.py \\
        --candidates path/to/candidates.jsonl \\
        --config config/jd_config.yaml \\
        --artifacts-dir artifacts \\
        --out submission.csv
"""

from __future__ import annotations
import argparse
import csv
import json
import os
import sys
import time
from datetime import date

import numpy as np
import yaml

sys.path.insert(0, os.path.dirname(__file__))
from engine.data import iter_candidates, cid, career_text, skill_text, current_title, parse_date
from engine.embedder import get_embedder, embed_texts, cosine_sim_matrix, is_using_fallback, active_tier
from engine.channel_semantic import compute_semantic_scores
from engine.channel_skills import compute_skill_score
from engine.channel_career import compute_career_score
from engine.channel_behavioral import compute_behavioral
from engine.channel_integrity import compute_integrity
from engine.fusion import compute_stuffer_penalty, fuse
from engine.reasoning import generate_reasoning
from engine.skill_synonyms import load_overrides


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_npz_embeddings(path: str):
    data = np.load(path, allow_pickle=True)
    return data["ids"], data["embeddings"].astype(np.float32)


def load_derived_skill_clusters(artifacts_dir: str, config_clusters: dict) -> dict:
    """
    Load embedding-derived skill clusters from artifacts/skill_clusters.yaml
    if present (written by precompute/derive_skill_clusters.py).
    Falls back to the static config clusters if not found.

    This implements Item 2: derived clusters replace hand-curated lists,
    with the static config as fallback for runs that haven't derived yet.
    """
    clusters_path = os.path.join(artifacts_dir, "skill_clusters.yaml")
    if os.path.exists(clusters_path):
        with open(clusters_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        derived = data.get("skill_clusters", {})
        if derived:
            print(f"[rank] loaded derived skill clusters from {clusters_path} "
                  f"({sum(len(v) for v in derived.values())} skills total)")
            return derived
    return config_clusters


def compute_score_floors(
    semantic_scores: np.ndarray,
    skill_scores: np.ndarray,
    career_scores: np.ndarray,
    raw_scores: np.ndarray,
    top_n: int = 500,
    percentile: float = 25.0,
) -> dict:
    """
    Compute percentile-based score floors from the top-N candidates by
    raw_score. These replace the hardcoded LSA-calibrated constants in
    reasoning.py, self-calibrating to whichever embedding backend is active
    and whichever JD's query distribution is in use.

    Item 4b: floors are the 25th percentile of the top-500 raw-score
    candidates' per-channel scores.
    """
    n = len(raw_scores)
    top_idx = np.argsort(raw_scores)[::-1][:min(top_n, n)]
    return {
        "semantic": float(np.percentile(semantic_scores[top_idx], percentile)),
        "skill": float(np.percentile(skill_scores[top_idx], percentile)),
        "career": float(np.percentile(career_scores[top_idx], percentile)),
    }


def build_query_weights(cfg: dict) -> np.ndarray | None:
    """
    Build per-query weights from the config's query_polarities if present
    (written by the JD compiler). Returns None if no polarity info available
    (all queries get equal weight).

    Polarity weights: must_have=1.0, nice_to_have=0.6, not_wanted=0.0
    """
    polarities = cfg.get("query_polarities")
    if not polarities:
        return None
    weight_map = {"must_have": 1.0, "nice_to_have": 0.6, "not_wanted": 0.0, "neutral": 0.8}
    weights = [weight_map.get(p, 0.8) for p in polarities]
    return np.array(weights, dtype=np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", required=True)
    ap.add_argument("--config", default="config/jd_config.yaml")
    ap.add_argument("--artifacts-dir", default="artifacts")
    ap.add_argument("--out", default="submission.csv")
    ap.add_argument("--limit", type=int, default=None, help="optional cap for quick testing")
    ap.add_argument("--force-lsa", action="store_true",
                    help="Skip real model loading and use the TF-IDF+LSA fallback directly. "
                         "Useful for fast CPU-only runs when the real model is in the HF cache "
                         "but too slow to run at scale.")
    args = ap.parse_args()

    t_start = time.time()
    print("[rank] loading config ...")
    cfg = load_config(args.config)

    # Load runtime skill synonym overrides from config (Item 2)
    load_overrides(cfg.get("skill_overrides"))

    print(f"[rank] loading candidates from {args.candidates} ...")
    candidates = []
    for i, c in enumerate(iter_candidates(args.candidates)):
        if args.limit is not None and i >= args.limit:
            break
        candidates.append(c)
    n = len(candidates)
    ids = [cid(c) for c in candidates]
    print(f"[rank] {n} candidates loaded")

    career_path = os.path.join(args.artifacts_dir, "career_embeddings.npz")
    skill_path = os.path.join(args.artifacts_dir, "skill_embeddings.npz")
    features_path = os.path.join(args.artifacts_dir, "features.json")

    # Item 5: load embedding backend.
    # --force-lsa: skip real model entirely (e.g. HF cache too slow on CPU).
    # Normal path: try local artifacts/models/<name>/ first, then HF Hub.
    if args.force_lsa:
        print("[rank] --force-lsa: skipping real model, loading TF-IDF+LSA directly ...")
        from engine.embedder import ensure_corpus_fitted
        corpus_for_fit = None
        if not os.path.exists(os.path.join(args.artifacts_dir, "lsa_model.pkl")):
            corpus_for_fit = [career_text(c) for c in candidates] + [skill_text(c) for c in candidates]
        ensure_corpus_fitted(corpus_for_fit, artifacts_dir=args.artifacts_dir)
    else:
        get_embedder(
            cfg["embedding"]["model_name"],
            cfg["embedding"]["dimension"],
            artifacts_dir=args.artifacts_dir,
        )

    # --- Embedding tier assertion ---
    _tier = active_tier()
    print(f"[rank] EMBEDDING TIER: {_tier}")
    if _tier != "real":
        print(f"[rank] WARNING: real model NOT active (tier={_tier!r}) — semantic recall is "
              f"degraded. Run: python scripts/download_model.py --artifacts-dir {args.artifacts_dir}")
    else:
        print(f"[rank] real {cfg['embedding']['model_name']} loaded — full semantic recall active.")

    if is_using_fallback() and not args.force_lsa:
        from engine.embedder import ensure_corpus_fitted
        corpus_for_fit = None
        if not os.path.exists(os.path.join(args.artifacts_dir, "lsa_model.pkl")):
            corpus_for_fit = [career_text(c) for c in candidates] + [skill_text(c) for c in candidates]
        ensure_corpus_fitted(corpus_for_fit, artifacts_dir=args.artifacts_dir)
        print("[rank] WARNING: offline embedding fallback active. "
              "Run scripts/download_model.py for zero-network Tier 1 setup.")

    if os.path.exists(career_path) and os.path.exists(skill_path):
        print("[rank] loading pre-computed embeddings ...")
        ids_career, career_embs = load_npz_embeddings(career_path)
        ids_skill, skill_embs = load_npz_embeddings(skill_path)
        assert list(ids_career)[:n] == ids, \
            "career embedding ids do not match candidate order; re-run precompute or drop --limit"
        career_embs = career_embs[:n]
        skill_embs = skill_embs[:n]
    else:
        print("[rank] no pre-computed embeddings found — embedding on the fly "
              "(slow at scale; use precompute/embed_candidates.py for full runs).")
        career_embs = embed_texts([career_text(c) for c in candidates])
        skill_embs = embed_texts([skill_text(c) for c in candidates])

    features = {}
    reference_date = date.today()
    if os.path.exists(features_path):
        print("[rank] loading pre-computed features ...")
        with open(features_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        features = raw.get("candidates", raw)  # backward-compatible
        ref_str = raw.get("reference_date")
        parsed_ref = parse_date(ref_str) if isinstance(raw, dict) else None
        if parsed_ref:
            reference_date = parsed_ref
    else:
        print("[rank] no pre-computed features found — computing on the fly, "
              "and using max(last_active_date) in this pool as reference_date.")
        max_date = None
        for c in candidates:
            d = parse_date(c.get("redrob_signals", {}).get("last_active_date"))
            if d and (max_date is None or d > max_date):
                max_date = d
        if max_date:
            reference_date = max_date
    print(f"[rank] using reference_date={reference_date} for recency scoring")

    # ---- Channel 1: Semantic ----
    print("[rank] Channel 1: semantic relevance ...")
    query_embs = embed_texts(cfg["semantic_queries"])

    # Item 6: query polarity weights + blend_alpha
    query_weights = build_query_weights(cfg)
    sd_cfg = cfg.get("stuffer_detection", {})
    blend_alpha = sd_cfg.get("semantic_blend_alpha", 0.5)

    semantic_scores, _ = compute_semantic_scores(
        career_embs, query_embs,
        query_weights=query_weights,
        blend_alpha=blend_alpha,
    )
    skill_emb_sims = cosine_sim_matrix(skill_embs, query_embs)
    skill_sim_mean = np.clip((skill_emb_sims.mean(axis=1) + 1.0) / 2.0, 0.0, 1.0)
    career_sim_mean = semantic_scores

    # ---- Channel 2: Skills ----
    print("[rank] Channel 2: skill match ...")

    # Item 2: load derived clusters if available, else fall back to config
    skill_clusters = load_derived_skill_clusters(args.artifacts_dir, cfg["skill_clusters"])

    skill_scores = np.zeros(n, dtype=np.float32)
    matched_skills_list = []
    skill_channel_details = []
    for i, c in enumerate(candidates):
        cred_table = features.get(cid(c), {}).get("skill_credibility")
        res = compute_skill_score(c, skill_clusters, credibility_table=cred_table)
        skill_scores[i] = res["skill_score"]
        matched_skills_list.append(res["matched_skills"])
        skill_channel_details.append(res)

    # Set of core skill names (lowercased) for the shallow_ai_recent_only disqualifier check
    core_skill_names_lower = {s.lower() for s in skill_clusters.get("core", [])}

    # ---- Channel 3: Career trajectory + disqualifiers ----
    print("[rank] Channel 3: career trajectory + disqualifiers ...")
    jd_title = cfg["job_title"]
    jd_title_emb = embed_texts([jd_title])
    title_embs_cache = embed_texts([current_title(c) for c in candidates])
    career_scores = np.zeros(n, dtype=np.float32)
    career_details = []
    for i, c in enumerate(candidates):
        res = compute_career_score(
            c, cfg["constraints"], jd_title,
            title_emb=title_embs_cache[i:i+1], jd_title_emb=jd_title_emb,
            core_skill_names_lower=core_skill_names_lower,
        )
        career_scores[i] = res["career_score"]
        career_details.append(res)

    # ---- Channel 4: Behavioral ----
    print("[rank] Channel 4: behavioral / availability ...")
    behavioral_multipliers = np.zeros(n, dtype=np.float32)
    behavioral_details = []
    for i, c in enumerate(candidates):
        pre = features.get(cid(c), {}).get("behavioral")
        res = pre if pre else compute_behavioral(c, cfg.get("behavioral"), reference_date=reference_date)
        behavioral_multipliers[i] = res["multiplier"]
        behavioral_details.append(res)

    # ---- Channel 5: Integrity ----
    # IMPORTANT: integrity is NEVER read from the features cache. It is fast
    # pure-Python (no embeddings, <1s for 100K) and its params can change via
    # config. Using a cached result was the exact bug that let CAND_0039754
    # slip through after its config tolerance was fixed: the old cache said
    # n_failed=1, the new code+config says n_failed=2, but rank.py was reading
    # the old value. Always recompute — it's not worth caching.
    print("[rank] Channel 5: integrity / anomaly detection (always recomputed) ...")
    integrity_multipliers = np.zeros(n, dtype=np.float32)
    integrity_details = []
    for i, c in enumerate(candidates):
        res = compute_integrity(
            c, cfg.get("integrity"),
            title_emb=title_embs_cache[i:i+1], skill_emb=skill_embs[i:i+1], career_emb=career_embs[i:i+1],
        )
        integrity_multipliers[i] = res["multiplier"]
        integrity_details.append(res)

    # ---- Stuffer detection (Item 6b: ramped penalty) ----
    print("[rank] Stuffer detection ...")
    ramp_width = sd_cfg.get("stuffer_ramp_width", 0.15)
    stuffer_penalty, gap = compute_stuffer_penalty(
        skill_sim_mean, career_sim_mean,
        sd_cfg.get("gap_threshold", 0.4),
        sd_cfg.get("penalty_multiplier", 0.3),
        ramp_width=ramp_width,
    )

    # ---- Fusion ----
    print("[rank] Fusion ...")
    raw_score, final_score = fuse(
        semantic_scores, skill_scores, career_scores,
        behavioral_multipliers, integrity_multipliers, stuffer_penalty,
        cfg["channel_weights"],
    )

    # ---- Item 4b: Compute percentile-based score floors ----
    score_floors = compute_score_floors(
        semantic_scores, skill_scores, career_scores, raw_score,
        top_n=500, percentile=25.0,
    )
    print(f"[rank] score floors (p25 of top-500): "
          f"semantic={score_floors['semantic']:.3f}, "
          f"skill={score_floors['skill']:.3f}, "
          f"career={score_floors['career']:.3f}")

    # Role descriptor for reasoning (Item 4a)
    role_descriptor = cfg.get("role_descriptor", None)

    # ---- Rank, applying the EXACT tie-break rule the validator checks ----
    # Round first, then sort by (rounded score descending, candidate_id
    # ascending). Sorting on the already-rounded value (rather than full
    # precision, then rounding afterward) is what guarantees the output is
    # non-increasing AND has candidate_id ascending within any rounded-score
    # tie group — no post-hoc clamping needed, and none of its edge cases.
    rounded_scores = [round(float(s), 6) for s in final_score]
    rounded_raw = [round(float(s), 6) for s in raw_score]
    top_n = cfg["output"]["top_n"]

    # STAGE 1: select the top_n pool by raw_score (pure fit: semantic + skills +
    # career) with explicit honeypot exclusion. Using raw_score here means
    # behavioral/integrity multipliers can NEVER evict a genuine tier-5 from
    # the pool — a strong-fit engineer with a long notice or low response rate
    # still makes the shortlist. Honeypots are removed via the direct
    # is_honeypot flag (n_failed >= 3), not by hoping their multiplier=0
    # pushes them out.
    honeypot_mask = [integrity_details[i].get("is_honeypot", False) for i in range(n)]
    raw_order = sorted(range(n), key=lambda i: (-rounded_raw[i], ids[i]))
    pool_idx = [i for i in raw_order if not honeypot_mask[i]][:top_n]

    # STAGE 2: Base ordering — sort the full pool by final_score descending.
    # final_score = raw_score * behavioral * integrity * stuffer_penalty, so
    # behavioral/availability acts as a natural tiebreaker across the mid/lower
    # list without evicting genuine tier-5s from the pool (Stage 1 guarantees that).
    pool_by_final = sorted(
        pool_idx,
        key=lambda i: (-rounded_scores[i], -rounded_raw[i], ids[i]),
    )

    # STAGE 2.5: Top-10 fit-first reorder.
    # Re-sort ONLY the top 10 positions by fit-dominant key so that a tier-5
    # with a 90-day notice is never displaced by a tier-4 with a 15-day notice
    # inside positions 1–10. raw_score (semantic+skills+career, no behavioral)
    # is primary; final_score is strictly a tiebreaker. The rest of the list
    # (positions 11–100) stays in final_score order — behavioral is fully active
    # as a signal for the mid/lower band where it correctly separates otherwise-
    # equivalent candidates.
    TOP_REORDER_N = 10
    top_10_reordered = sorted(
        pool_by_final[:TOP_REORDER_N],
        key=lambda i: (-rounded_raw[i], -rounded_scores[i], ids[i]),
    )
    top_idx = top_10_reordered + pool_by_final[TOP_REORDER_N:]

    # STAGE 3: Build a GUARANTEED NON-INCREASING score sequence.
    # The reported `score` column must satisfy score[1] >= score[2] >= ... >= score[100]
    # (validator hard-rejects any violation). After the top-10 fit-first reorder,
    # neither raw_score nor final_score is guaranteed monotone across the full list.
    # Solution: walk the final rank order and assign score[rank] = min(final_score
    # of this candidate, score[rank-1]) — a running minimum of final_scores.
    # This is spec-compliant and safe: scores don't affect NDCG, only order does.
    reported_scores = []
    running_min = float(final_score[top_idx[0]])
    for idx in top_idx:
        s = float(final_score[idx])
        running_min = min(running_min, s)
        reported_scores.append(round(running_min, 6))

    # STAGE 3b: Tie-break fix — validator requires ascending candidate_id within
    # any equal-score group. The running-minimum can create ties (e.g. two candidates
    # both clamped to the same floor value). Post-process: within each consecutive
    # equal-score segment, re-sort top_idx by ascending candidate_id.
    # This preserves non-increasing scores AND satisfies the tie-break rule.
    top_idx = list(top_idx)  # ensure mutable
    i = 0
    while i < len(top_idx):
        j = i + 1
        while j < len(top_idx) and reported_scores[j] == reported_scores[i]:
            j += 1
        if j > i + 1:  # tied segment of length > 1
            top_idx[i:j] = sorted(top_idx[i:j], key=lambda k: ids[k])
        i = j

    print(f"[rank] generating reasoning for top {top_n} ...")
    rows = []
    for rank_pos, (idx, score) in enumerate(zip(top_idx, reported_scores), start=1):
        c = candidates[idx]

        channel_results = {
            "semantic_score": float(semantic_scores[idx]),
            "skill_score": float(skill_scores[idx]),
            "career_score": float(career_scores[idx]),
            "matched_skills": matched_skills_list[idx],
            "behavioral": behavioral_details[idx],
            "integrity": integrity_details[idx],
            "disqualifier_flags": career_details[idx].get("disqualifier_flags", {}),
            # Pass experience_fit so reasoning can name out-of-band experience
            "experience_fit": career_details[idx].get("subscores", {}).get("experience_fit", 1.0),
            "experience_yoe": float(raw_score[idx]),  # placeholder; yoe read in reasoning from candidate
        }
        reasoning = generate_reasoning(
            c, channel_results,
            rank=rank_pos,
            role_descriptor=role_descriptor,
            score_floors=score_floors,
        )

        rows.append({
            "candidate_id": cid(c),
            "rank": rank_pos,
            "score": score,
            "reasoning": reasoning,
        })

    # ---- Regression guard: hard fail before writing if any honeypot found ----
    honeypots_in_top = sum(1 for idx in top_idx if integrity_details[idx].get("is_honeypot"))
    if honeypots_in_top > 0:
        honeypot_ids = [ids[idx] for idx in top_idx if integrity_details[idx].get("is_honeypot")]
        print(f"[rank] FATAL: {honeypots_in_top} honeypot(s) in top {top_n}: {honeypot_ids}")
        print(f"[rank] Submission NOT written. Fix integrity config and rerun.")
        sys.exit(1)

    print(f"[rank] writing {args.out} ...")
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["candidate_id", "rank", "score", "reasoning"])
        writer.writeheader()
        writer.writerows(rows)

    dt = time.time() - t_start
    print(f"[rank] DONE in {dt:.1f}s. Wrote top {len(rows)} candidates to {args.out}")
    print(f"[rank] honeypots in top {top_n}: 0 (OK)")



if __name__ == "__main__":
    main()
