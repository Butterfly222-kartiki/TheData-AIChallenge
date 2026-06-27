"""
evaluate.py — SELF-EVALUATION FRAMEWORK.

No leaderboard, no labels, no feedback during the competition. Run this
after every ranking to catch the four trap types AND basic format issues
before submitting. (For the authoritative format check, also run the
organizers' own validate_submission.py — this script's Check 0 mirrors
its rules but is not a replacement for it.)

Usage:
    python evaluate.py \
        --submission submission.csv \
        --candidates path/to/candidates.jsonl \
        --config config/jd_config.yaml
"""

from __future__ import annotations
import argparse
import csv
import os
import sys

import yaml

sys.path.insert(0, os.path.dirname(__file__))
from engine.data import iter_candidates, cid, career_text, skill_text, current_title, yoe as get_yoe, signals
from engine.embedder import get_embedder, embed_texts, cosine_sim_matrix
from engine.channel_integrity import compute_integrity


def load_submission(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def check_format(rows: list[dict]) -> list[str]:
    errs = []
    if len(rows) != 100:
        errs.append(f"Expected exactly 100 rows, found {len(rows)}")
    ranks = [int(r["rank"]) for r in rows]
    if sorted(ranks) != list(range(1, len(rows) + 1)):
        errs.append("Ranks are not exactly 1..N with no gaps/duplicates")
    by_rank = sorted(rows, key=lambda r: int(r["rank"]))
    for i in range(len(by_rank) - 1):
        s1, s2 = float(by_rank[i]["score"]), float(by_rank[i + 1]["score"])
        if s1 < s2:
            errs.append(f"Score not non-increasing at rank {by_rank[i]['rank']}->{by_rank[i+1]['rank']}")
        if s1 == s2 and by_rank[i]["candidate_id"] > by_rank[i + 1]["candidate_id"]:
            errs.append(f"Tie-break violated at ranks {by_rank[i]['rank']}/{by_rank[i+1]['rank']}: "
                        f"candidate_id must be ascending on equal scores")
    return errs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--submission", required=True)
    ap.add_argument("--candidates", required=True)
    ap.add_argument("--config", default="config/jd_config.yaml")
    ap.add_argument("--spot-check-n", type=int, default=5)
    ap.add_argument("--artifacts-dir", default="artifacts",
                     help="used to locate a pre-fit LSA model, if the real embedding model is unavailable")
    args = ap.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    print("[evaluate] loading candidate pool (for lookup; this may take a moment on the full 100K) ...")
    cand_by_id = {}
    for c in iter_candidates(args.candidates):
        cand_by_id[cid(c)] = c

    submission = load_submission(args.submission)
    print(f"[evaluate] loaded {len(submission)} submission rows")

    print("\n=== CHECK 0: Format & tie-break ===")
    fmt_errs = check_format(submission)
    if fmt_errs:
        for e in fmt_errs:
            print(f"  FAIL: {e}")
    else:
        print("  OK — 100 rows, ranks 1-100, scores non-increasing, ties broken by candidate_id ascending.")

    top10_ids = [r["candidate_id"] for r in submission if int(r["rank"]) <= 10]
    top100_ids = [r["candidate_id"] for r in submission]

    get_embedder(cfg["embedding"]["model_name"], cfg["embedding"]["dimension"])
    if is_using_fallback():
        from engine.embedder import ensure_corpus_fitted
        ensure_corpus_fitted(None, artifacts_dir=args.artifacts_dir)  # try to reuse the model fit during precompute
        if is_using_fallback():
            print("[evaluate] WARNING: no pre-fit LSA model found at the given --artifacts-dir; "
                  "fitting a fresh one from the candidates being evaluated only (smaller corpus than precompute used).")
            corpus = [career_text(c) for c in cand_by_id.values()] + [skill_text(c) for c in cand_by_id.values()]
            ensure_corpus_fitted(corpus, artifacts_dir=args.artifacts_dir)
    query_embs = embed_texts(cfg["semantic_queries"])

    print("\n=== CHECK 1: Coherence (top 10 vs JD semantic queries) ===")
    low_coherence = []
    for c_id in top10_ids:
        c = cand_by_id.get(c_id)
        if not c:
            print(f"  WARNING: candidate {c_id} in submission not found in candidate pool")
            continue
        emb = embed_texts([career_text(c)])
        sim = float(cosine_sim_matrix(emb, query_embs).mean())
        sim_rescaled = (sim + 1.0) / 2.0
        if sim_rescaled < 0.5:
            low_coherence.append((c_id, round(sim_rescaled, 3)))
    if low_coherence:
        print(f"  FLAGGED {len(low_coherence)} top-10 candidates below 0.5 semantic coherence:")
        for c_id, sim in low_coherence:
            print(f"    {c_id}: {sim}")
    else:
        print("  OK — all top-10 candidates score >= 0.5 semantic coherence.")

    print("\n=== CHECK 2: Stuffer leakage (top 100, skill-vs-career gap > 0.3) ===")
    flagged_stuffers = []
    for c_id in top100_ids:
        c = cand_by_id.get(c_id)
        if not c:
            continue
        career_emb = embed_texts([career_text(c)])
        skill_emb = embed_texts([skill_text(c)])
        career_sim = (float(cosine_sim_matrix(career_emb, query_embs).mean()) + 1.0) / 2.0
        skill_sim = (float(cosine_sim_matrix(skill_emb, query_embs).mean()) + 1.0) / 2.0
        gap = skill_sim - career_sim
        if gap > 0.3:
            flagged_stuffers.append((c_id, round(gap, 3)))
    if flagged_stuffers:
        print(f"  FLAGGED {len(flagged_stuffers)} likely stuffers leaking into top 100:")
        for c_id, g in flagged_stuffers[:10]:
            print(f"    {c_id}: gap={g}")
    else:
        print("  OK — no stuffer leakage detected in top 100.")

    print("\n=== CHECK 3: Integrity (top 100, candidates failing 2+ checks) ===")
    flagged_integrity = []
    for c_id in top100_ids:
        c = cand_by_id.get(c_id)
        if not c:
            continue
        res = compute_integrity(c, cfg.get("integrity"))
        if res["n_failed"] >= 2:
            flagged_integrity.append((c_id, res["n_failed"]))
    if flagged_integrity:
        print(f"  FLAGGED {len(flagged_integrity)} candidates failing 2+ integrity checks:")
        for c_id, nf in flagged_integrity[:10]:
            print(f"    {c_id}: {nf} checks failed")
        pct = len(flagged_integrity) / max(1, len(top100_ids)) * 100
        if pct > 10:
            print(f"  *** DISQUALIFICATION RISK: {pct:.1f}% of top 100 are likely honeypots (>10% threshold) ***")
    else:
        print("  OK — no candidates failing 2+ integrity checks in top 100.")

    print("\n=== CHECK 4: Behavioral (top 100, dead/unresponsive profiles) ===")
    flagged_behavioral = []
    for c_id in top100_ids:
        c = cand_by_id.get(c_id)
        if not c:
            continue
        sig = signals(c)
        resp = float(sig.get("recruiter_response_rate", 0.0) or 0.0)
        # A candidate who isn't actively flagged open-to-work but still
        # responds well isn't "dead" — only flag genuinely unresponsive
        # profiles (the actual signal the JD cares about for hireability).
        if resp < 0.15:
            flagged_behavioral.append((c_id, resp))
    if flagged_behavioral:
        print(f"  FLAGGED {len(flagged_behavioral)} candidates with a low recruiter response rate (<0.15):")
        for c_id, resp in flagged_behavioral[:10]:
            print(f"    {c_id}: response_rate={resp}")
    else:
        print("  OK — no genuinely unresponsive profiles detected in top 100.")

    print(f"\n=== CHECK 5: Manual spot check (top {args.spot_check_n} + bottom {args.spot_check_n}) ===")
    sorted_rows = sorted(submission, key=lambda r: int(r["rank"]))
    top_rows = sorted_rows[: args.spot_check_n]
    bottom_rows = sorted_rows[-args.spot_check_n:]

    def print_profile(row):
        c = cand_by_id.get(row["candidate_id"], {})
        profile = c.get("profile", {})
        print(f"  rank {row['rank']} | score {row['score']} | {row['candidate_id']}")
        print(f"    title: {profile.get('current_title')} | yoe: {profile.get('years_of_experience')} "
              f"| industry: {profile.get('current_industry')}")
        print(f"    reasoning: {row['reasoning']}")

    print("  --- TOP ---")
    for r in top_rows:
        print_profile(r)
    print("  --- BOTTOM ---")
    for r in bottom_rows:
        print_profile(r)

    print("\n[evaluate] Done. Review flags above before submitting.")


if __name__ == "__main__":
    main()
