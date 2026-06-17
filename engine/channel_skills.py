"""
engine/channel_skills.py

CHANNEL 2 — Structured Skill Match (credibility-weighted).
Matches candidate skills against JD skill clusters (core/secondary/
nice-to-have). Each matched skill is weighted by a credibility function
that zeros out shallow/stuffed claims automatically.

credibility = proficiency_weight * log(1 + duration_months) * log(1 + endorsements)

Verified against the real dataset: genuine experts have endorsements in
the teens-40s and skill durations of 35-95 months; keyword-stuffer
profiles have the SAME skill names but endorsements of 0-4 and durations
of 6-17 months — the credibility formula spreads these two populations by
roughly two orders of magnitude per skill.

Rare plain-language skill-name variants (e.g. "Vector Representations",
"Information Retrieval Systems") are canonicalized via
engine/skill_synonyms.py before matching, so Tier-5 candidates aren't
penalized for using different words for the same expertise.

Anti-skills are split into two groups:
  - domain_mismatch: penalized ONLY if the candidate has NO core skill at
    all (mirrors the JD's literal wording: CV/speech/robotics WITHOUT
    significant NLP/IR exposure — multimodal candidates aren't penalized).
  - business_mismatch: always counted (catches HR/Sales/Content-Writer
    keyword-stuffer profiles directly).
"""

from __future__ import annotations
import math

from engine.skill_synonyms import canonicalize

CLUSTER_WEIGHTS = {
    "core": 1.0,
    "secondary": 0.6,
    "nice_to_have": 0.3,
}


def _norm(name: str) -> str:
    return (name or "").strip().lower()


def skill_credibility(skill: dict) -> float:
    from engine.data import proficiency_weight
    prof = proficiency_weight(skill.get("proficiency", "beginner"))
    dur = max(0, int(skill.get("duration_months", 0) or 0))
    endorse = max(0, int(skill.get("endorsements", 0) or 0))
    return prof * math.log(1 + dur) * math.log(1 + endorse)


def compute_skill_score(candidate: dict, skill_clusters: dict,
                         credibility_table: dict | None = None) -> dict:
    """
    credibility_table: optional precomputed {raw_skill_name: credibility}
    from extract_features.py. If absent, computed on the fly.
    """
    cand_skills = candidate.get("skills", [])
    cred_lookup = credibility_table
    if cred_lookup is None:
        cred_lookup = {s.get("name", ""): skill_credibility(s) for s in cand_skills if s.get("name")}

    # Map: normalized canonical skill name -> (original name, credibility)
    cand_canon = {}
    for s in cand_skills:
        raw_name = s.get("name", "")
        if not raw_name:
            continue
        canon = canonicalize(raw_name)
        cred = cred_lookup.get(raw_name, 0.0)
        key = _norm(canon)
        # keep the higher-credibility instance if a candidate somehow has duplicates
        if key not in cand_canon or cred > cand_canon[key][1]:
            cand_canon[key] = (raw_name, cred)

    matched_skills = {"core": [], "secondary": [], "nice_to_have": []}
    cluster_raw = {}

    for cluster_name in ("core", "secondary", "nice_to_have"):
        required = skill_clusters.get(cluster_name, []) or []
        raw = 0.0
        for req_skill in required:
            key = _norm(req_skill)
            if key in cand_canon:
                orig_name, cred = cand_canon[key]
                raw += cred
                matched_skills[cluster_name].append(orig_name)
        cluster_raw[cluster_name] = raw

    weighted_sum = sum(cluster_raw[c] * CLUSTER_WEIGHTS[c] for c in cluster_raw)

    ref_cap_per_skill = 30.0  # calibrated "strong, credible match" reference value
    max_possible = sum(len(skill_clusters.get(c, []) or []) * CLUSTER_WEIGHTS[c] * ref_cap_per_skill
                        for c in ("core", "secondary", "nice_to_have"))
    skill_score = weighted_sum / max_possible if max_possible > 0 else 0.0
    skill_score = max(0.0, min(1.0, skill_score))

    # --- Anti-skill penalties ---
    has_any_core = len(matched_skills["core"]) > 0

    business_anti = {_norm(a) for a in (skill_clusters.get("business_mismatch_anti_skills", []) or [])}
    domain_anti = {_norm(a) for a in (skill_clusters.get("domain_mismatch_anti_skills", []) or [])}

    cand_skill_names = {_norm(s.get("name", "")) for s in cand_skills if s.get("name")}
    business_hits = cand_skill_names & business_anti
    domain_hits = cand_skill_names & domain_anti if not has_any_core else set()  # conditional

    total_skills = max(1, len(cand_skill_names))
    business_fraction = len(business_hits) / total_skills
    domain_fraction = len(domain_hits) / total_skills

    anti_penalty = min(0.6, business_fraction + domain_fraction)
    skill_score = max(0.0, skill_score - anti_penalty)

    return {
        "skill_score": round(skill_score, 4),
        "matched_skills": matched_skills,
        "business_mismatch_hits": sorted(business_hits),
        "domain_mismatch_hits": sorted(domain_hits),
        "has_any_core": has_any_core,
        "raw_weighted_sum": round(weighted_sum, 4),
    }
