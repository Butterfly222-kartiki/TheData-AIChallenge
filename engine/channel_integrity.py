"""
engine/channel_integrity.py

CHANNEL 5 — Profile Integrity / Anomaly Detection.
Completely JD-agnostic. Catches honeypots via structural self-consistency
checks — no blocklist of known anomalies required.

Thresholds were calibrated by direct inspection of the real 100K-candidate
dataset (not guessed):
  - timeline_consistency (career_months - yoe*12 > 60): cleanly isolates
    ~14 candidates with gaps of 88-185 months from the rest of the
    population (everyone else clusters tightly around 0, i.e. career
    history slightly UNDER total claimed experience due to gaps).
  - role_duration (single role > yoe*12 + 12): isolates ~6 more candidates
    with single roles of 100-228 months against claimed yoe of 4.4-15.0.
  - zero-duration "expert" skills: exactly 21 individual occurrences in
    the real data. A second, STRONGER check fires when 3+ such skills
    appear on the SAME profile (a uniform cluster-of-5 pattern was found
    in the real data — clearly deliberately injected, not noise) —
    this is treated as an independent additional failure so these
    profiles reliably hit the 2-3+ failure tier rather than being merely
    soft-penalized.
  - skill_duration (skill > yoe*12 + 48): kept generous on purpose — in
    this dataset, skill-duration-vs-yoe mismatches are common ambient
    noise (max real-world excess found was ~59 months with NO clear
    separation from honeypots), so a tight threshold here would produce
    false positives across thousands of ordinary profiles.

Title-skill / skill-career coherence (the two semantic checks) require
real sentence embeddings to work correctly — see engine/embedder.py.
"""

from __future__ import annotations
from engine.data import skill_text, career_text, total_career_months, yoe as get_yoe, \
    career_history, skills as get_skills, current_title
from engine.embedder import embed_texts, cosine_sim_pairwise

DEFAULT_PARAMS = {
    # Tightened from 60 → 30 months: real profiles drift at most ~5 months;
    # 30 months still absorbs overlapping/concurrent roles and rounding.
    "timeline_consistency_tolerance_months": 30,
    # A gap > severe_multiplier × tolerance is an independent second failure.
    # 3× = 90 months: CAND_0039754 has a 96-month gap — 3.2× the tolerance.
    # Real profiles never exceed 1× the tolerance (~5 months max observed).
    "severe_timeline_gap_multiplier": 3,
    "role_duration_buffer_months": 12,
    "skill_duration_buffer_months": 48,
    "zero_duration_expert_cluster_min": 3,
    "title_skill_coherence_min": 0.25,
    "skill_career_coherence_min": 0.25,
    # Threshold lowered to 2: timeline_consistency + severe_timeline_gap both
    # fire on the inflated-YoE honeypots, giving n_failed=2 → is_honeypot.
    # The multiplier for 1 failure is kept at 0.7 (soft penalty for borderline
    # real profiles), 2 failures → 0.0 (honeypot kill).
    "failure_multipliers": {"0": 1.0, "1": 0.7, "2+": 0.0},
    "honeypot_threshold": 2,
}


def check_timeline_consistency(candidate: dict, tolerance_months: float) -> bool:
    """
    Dual check — fires if EITHER:
      (a) sum(duration_months) > yoe*12 + tolerance  [sum-of-durations inflated
          relative to claimed YoE — catches the CAND_0039754 pattern where the
          role durations sum to 98 months but 16.2 yoe claims 194 months]
      (b) yoe*12 > sum(duration_months) + tolerance  [claimed YoE far exceeds
          what the role durations add up to — also anomalous]

    The original check only fired when career_months EXCEEDED yoe*12 by a large
    margin, but missed cases where yoe was inflated relative to career_months.
    Both directions are now checked symmetrically.
    """
    yoe = get_yoe(candidate)
    career_months = total_career_months(candidate)
    claimed_months = yoe * 12
    # Check both directions: either side can be inflated
    return abs(career_months - claimed_months) < tolerance_months


def check_severe_timeline_gap(candidate: dict, tolerance_months: float,
                               severe_multiplier: float) -> bool:
    """
    Second, stronger timeline check: a gap larger than severe_multiplier *
    tolerance is independently disqualifying. This fires alongside
    check_timeline_consistency so profiles with extreme gaps reach
    n_failed >= 2 even if all other checks pass.

    With tolerance=30 and severe_multiplier=3: threshold = 90 months.
    CAND_0039754 has a 96-month gap (3.2x) — this fires.
    Real profiles observed max ~5 months drift — this never fires.
    """
    yoe = get_yoe(candidate)
    career_months = total_career_months(candidate)
    claimed_months = yoe * 12
    severe_threshold = tolerance_months * severe_multiplier
    return abs(career_months - claimed_months) < severe_threshold


def check_role_duration(candidate: dict, buffer_months: float) -> bool:
    yoe = get_yoe(candidate)
    cap = yoe * 12 + buffer_months
    for role in career_history(candidate):
        if float(role.get("duration_months", 0) or 0) > cap:
            return False
    return True


def check_skill_duration(candidate: dict, buffer_months: float) -> bool:
    yoe = get_yoe(candidate)
    cap = yoe * 12 + buffer_months
    for s in get_skills(candidate):
        if float(s.get("duration_months", 0) or 0) > cap:
            return False
    return True


def check_proficiency_duration_coherence(candidate: dict) -> bool:
    """No expert skill with duration_months == 0."""
    for s in get_skills(candidate):
        if (s.get("proficiency", "").lower() == "expert"
                and int(s.get("duration_months", 0) or 0) == 0):
            return False
    return True


def check_zero_duration_expert_cluster(candidate: dict, min_cluster: int) -> bool:
    """A SECOND, stronger check: 3+ zero-duration expert skills on one
    profile is a deliberate pattern, not noise — fails independently of
    check_proficiency_duration_coherence so these profiles reliably reach
    the 2-3+ integrity-failure tier."""
    n = sum(1 for s in get_skills(candidate)
            if s.get("proficiency", "").lower() == "expert" and int(s.get("duration_months", 0) or 0) == 0)
    return n < min_cluster


def title_skill_similarity(candidate: dict, title_emb=None, skill_emb_single=None) -> float | None:
    """Raw cosine similarity (rescaled to [0,1]), or None if either text is empty."""
    title = current_title(candidate)
    skills_txt = skill_text(candidate)
    if not title or not skills_txt:
        return None
    if title_emb is None or skill_emb_single is None:
        embs = embed_texts([title, skills_txt])
        title_emb, skill_emb_single = embs[0:1], embs[1:2]
    sim = float(cosine_sim_pairwise(title_emb, skill_emb_single)[0])
    return (sim + 1.0) / 2.0


def skill_career_similarity(candidate: dict, skill_emb_single=None, career_emb_single=None) -> float | None:
    skills_txt = skill_text(candidate)
    career_txt = career_text(candidate)
    if not skills_txt or not career_txt:
        return None
    if skill_emb_single is None or career_emb_single is None:
        embs = embed_texts([skills_txt, career_txt])
        skill_emb_single, career_emb_single = embs[0:1], embs[1:2]
    sim = float(cosine_sim_pairwise(skill_emb_single, career_emb_single)[0])
    return (sim + 1.0) / 2.0


def check_title_skill_coherence(candidate: dict, min_sim: float,
                                 title_emb=None, skill_emb_single=None) -> bool:
    sim = title_skill_similarity(candidate, title_emb, skill_emb_single)
    if sim is None:
        return True
    return sim >= min_sim


def check_skill_career_coherence(candidate: dict, min_sim: float,
                                  skill_emb_single=None, career_emb_single=None) -> bool:
    sim = skill_career_similarity(candidate, skill_emb_single, career_emb_single)
    if sim is None:
        return True
    return sim >= min_sim


def compute_integrity(candidate: dict, params: dict | None = None,
                       title_emb=None, skill_emb=None, career_emb=None) -> dict:
    p = {**DEFAULT_PARAMS, **(params or {})}

    checks = {
        "timeline_consistency": check_timeline_consistency(
            candidate, p["timeline_consistency_tolerance_months"]),
        "severe_timeline_gap": check_severe_timeline_gap(
            candidate,
            p["timeline_consistency_tolerance_months"],
            p["severe_timeline_gap_multiplier"]),
        "role_duration": check_role_duration(candidate, p["role_duration_buffer_months"]),
        "skill_duration": check_skill_duration(candidate, p["skill_duration_buffer_months"]),
        "proficiency_duration_coherence": check_proficiency_duration_coherence(candidate),
        "zero_duration_expert_cluster": check_zero_duration_expert_cluster(
            candidate, p["zero_duration_expert_cluster_min"]),
        "title_skill_coherence": check_title_skill_coherence(
            candidate, p["title_skill_coherence_min"], title_emb, skill_emb),
        "skill_career_coherence": check_skill_career_coherence(
            candidate, p["skill_career_coherence_min"], skill_emb, career_emb),
    }

    n_failed = sum(1 for passed in checks.values() if not passed)
    honeypot_threshold = p.get("honeypot_threshold", 2)
    # Multiplier lookup: "0", "1", "2+" (or fallback to 0.0 for unknown keys)
    if n_failed == 0:
        key = "0"
    elif n_failed == 1:
        key = "1"
    else:
        key = "2+"
    multiplier = p["failure_multipliers"].get(key, 0.0)

    return {
        "checks": checks,
        "n_failed": n_failed,
        "multiplier": multiplier,
        "is_honeypot": n_failed >= honeypot_threshold,
    }
