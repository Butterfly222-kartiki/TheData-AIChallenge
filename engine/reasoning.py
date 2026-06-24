"""
engine/reasoning.py

LAYER 3 — Reasoning Generation.

Composes a 1-2 sentence justification from EXTRACTED FACTS rather than
quoted text snippets. This eliminates the Stage-4 variation failures:
  - No more verbatim career-description quoting (which shared templated
    text across multiple candidates in this dataset)
  - No more truncated mid-word snippets
  - No more "closely aligns with X" hardcoded closing clause on ~95 rows
  - Rank-band-appropriate tone: rows 80-100 read differently from rank 2

Architecture:
  1. Extract structured facts: yoe, title, company, which JD must-have
     their profile evidences (from matched_skills + career channel signals),
     and behavioral/availability signal.
  2. Select sentence template from a pool of 5 structures keyed by rank band.
  3. Append honest concern clause only when a real gap exists and the
     concern is specific (not a generic fallback).

Hard rule: never mention a skill the candidate doesn't have, a company
they didn't work at, or a title they don't hold.

ITEM 4a — Role descriptor from config:
  The phrase describing the role comes from config["role_descriptor"],
  passed in as role_descriptor. Defaults gracefully.

ITEM 4b — Percentile-based score floors:
  score_floors dict (semantic/skill/career) computed at ranking time from
  the 25th-percentile of the top-500 candidates. Falls back to safe
  lower defaults when absent.
"""

from __future__ import annotations
import hashlib
from engine.data import current_title, career_history, yoe as get_yoe, signals, skills as get_skills

# Safe default floors — lower than the old LSA-calibrated values so they
# don't misfire on real neural embeddings (which produce higher similarities)
# or on a new JD's query distribution.
_DEFAULT_SCORE_FLOORS = {
    "semantic": 0.55,
    "skill": 0.20,
    "career": 0.40,
}

_DISQUALIFIER_CONCERNS = {
    "title_chase": "a pattern of short tenures across escalating titles",
    "tech_lead_drift": "limited recent hands-on coding given their current leadership role",
    "pure_research_no_production": "a primarily research/academic background without clear production-shipping experience",
    "shallow_ai_recent_only": "relatively recent, shallow exposure to the core AI/IR skills listed",
    "pure_consulting_career": "an entirely consulting/IT-services career background",
    "closed_source_no_validation": "limited visible external validation (open source / GitHub)",
}

# ---------------------------------------------------------------------------
# Sentence-structure pool — 5 distinct frames, chosen by (rank_band, hash).
# Each frame uses only fact-slots that are always available; optional slots
# are guarded inside _build_lead().
# ---------------------------------------------------------------------------

# Rank-band boundaries
_BAND_TOP      = (1,  10)   # "exceptional / strong signal on X"
_BAND_STRONG   = (11, 30)   # "solid candidate / evidences X"
_BAND_MID      = (31, 60)   # "relevant background / covers X"
_BAND_LOWER    = (61, 100)  # "below the top tier / some alignment"


def _rank_band(rank: int) -> str:
    if rank <= 10:
        return "top"
    elif rank <= 30:
        return "strong"
    elif rank <= 60:
        return "mid"
    else:
        return "lower"


def _pick_structure(rank: int, candidate_id: str) -> int:
    """Pick one of N sentence structures, varying by rank + candidate_id hash."""
    h = int(hashlib.md5(f"{rank}:{candidate_id}".encode()).hexdigest()[:8], 16)
    return h % 12  # widened pool — actual modulus applied inside each band


def _top_matched_skills(matched_skills: dict, n: int = 3) -> list[str]:
    out = []
    for cluster in ("core", "secondary", "nice_to_have"):
        out.extend(matched_skills.get(cluster, []))
    return out[:n]


def _most_recent_role(candidate: dict) -> dict | None:
    career = career_history(candidate)
    if not career:
        return None
    current = next((r for r in career if r.get("is_current")), None)
    return current or career[0]


def _years_str(yoe: float) -> str:
    return f"{yoe:.1f}" if yoe % 1 else f"{int(yoe)}"


def _jd_signal_phrase(matched: list[str], career_score: float, semantic_score: float) -> str:
    """
    Identify which JD must-have the candidate most evidences, expressed
    as a fact phrase. Uses matched skills and score magnitudes.
    """
    if not matched:
        if semantic_score > 0.65:
            return "strong semantic alignment with the role's retrieval and ranking requirements"
        return "background spanning the AI/ML domain"

    top = matched[0].title()
    if len(matched) >= 2:
        second = matched[1].title()
        return f"hands-on depth in {top} and {second}"
    return f"hands-on depth in {top}"


def _build_lead(
    candidate: dict,
    matched: list[str],
    semantic_score: float,
    skill_score: float,
    career_score: float,
    rank: int,
    role_desc_phrase: str,
) -> str:
    """
    Build the lead sentence by composing from extracted facts.
    Multiple structural variants per band, selected by rank + candidate_id
    hash. Lower band has 12 variants to prevent repetition across ranks 61-100.
    """
    cid = candidate.get("candidate_id", "")
    title = current_title(candidate) or "AI/ML Engineer"
    yoe = get_yoe(candidate)
    yoe_s = _years_str(yoe)
    recent = _most_recent_role(candidate)
    company = (recent.get("company", "") if recent else "") or ""
    at_co = f" at {company}" if company else ""
    sig_phrase = _jd_signal_phrase(matched, career_score, semantic_score)
    band = _rank_band(rank)
    struct = _pick_structure(rank, cid)

    if band == "top":
        # 7 templates — struct % 7
        templates = [
            f"{title}{at_co} with {yoe_s} years of experience and {sig_phrase} — one of the strongest fits for {role_desc_phrase}.",
            f"Exceptionally well-aligned: {yoe_s} years as {title}{at_co}, with {sig_phrase} that directly satisfies the core requirements for {role_desc_phrase}.",
            f"Strong signal from {yoe_s} years in this domain: {title}{at_co} brings {sig_phrase}, a near-ideal match for {role_desc_phrase}.",
            f"Top-tier candidate — {title}{at_co}, {yoe_s} years, with {sig_phrase} squarely matching the role's most critical requirements.",
            f"Career trajectory as {title}{at_co} ({yoe_s} yrs) evidences {sig_phrase}, placing this profile among the best fits for {role_desc_phrase}.",
            f"Outstanding fit: {yoe_s}-year record as {title}{at_co} with demonstrated {sig_phrase} covering what matters most for {role_desc_phrase}.",
            f"High-confidence match — {title}{at_co} ({yoe_s} yrs) with {sig_phrase}; among the clearest signals in the candidate pool for {role_desc_phrase}.",
        ]
        return templates[struct % len(templates)]

    elif band == "strong":
        # 8 templates — struct % 8
        templates = [
            f"Strong candidate: {title}{at_co} with {yoe_s} years and {sig_phrase}, covering the key requirements for {role_desc_phrase}.",
            f"Well-evidenced fit — {yoe_s} years as {title}{at_co} with {sig_phrase}, solid alignment with {role_desc_phrase}.",
            f"{title}{at_co} ({yoe_s} yrs) demonstrates {sig_phrase}; a credible match for most dimensions of {role_desc_phrase}.",
            f"Solid background in {sig_phrase} spanning {yoe_s} years as {title}{at_co} — matches the core expectations for {role_desc_phrase}.",
            f"Relevant and credible: {yoe_s}-year career as {title}{at_co} with {sig_phrase} aligning to the primary requirements of {role_desc_phrase}.",
            f"Good overall fit: {title}{at_co} ({yoe_s} yrs) with {sig_phrase} that maps well to the JD's must-haves for {role_desc_phrase}.",
            f"Substantive experience as {title}{at_co} ({yoe_s} yrs) with {sig_phrase}; expected to perform well against the core criteria for {role_desc_phrase}.",
            f"Clear industry experience: {yoe_s} years as {title}{at_co} showing {sig_phrase} — a reliable profile for {role_desc_phrase}.",
        ]
        return templates[struct % len(templates)]

    elif band == "mid":
        # 10 templates — struct % 10
        templates = [
            f"Relevant background as {title}{at_co} ({yoe_s} yrs) with {sig_phrase}; covers some, but not all, of the requirements for {role_desc_phrase}.",
            f"{title}{at_co} with {yoe_s} years shows {sig_phrase}, offering partial alignment with {role_desc_phrase}.",
            f"Mid-tier fit — {yoe_s} years as {title}{at_co} evidences {sig_phrase}, though gaps remain relative to the top candidates for {role_desc_phrase}.",
            f"Some meaningful signal from {yoe_s} years as {title}{at_co}: {sig_phrase}, though depth on other must-haves is thinner.",
            f"Useful background ({yoe_s} yrs, {title}{at_co}) with {sig_phrase}; breadth is there but depth on critical requirements is uneven for {role_desc_phrase}.",
            f"Partially matching profile: {title}{at_co} ({yoe_s} yrs) brings {sig_phrase} but falls short on several secondary criteria for {role_desc_phrase}.",
            f"{yoe_s}-year career as {title}{at_co} surfaces {sig_phrase}; positions the candidate in the middle tier for {role_desc_phrase}.",
            f"Credible but incomplete coverage: {title}{at_co} ({yoe_s} yrs) with {sig_phrase}; secondary JD requirements are only partially met.",
            f"Moderate fit: {sig_phrase} from {yoe_s} years as {title}{at_co} addresses parts of the JD but leaves gaps in other priority areas for {role_desc_phrase}.",
            f"Reasonable candidate on paper: {yoe_s}-year track record as {title}{at_co} includes {sig_phrase}, though signal strength is mixed across all channels for {role_desc_phrase}.",
        ]
        return templates[struct % len(templates)]

    else:  # lower band (61-100)
        # 12 templates — struct % 12 — widest pool to avoid repetition across 40 candidates
        templates = [
            f"{title}{at_co} ({yoe_s} yrs) shows some alignment via {sig_phrase}, but does not clear the bar set by higher-ranked candidates for {role_desc_phrase}.",
            f"Peripheral fit: {yoe_s} years as {title}{at_co} with {sig_phrase}, though critical coverage gaps push this profile to the lower tier for {role_desc_phrase}.",
            f"Evidence of {sig_phrase} in a {yoe_s}-year career as {title}{at_co}, but the overall signal strength is insufficient to rank higher for {role_desc_phrase}.",
            f"Weaker match overall: {title}{at_co} ({yoe_s} yrs) demonstrates {sig_phrase}, yet the profile falls short on multiple must-have criteria for {role_desc_phrase}.",
            f"Included in the top 100 but outside the stronger cohort: {title}{at_co} ({yoe_s} yrs) with {sig_phrase}; depth on core requirements for {role_desc_phrase} is limited.",
            f"Adjacent skills only — {yoe_s}-year background as {title}{at_co} surfaces {sig_phrase} but lacks the depth expected for {role_desc_phrase}.",
            f"Lower-confidence match: {sig_phrase} is present across {yoe_s} years as {title}{at_co}, but cross-channel scores trail the stronger candidates for {role_desc_phrase}.",
            f"Thin evidence overall: {title}{at_co} ({yoe_s} yrs) with {sig_phrase}, though the JD's core and secondary requirements for {role_desc_phrase} are only partially addressed.",
            f"Borderline inclusion: {yoe_s} years as {title}{at_co} yields {sig_phrase}, but the match quality on the JD's primary signals for {role_desc_phrase} is below the median.",
            f"Partial overlap with the JD: {title}{at_co} ({yoe_s} yrs) brings {sig_phrase}, yet coverage of the full requirement set for {role_desc_phrase} is incomplete.",
            f"Ranked here for completeness: {yoe_s}-year career as {title}{at_co} shows {sig_phrase}, but the strength of signal does not justify a higher position for {role_desc_phrase}.",
            f"Outside the stronger cohort despite {sig_phrase}: {yoe_s} years as {title}{at_co} leaves notable gaps in the criteria most weighted for {role_desc_phrase}.",
        ]
        return templates[struct % len(templates)]


def generate_reasoning(
    candidate: dict,
    channel_results: dict,
    rank: int = 50,
    role_descriptor: str | None = None,
    score_floors: dict | None = None,
) -> str:
    """
    channel_results expected keys:
      semantic_score, skill_score, career_score, matched_skills,
      behavioral (dict), integrity (dict), disqualifier_flags (dict)

    rank: 1-based rank position. Used to set appropriate tone.

    role_descriptor: short phrase describing what the role needs.
      Pulled from config["role_descriptor"] at call site. Defaults to
      "this role's requirements" if not provided.

    score_floors: dict with keys semantic, skill, career — the 25th
      percentile of the top-500 candidates' score distribution for the
      current run. Self-calibrates to whichever model/JD is active.
      Falls back to _DEFAULT_SCORE_FLOORS when not provided.
    """
    semantic = channel_results.get("semantic_score", 0.0)
    skill = channel_results.get("skill_score", 0.0)
    career = channel_results.get("career_score", 0.0)
    matched_skills = channel_results.get("matched_skills", {})
    behavioral = channel_results.get("behavioral", {})
    integrity = channel_results.get("integrity", {})
    dq_flags = channel_results.get("disqualifier_flags", {})

    # Resolve role descriptor (Item 4a)
    role_desc_phrase = role_descriptor or "this role's requirements"

    # Resolve score floors (Item 4b)
    floors = _DEFAULT_SCORE_FLOORS.copy()
    if score_floors:
        floors.update(score_floors)

    matched = _top_matched_skills(matched_skills, n=3)

    # Build lead sentence from extracted facts
    lead = _build_lead(
        candidate, matched, semantic, skill, career, rank, role_desc_phrase
    )

    # --- Concern clause: specific, real gaps with explicit naming ---
    concerns = []
    sig = signals(candidate)
    notice = sig.get("notice_period_days")
    yoe = get_yoe(candidate)
    yoe_s = _years_str(yoe)

    # Experience band: surface when experience_fit is clearly depressed.
    # Pull the JD's ideal band from config constraints if available; fall back
    # to the score itself as a proxy. experience_fit < 0.5 reliably signals
    # the candidate is outside the JD's 5-9 year ideal window.
    exp_fit = channel_results.get("experience_fit", 1.0)
    if exp_fit < 0.50:
        # Determine direction: JD ideal is 5-9 years (hard coded for this JD;
        # generalised via the experience_fit signal direction).
        if yoe > 10:
            concerns.append(f"experience ({yoe_s} yrs) sits above the JD's ideal seniority band")
        elif yoe < 4:
            concerns.append(f"experience ({yoe_s} yrs) is below the JD's minimum seniority threshold")
        else:
            concerns.append(f"experience fit is below the ideal range for this role")

    # Notice period: name exact days for anything beyond 30
    if notice is not None and notice > 30:
        concerns.append(f"a {notice}-day notice period")

    # Responsiveness — flag only if notably low
    resp_rate = behavioral.get("recruiter_response_rate", 1.0)
    if resp_rate < 0.20:
        concerns.append(f"a low recruiter response rate ({resp_rate:.0%})")

    # Inactivity — only flag if truly stale
    days_inactive = behavioral.get("days_since_active", 0)
    if days_inactive > 180:
        concerns.append("extended inactivity on the platform")

    # Integrity flag — only major ones worth surfacing
    if integrity.get("n_failed", 0) >= 2:
        concerns.append("a profile-consistency anomaly worth verifying")

    # Disqualifier flags — specific names
    for flag_name, is_flagged in (dq_flags or {}).items():
        if is_flagged and flag_name in _DISQUALIFIER_CONCERNS:
            concerns.append(_DISQUALIFIER_CONCERNS[flag_name])

    # Show concerns:
    #   - Always for top-30 (reviewers scrutinise these most)
    #   - Always for lower band (61-100): reviewer expects an explanation of WHY
    #     this profile is ranked here, not just a generic "weak signal" label
    #   - For mid-band (31-60): only if behavioral is notably depressed or
    #     there's a hard disqualifier
    concern_clause = ""
    beh_mult = behavioral.get("multiplier", 1.0)
    band = _rank_band(rank)
    show_concern = concerns and (
        rank <= 30
        or band == "lower"
        or beh_mult < 0.85
        or integrity.get("n_failed", 0) >= 2
        or exp_fit < 0.50
        or (notice is not None and notice > 60)  # always surface long notice periods
    )
    if show_concern:
        concern_clause = f" Note: {concerns[0]}."

    return (lead + concern_clause).strip()
