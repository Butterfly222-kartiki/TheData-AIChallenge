"""
engine/channel_career.py

CHANNEL 3 — Career Trajectory Coherence.
Checks whether the candidate's overall career trajectory aligns with the
JD's expectations: title fit, experience fit, industry fit, company-size
fit, career-wide consulting-only penalty, location fit, notice-period fit
— PLUS a set of explicit disqualifier checks.

ITEM 3 — Per-disqualifier `enabled` flags:
  Every disqualifier check is gated on an `enabled` key in the config.
  If `enabled: false` (or absent for backward compat), the check is
  skipped entirely (no flag, no penalty). The JD compiler sets these
  flags automatically based on semantic similarity to each disqualifier's
  anchor phrase. For the original JD they all default to true.

  Disqualifiers:
    title_chase, tech_lead_drift, pure_research_no_production,
    shallow_ai_recent_only, pure_consulting_career, closed_source_no_validation

ITEM 7 — Constraint scorers degrade to neutral when absent:
  Every sub-scorer returns 1.0 (neutral) when the relevant config block
  is absent or empty. This allows the JD compiler to omit constraints the
  JD doesn't mention, and nothing silently penalizes candidates.

  Specific rules:
    location_fit_score   → 1.0 when preferred_cities, welcome_cities, and
                           required_country are all absent/empty
    notice_fit_score     → 1.0 when ideal_max_days is absent/None
    company_size_fit_score → 1.0 when preferred_sizes is empty/absent
    industry_fit_score   → 1.0 when preferred_industries is empty/absent
"""

from __future__ import annotations
import math
from engine.embedder import embed_texts, cosine_sim_pairwise
from engine.data import current_title, current_industry, current_company_size, \
    location_city, location_country, career_history, yoe as get_yoe, signals


SUBSCORE_WEIGHTS = {
    "title_fit": 0.18,
    "experience_fit": 0.18,
    "industry_fit": 0.10,
    "company_size_fit": 0.07,
    "location_fit": 0.12,
    "notice_fit": 0.08,
    "validation_fit": 0.05,
    # Tier-5 discriminators: additive bonuses that separate production-shipping
    # candidates from keyword-matched ones. These are the exact signals the JD
    # calls out as must-haves that tier-4s typically lack.
    "production_signal": 0.12,   # shipped/deployed/production/scale/real-users in career text
    "ranking_eval": 0.07,        # NDCG/MRR/A-B/MAP/offline-online evaluation in career text
    "india_location": 0.03,      # India-based: JD strongly prefers, most tier-5s are India-located
}
# weights sum to 1.0


def title_fit_score(candidate_title: str, jd_title: str, title_emb=None, jd_title_emb=None) -> float:
    if not candidate_title or not jd_title:
        return 0.3
    if title_emb is None or jd_title_emb is None:
        embs = embed_texts([candidate_title, jd_title])
        title_emb, jd_title_emb = embs[0:1], embs[1:2]
    sim = cosine_sim_pairwise(title_emb, jd_title_emb)[0]
    return float(max(0.0, min(1.0, (sim + 1.0) / 2.0)))


def experience_fit_score(yoe: float, ideal_center: float, ideal_min: float, ideal_max: float,
                          hard_min: float, hard_max: float) -> float:
    sigma = max(1.0, (ideal_max - ideal_min) / 2.0)
    base = math.exp(-((yoe - ideal_center) ** 2) / (2 * sigma ** 2))
    if yoe < hard_min or yoe > hard_max:
        return float(base * 0.3)
    return float(base)


def industry_fit_score(industry: str, preferred_industries: list[str]) -> float:
    """
    Returns 1.0 (neutral) when preferred_industries is empty — the JD
    didn't express a preference, so no penalty for any industry.
    """
    if not preferred_industries:
        return 1.0  # no preference specified → neutral
    if not industry:
        return 0.5
    return 1.0 if industry in preferred_industries else 0.4


def company_size_fit_score(size: str, preferred_sizes: list[str]) -> float:
    """
    Returns 1.0 (neutral) when preferred_sizes is empty — the JD didn't
    express a company size preference.
    """
    if not preferred_sizes:
        return 1.0  # no preference specified → neutral
    if not size:
        return 0.5
    return 1.0 if size in preferred_sizes else 0.5


def location_fit_score(
    city: str,
    country: str,
    preferred_cities: list[str],
    welcome_cities: list[str],
    required_country: str,
    willing_to_relocate: bool,
    relocation_credit: float,
) -> float:
    """
    Returns 1.0 (neutral) when no location constraint is specified in the
    config (preferred_cities, welcome_cities, and required_country all
    absent/empty). This handles remote-friendly JDs gracefully — removing
    a location block from the config removes the penalty entirely.
    """
    # Neutral when no location constraint is configured
    has_city_pref = bool(preferred_cities or welcome_cities)
    has_country_req = bool(required_country)
    if not has_city_pref and not has_country_req:
        return 1.0

    if preferred_cities and city in preferred_cities:
        return 1.0
    if welcome_cities and city in welcome_cities:
        return 0.85
    if has_country_req and country == required_country:
        return 0.6
    if not has_country_req:
        # No country requirement but city didn't match → small penalty
        return 0.7
    if willing_to_relocate:
        return relocation_credit
    return 0.1


def notice_fit_score(notice_period_days: int, ideal_max_days: float | None) -> float:
    """
    Returns 1.0 (neutral) when ideal_max_days is None/absent — the JD
    didn't mention a notice period preference.
    """
    if ideal_max_days is None:
        return 1.0  # no notice constraint → neutral
    try:
        return float(1.0 / (1.0 + math.exp((notice_period_days - ideal_max_days) / 15.0)))
    except OverflowError:
        return 0.0


def production_signal_bonus(candidate: dict) -> float:
    """
    Additive bonus for candidates whose career text contains explicit production-
    shipping language — the JD's primary differentiator for tier-5 vs tier-4.
    Keywords drawn directly from the JD's 'how to read between the lines' section.
    Returns 0.0–1.0 scaled by evidence density.
    """
    career = career_history(candidate)
    all_text = " ".join(
        ((r.get("description", "") or "") + " " + (r.get("title", "") or "")).lower()
        for r in career
    )
    strong_signals = [
        "production", "shipped", "deployed", "real users", "at scale",
        "serving", "live", "launched", "million", "billion", "latency",
        "throughput", "inference", "model serving", "online",
    ]
    hits = sum(1 for kw in strong_signals if kw in all_text)
    # Sigmoid-style: 3+ hits → ~1.0, 1 hit → ~0.4, 0 hits → 0.0
    if hits == 0:
        return 0.0
    return min(1.0, 0.25 + hits * 0.15)


def ranking_eval_bonus(candidate: dict) -> float:
    """
    Additive bonus for candidates who demonstrate ranking/retrieval evaluation
    literacy — NDCG, MRR, A/B testing, MAP, precision@k. These appear in tier-5
    profiles because they've actually built and measured ranking systems; tier-4s
    typically lack this depth.
    """
    career = career_history(candidate)
    all_text = " ".join(
        ((r.get("description", "") or "") + " " + (r.get("title", "") or "")).lower()
        for r in career
    )
    eval_signals = [
        "ndcg", "mrr", "map", "a/b", "a-b test", "ab test", "offline",
        "precision@", "recall@", "mean reciprocal", "normalized discounted",
        "evaluation framework", "ranking metric", "retrieval metric",
        "click-through", "ctr", "engagement", "conversion",
    ]
    hits = sum(1 for kw in eval_signals if kw in all_text)
    if hits == 0:
        return 0.0
    return min(1.0, 0.3 + hits * 0.2)


def india_location_bonus(city: str, country: str) -> float:
    """
    Small additive boost for India-based candidates. The JD requires India
    location (no visa sponsorship, Pune/Noida preferred) so India-based
    candidates are strictly preferred over overseas ones.
    """
    if country == "India":
        return 1.0
    if country in ("IN", "India, "):
        return 1.0
    return 0.0


# --------------------------- Disqualifier checks ---------------------------

def check_title_chase(career: list[dict], max_avg_tenure: float, min_roles: int,
                       seniority_words: list[str]) -> bool:
    """True if flagged as a title-chaser."""
    if len(career) < min_roles:
        return False
    seniority_hits = sum(
        1 for r in career if any(w in (r.get("title", "") or "").lower() for w in seniority_words)
    )
    if seniority_hits < 2:
        return False
    avg_tenure = sum(r.get("duration_months", 0) or 0 for r in career) / len(career)
    return avg_tenure < max_avg_tenure


def check_tech_lead_drift(title: str, current_role_months: int, drift_words: list[str],
                           min_months: float) -> bool:
    title_l = (title or "").lower()
    if not any(w in title_l for w in drift_words):
        return False
    return current_role_months >= min_months


def check_pure_research_no_production(career: list[dict], current_ind: str,
                                       research_industries: list[str], production_keywords: list[str]) -> bool:
    industries = {r.get("industry", "") for r in career if r.get("industry")}
    industries.add(current_ind)
    industries.discard("")
    if not industries or not industries.issubset(set(research_industries)):
        return False
    text = " ".join((r.get("description", "") or "").lower() for r in career)
    return not any(kw in text for kw in production_keywords)


def check_pure_consulting_career(career: list[dict], current_ind: str, current_company: str,
                                  penalized_industries: list[str], consulting_firms: list[str]) -> bool:
    industries = {r.get("industry", "") for r in career if r.get("industry")}
    industries.add(current_ind)
    industries.discard("")
    companies = {r.get("company", "") for r in career if r.get("company")}
    companies.add(current_company)
    companies.discard("")
    if industries and industries.issubset(set(penalized_industries)):
        return True
    if companies and companies.issubset(set(consulting_firms)):
        return True
    return False


def check_shallow_ai_recent_only(candidate_skills: list[dict], core_skill_names_lower: set[str],
                                  max_duration: float, yoe_val: float, min_yoe: float) -> bool:
    if yoe_val < min_yoe:
        return False
    core_hits = [s for s in candidate_skills if (s.get("name", "") or "").strip().lower() in core_skill_names_lower]
    if not core_hits:
        return False
    return all((s.get("duration_months", 0) or 0) <= max_duration for s in core_hits)


def _is_enabled(dq_block: dict | None) -> bool:
    """
    Check whether a disqualifier block is enabled.
    Absent or non-dict → True (backward compatible: old configs without
    the `enabled` key keep all disqualifiers active).
    Dict with enabled=False → False.
    """
    if dq_block is None:
        return True
    if isinstance(dq_block, dict):
        return dq_block.get("enabled", True)
    return True


def compute_career_score(candidate: dict, constraints: dict, jd_title: str,
                          title_emb=None, jd_title_emb=None,
                          core_skill_names_lower: set[str] | None = None) -> dict:
    exp = constraints.get("experience", {}) or {}
    loc = constraints.get("location", {}) or {}
    comp = constraints.get("company", {}) or {}
    notice_cfg = constraints.get("notice_period", {}) or {}
    dq = constraints.get("disqualifiers", {}) or {}

    career = career_history(candidate)
    title = current_title(candidate)
    ind = current_industry(candidate)
    size = current_company_size(candidate)
    city = location_city(candidate)
    country = location_country(candidate)
    yoe_val = get_yoe(candidate)
    sig = signals(candidate)
    current_company = candidate.get("profile", {}).get("current_company", "")
    current_role_months = next((r.get("duration_months", 0) for r in career if r.get("is_current")), 0)

    title_fit = title_fit_score(title, jd_title, title_emb, jd_title_emb)

    # Experience fit — neutral defaults when block is absent
    if exp:
        experience_fit = experience_fit_score(
            yoe_val,
            exp.get("ideal_center", 7),
            exp.get("ideal_min", 5),
            exp.get("ideal_max", 9),
            exp.get("hard_min", 3),
            exp.get("hard_max", 14),
        )
    else:
        experience_fit = 1.0  # no experience constraint → neutral

    industry_fit = industry_fit_score(ind, comp.get("preferred_industries", []))
    company_size_fit = company_size_fit_score(size, comp.get("preferred_company_sizes", []))

    # Location fit — neutral when block absent (Item 7)
    location_fit = location_fit_score(
        city, country,
        loc.get("preferred_cities", []),
        loc.get("welcome_cities", []),
        loc.get("required_country", ""),
        bool(sig.get("willing_to_relocate", False)),
        loc.get("relocation_credit", 0.5),
    )

    # Notice fit — neutral when ideal_max_days absent (Item 7)
    notice_ideal_max = notice_cfg.get("ideal_max_days", None) if notice_cfg else None
    notice_fit = notice_fit_score(
        int(sig.get("notice_period_days", 60) or 60),
        notice_ideal_max,
    )

    # Restore validation_fit_score function logic inline or call it directly
    def calc_validation_fit(github_score, yoe):
        if github_score is None or github_score < 0:
            return 0.4
        return max(0.0, min(1.0, github_score / 100.0))
    validation_fit = calc_validation_fit(sig.get("github_activity_score"), yoe_val)

    prod_bonus = production_signal_bonus(candidate)
    rank_bonus = ranking_eval_bonus(candidate)
    india_bonus = india_location_bonus(city, country)

    subscores = {
        "title_fit": title_fit,
        "experience_fit": experience_fit,
        "industry_fit": industry_fit,
        "company_size_fit": company_size_fit,
        "location_fit": location_fit,
        "notice_fit": notice_fit,
        "validation_fit": validation_fit,
        "production_signal": prod_bonus,
        "ranking_eval": rank_bonus,
        "india_location": india_bonus,
    }
    weighted = sum(subscores[k] * SUBSCORE_WEIGHTS[k] for k in subscores)

    # --- Disqualifier checks (Item 3: gated on enabled flag) ---
    flags = {}
    penalty = 1.0

    tc = dq.get("title_chase", {}) or {}
    if _is_enabled(tc):
        flags["title_chase"] = check_title_chase(
            career,
            tc.get("max_avg_tenure_months", 18),
            tc.get("min_roles_to_flag", 3),
            tc.get("seniority_words", []),
        )
        if flags["title_chase"]:
            penalty *= 0.6
    else:
        flags["title_chase"] = False

    tld = dq.get("tech_lead_drift", {}) or {}
    if _is_enabled(tld):
        flags["tech_lead_drift"] = check_tech_lead_drift(
            title, current_role_months,
            tld.get("drift_title_words", []),
            tld.get("min_current_role_months", 18),
        )
        if flags["tech_lead_drift"]:
            penalty *= 0.7
    else:
        flags["tech_lead_drift"] = False

    prnp = dq.get("pure_research_no_production", {}) or {}
    if _is_enabled(prnp):
        flags["pure_research_no_production"] = check_pure_research_no_production(
            career, ind,
            prnp.get("research_industries", []),
            prnp.get("production_keywords", []),
        )
        if flags["pure_research_no_production"]:
            penalty *= 0.4  # JD: "we will not move forward" — heavy penalty
    else:
        flags["pure_research_no_production"] = False

    sar = dq.get("shallow_ai_recent_only", {}) or {}
    if _is_enabled(sar):
        flags["shallow_ai_recent_only"] = check_shallow_ai_recent_only(
            candidate.get("skills", []),
            core_skill_names_lower or set(),
            sar.get("max_core_skill_duration_months", 12),
            yoe_val,
            sar.get("min_yoe_to_flag", 2),
        )
        if flags["shallow_ai_recent_only"]:
            penalty *= 0.6
    else:
        flags["shallow_ai_recent_only"] = False

    # pure_consulting_career: special case — its "enabled" is a top-level key
    pc_enabled = dq.get("pure_consulting_career_enabled", True)
    # Also support enabled key inside a dict form for consistency
    pc_block = dq.get("pure_consulting_career", None)
    if isinstance(pc_block, dict):
        pc_enabled = pc_block.get("enabled", pc_enabled)
    pc_penalty_mult = dq.get("pure_consulting_career_penalty_multiplier", 0.5)
    if isinstance(pc_block, dict):
        pc_penalty_mult = pc_block.get("penalty_multiplier", pc_penalty_mult)

    if pc_enabled:
        flags["pure_consulting_career"] = check_pure_consulting_career(
            career, ind, current_company,
            comp.get("penalized_industries", []),
            comp.get("consulting_firm_names", []),
        )
        if flags["pure_consulting_career"]:
            penalty *= pc_penalty_mult
    else:
        flags["pure_consulting_career"] = False

    csnv = dq.get("closed_source_no_validation", {}) or {}
    if _is_enabled(csnv):
        github_score = sig.get("github_activity_score", -1)
        flags["closed_source_no_validation"] = (
            yoe_val >= csnv.get("min_yoe_to_flag", 5)
            and (github_score is None or github_score <= csnv.get("github_activity_threshold", 0))
        )
        if flags["closed_source_no_validation"]:
            penalty *= csnv.get("penalty_multiplier", 0.92)
    else:
        flags["closed_source_no_validation"] = False

    career_score = max(0.0, min(1.0, weighted * penalty))

    return {
        "career_score": round(career_score, 4),
        "subscores": {k: round(v, 4) for k, v in subscores.items()},
        "disqualifier_flags": flags,
        "disqualifier_penalty": round(penalty, 4),
    }
