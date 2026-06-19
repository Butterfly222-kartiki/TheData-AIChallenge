"""
engine/channel_behavioral.py

CHANNEL 4 — Behavioral / Availability Signals.
Completely JD-agnostic. Reads from candidate["redrob_signals"], the real
23-field Redrob signal block. Combined as a weighted mean, then rescaled
to [floor, ceiling] (default [0.3, 1.0]) to form a `behavioral_multiplier`
applied MULTIPLICATIVELY in fusion.

Recency is computed from `last_active_date` (a "YYYY-MM-DD" string)
against a `reference_date`. Since this system must be reproducible
regardless of when it's actually run, the reference date is NOT
wall-clock "today" by default — rank.py computes it once as
max(last_active_date across the whole candidate pool) and passes it in,
so behavioral scoring doesn't silently drift if graded weeks after the
dataset was generated.
"""

from __future__ import annotations
import math
from datetime import date
from engine.data import parse_date, days_since, signals


DEFAULT_PARAMS = {
    "recency_midpoint_days": 90,
    "recency_scale_days": 30,
    "availability_open_to_work_weight": 0.6,
    "availability_notice_weight": 0.4,
    "availability_notice_cap_days": 180,
    "market_validation_max_saves_30d": 80,
    # Floor raised from 0.3 → 0.75: with scores in a narrow band, a 0.3 floor
    # creates a 3.3× swing that reorders candidates by availability alone.
    # 0.75 means behavioral can nudge by at most 25%, keeping fit dominant.
    "rescale_floor": 0.75,
    "rescale_ceiling": 1.0,
    "weights": {
        "recency": 0.30,           # was 0.25 — active-looking signal is reliable
        "responsiveness": 0.10,    # was 0.25 — noisy; JD says 60d-notice Sr Eng still in scope
        "availability": 0.35,      # was 0.25 — open_to_work + notice period most predictive
        "market_validation": 0.10,
        "reliability": 0.10,
        "verification": 0.025,
        "profile_investment": 0.025,
    },
}


def _clip01(x: float) -> float:
    return max(0.0, min(1.0, x))


def recency_score(days_since_active: int, midpoint: float, scale: float) -> float:
    try:
        return 1.0 / (1.0 + math.exp((days_since_active - midpoint) / scale))
    except OverflowError:
        return 0.0


def availability_score(open_to_work: bool, notice_period_days: int,
                        notice_cap: float, w_open: float, w_notice: float) -> float:
    open_component = 1.0 if open_to_work else 0.0
    notice_component = _clip01(1.0 - (notice_period_days / notice_cap))
    return w_open * open_component + w_notice * notice_component


def market_validation_score(saved_by_recruiters_30d: int, max_saves: float) -> float:
    if max_saves <= 0:
        return 0.0
    return _clip01(saved_by_recruiters_30d / max_saves)


def verification_score(verified_email: bool, verified_phone: bool, linkedin_connected: bool) -> float:
    return (int(bool(verified_email)) + int(bool(verified_phone)) + int(bool(linkedin_connected))) / 3.0


def compute_behavioral(candidate: dict, params: dict | None = None,
                        reference_date: date | None = None) -> dict:
    p = {**DEFAULT_PARAMS, **(params or {})}
    w = {**DEFAULT_PARAMS["weights"], **(p.get("weights") or {})}
    sig = signals(candidate)

    if reference_date is None:
        reference_date = date.today()

    last_active = parse_date(sig.get("last_active_date"))
    days_since_active = days_since(last_active, reference_date)
    rec = recency_score(days_since_active, p["recency_midpoint_days"], p["recency_scale_days"])

    resp = _clip01(float(sig.get("recruiter_response_rate", 0.0) or 0.0))

    avail = availability_score(
        bool(sig.get("open_to_work_flag", False)),
        int(sig.get("notice_period_days", 180) or 180),
        p["availability_notice_cap_days"],
        p["availability_open_to_work_weight"],
        p["availability_notice_weight"],
    )

    market = market_validation_score(
        int(sig.get("saved_by_recruiters_30d", 0) or 0),
        p["market_validation_max_saves_30d"],
    )

    reliability = _clip01(float(sig.get("interview_completion_rate", 0.0) or 0.0))

    verification = verification_score(
        sig.get("verified_email", False), sig.get("verified_phone", False), sig.get("linkedin_connected", False),
    )

    profile_investment = _clip01(float(sig.get("profile_completeness_score", 50.0) or 0.0) / 100.0)

    weighted_mean = _clip01(
        w["recency"] * rec + w["responsiveness"] * resp + w["availability"] * avail
        + w["market_validation"] * market + w["reliability"] * reliability
        + w["verification"] * verification + w["profile_investment"] * profile_investment
    )

    floor = p["rescale_floor"]
    ceiling = p["rescale_ceiling"]
    multiplier = floor + weighted_mean * (ceiling - floor)

    return {
        "recency": round(rec, 4),
        "responsiveness": round(resp, 4),
        "availability": round(avail, 4),
        "market_validation": round(market, 4),
        "reliability": round(reliability, 4),
        "verification": round(verification, 4),
        "profile_investment": round(profile_investment, 4),
        "weighted_mean": round(weighted_mean, 4),
        "multiplier": round(multiplier, 4),
        "days_since_active": days_since_active,
        "recruiter_response_rate": resp,
    }
